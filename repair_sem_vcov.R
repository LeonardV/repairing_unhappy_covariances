# =============================================================================
# Title   : Generic SEM covariance matrix repair (with ML-based detection)
# Author  : Leonard Vanbrabant (Ghent University / GGD West-Brabant)
# Date    : 2026-03-19
# Purpose : Full pipeline for repairing non-converging SEM covariance matrices.
#           For each new SEM model: (1) fit SEM on adversarial VCOVs to get
#           convergence labels, (2) train XGBoost on those labels,
#           (3) use SHAP to identify destabilizing correlations, (4) repair.
#
#           The adversarial VCOVs are MODEL-INDEPENDENT (only depend on p).
#           Generate them once per dimensionality and reuse across models.
#
# Usage   :
#   source("repair_sem_vcov.R")
#
#   # Step 1: Load pre-generated adversarial VCOVs
#   load("R_list.RData")
#
#   # Step 2: Train a detector (only relabels convergence + trains XGBoost)
#   detector <- train_detector(
#     model  = my_model,
#     R_list = R_list,
#     nobs   = 30
#   )
#
#   # Step 3: Repair using the trained detector
#   result <- repair_sem_vcov(
#     data     = my_data,
#     model    = my_model,
#     detector = detector
#   )
#
#   # For a DIFFERENT SEM model with the same p, reuse R_list:
#   detector2 <- train_detector(model = other_model, R_list = R_list)
# =============================================================================


# =============================================================================
# Dependencies
# =============================================================================

for (pkg in c("lavaan", "Matrix", "data.table", "xgboost")) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
library(lavaan)
library(Matrix)
library(data.table)
library(xgboost)


# =============================================================================
# 1. Core utilities
# =============================================================================

extract_ov_names <- function(model) {
  pt <- lavaan::lavaanify(model)
  lv <- unique(pt$lhs[pt$op == "=~"])
  rhs_ind <- unique(pt$rhs[pt$op == "=~"])
  rhs_reg <- unique(pt$rhs[pt$op == "~"])
  lhs_reg <- unique(pt$lhs[pt$op == "~"])
  all_vars <- unique(c(rhs_ind, rhs_reg, lhs_reg))
  setdiff(all_vars, c(lv, ""))
}

safe_cov2cor <- function(S) {
  d <- diag(S)
  d[!is.finite(d) | d <= 0] <- 1e-8
  inv_sd <- 1 / sqrt(d)
  inv_sd[!is.finite(inv_sd)] <- 1e8
  D <- diag(inv_sd)
  R <- D %*% S %*% D
  R[!is.finite(R)] <- 0
  diag(R) <- 1
  R
}

sanitize_R <- function(R) {
  R[!is.finite(R)] <- 0
  R <- (R + t(R)) / 2
  diag(R) <- 1
  R
}

min_eig <- function(R) {
  min(eigen(R, symmetric = TRUE, only.values = TRUE)$values)
}

sigmoid_weight <- function(rho, center = 0.65, sharpness = 10,
                           w_min = 0.00, w_max = 0.30) {
  s <- 1 / (1 + exp(-sharpness * (abs(rho) - center)))
  w_min + (w_max - w_min) * s
}

fisher_z <- function(r, eps = 1e-6) {
  r <- pmin(pmax(r, -1 + eps), 1 - eps)
  0.5 * log((1 + r) / (1 - r))
}


# =============================================================================
# 2. SEM fitting (generic)
# =============================================================================

fit_sem_on_R <- function(R, model, nobs) {
  fit <- tryCatch(
    lavaan::sem(
      model = model, sample.cov = R,
      sample.nobs = nobs, sample.cov.rescale = FALSE,
      warn = FALSE, control = list(iter.max = 200)
    ),
    error = function(e) NULL
  )
  if (is.null(fit)) {
    return(list(fit = NULL, converged = FALSE,
                post_check = "fitting error", iterations = NA_integer_))
  }
  post_check_msg <- tryCatch(
    lavaan:::lav_object_post_check(fit),
    error = function(e) "post-check error"
  )
  pe <- lavaan::parameterEstimates(fit)
  has_na_se <- any(is.na(pe$se))
  list(
    fit        = fit,
    converged  = isTRUE(lavaan::inspect(fit, "converged")) && !has_na_se,
    post_check = post_check_msg,
    iterations = lavaan::lavInspect(fit, "iterations")
  )
}


# =============================================================================
# 5. Feature extraction
# =============================================================================

build_feature_schema <- function(varnames) {
  p <- length(varnames)
  idx <- which(lower.tri(matrix(0, p, p)), arr.ind = TRUE)
  paste0(varnames[idx[, 1]], "_", varnames[idx[, 2]])
}

build_feature_matrix <- function(R_list, schema) {
  mat <- do.call(rbind, lapply(R_list, function(R) {
    v <- R[lower.tri(R)]
    names(v) <- schema
    v
  }))
  mat
}

R_to_one_case <- function(R, schema) {
  v <- R[lower.tri(R)]
  names(v) <- schema
  v
}


# =============================================================================
# 6. SEM batch fitting (sequential, iter.max=200)
# =============================================================================

fit_sem_batch <- function(R_list, model, nobs, verbose = TRUE) {
  n <- length(R_list)
  labels <- character(n)
  t0 <- proc.time()["elapsed"]
  
  for (i in seq_len(n)) {
    fit <- tryCatch(
      lavaan::sem(model = model, sample.cov = R_list[[i]],
                  sample.nobs = nobs, sample.cov.rescale = FALSE,
                  warn = FALSE, control = list(iter.max = 200)),
      error = function(e) NULL
    )
    if (is.null(fit)) { labels[i] <- "nonconverged"; next }
    pe <- tryCatch(lavaan::parameterEstimates(fit), error = function(e) NULL)
    if (is.null(pe)) { labels[i] <- "nonconverged"; next }
    ok <- isTRUE(lavaan::inspect(fit, "converged")) && !any(is.na(pe$se))
    labels[i] <- if (ok) "converged" else "nonconverged"
    
    if (verbose && (i %% 100 == 0 || i == n)) {
      elapsed <- proc.time()["elapsed"] - t0
      rate <- elapsed / i
      eta <- rate * (n - i)
      message(sprintf("  %d/%d (%.0f%%) | %.0f sec elapsed | ETA: %.0f sec",
                      i, n, 100*i/n, elapsed, eta))
    }
  }
  
  if (verbose) {
    elapsed <- proc.time()["elapsed"] - t0
    n_c <- sum(labels == "converged")
    n_nc <- sum(labels == "nonconverged")
    message(sprintf("  Done in %.1f sec | converged: %d | nonconverged: %d",
                    elapsed, n_c, n_nc))
  }
  labels
}


# =============================================================================
# 7. XGBoost training
#    Uses fixed eta = 0.1 with default hyperparameters. Only the optimal
#    number of boosting rounds is determined via 5-fold CV with early stopping.
# =============================================================================

train_xgb_detector <- function(X, y, schema, seed = 42, verbose = TRUE) {
  set.seed(seed)
  X_z <- apply(X, 2, fisher_z)
  
  spw <- sum(y == 0L) / max(sum(y == 1L), 1L)
  
  params <- list(
    objective        = "binary:logistic",
    eval_metric      = "logloss",
    eta              = 0.1,
    scale_pos_weight = spw,
    tree_method      = "hist"
  )
  
  dall <- xgb.DMatrix(X_z, label = y)
  
  # 5-fold CV to find optimal nrounds (only tuning step needed)
  cv <- xgb.cv(
    params = params, data = dall, nrounds = 500, nfold = 5,
    early_stopping_rounds = 15, verbose = 0, stratified = TRUE
  )
  best_nr <- cv$best_iteration
  best_ll <- min(cv$evaluation_log$test_logloss_mean)
  
  if (verbose) message(sprintf("  CV logloss: %.4f | optimal nrounds: %d",
                               best_ll, best_nr))
  
  # Final model on all data
  final_model <- xgb.train(
    params = params, data = dall,
    nrounds = best_nr, verbose = 0
  )
  
  # Global SHAP distribution
  shap_mat <- predict(final_model, dall, predcontrib = TRUE)
  shap_mat <- shap_mat[, -ncol(shap_mat), drop = FALSE]
  global_dist <- as.numeric(abs(shap_mat))
  
  list(
    model = final_model, global_dist = global_dist,
    best_params = params, best_nrounds = best_nr
  )
}


# =============================================================================
# 8. SHAP-based diagnostics
# =============================================================================

vec_to_corr <- function(v, schema, varnames = NULL) {
  if (is.null(varnames)) varnames <- sort(unique(unlist(strsplit(schema, "_"))))
  p <- length(varnames)
  R <- diag(1, p); colnames(R) <- rownames(R) <- varnames
  for (nm in schema) {
    ij <- strsplit(nm, "_")[[1]]
    i <- match(ij[1], varnames); j <- match(ij[2], varnames)
    R[i, j] <- v[nm]; R[j, i] <- v[nm]
  }
  R
}

explain_vcov_case <- function(final_model, new_row, schema,
                              cal_fun = NULL, use_fisher_z = TRUE,
                              top_n = 20, varnames = NULL) {
  if (is.data.frame(new_row) || data.table::is.data.table(new_row)) {
    x_raw <- as.numeric(new_row[1, schema, drop = TRUE])
  } else {
    x_raw <- as.numeric(new_row[schema])
  }
  names(x_raw) <- schema
  
  x_in <- if (use_fisher_z) fisher_z(x_raw) else x_raw
  dm <- xgboost::xgb.DMatrix(
    data = matrix(x_in, nrow = 1, dimnames = list(NULL, schema))
  )
  
  p_raw <- as.numeric(predict(final_model, dm))
  p_cal <- if (!is.null(cal_fun)) pmin(pmax(cal_fun(p_raw), 0), 1) else p_raw
  
  shap_mat <- predict(final_model, dm, predcontrib = TRUE)
  bias <- shap_mat[1, ncol(shap_mat)]
  shap_vals <- shap_mat[1, -ncol(shap_mat), drop = TRUE]
  
  shap_dt <- data.table(
    feature = schema, r_value = x_raw, model_input = x_in,
    shap_logit = as.numeric(shap_vals)
  )
  shap_dt[, c("var1", "var2") := data.table::tstrsplit(feature, "_")]
  shap_dt[, abs_shap_logit := abs(shap_logit)]
  
  base_prob <- stats::plogis(bias)
  shap_dt[, shap_prob_approx := stats::plogis(bias + shap_logit) - base_prob]
  shap_dt[, abs_shap_prob := abs(shap_prob_approx)]
  
  tot_abs <- sum(shap_dt$abs_shap_logit)
  shap_dt[, perc_logit := abs_shap_logit / tot_abs]
  
  data.table::setorder(shap_dt, -abs_shap_logit)
  shap_top <- shap_dt[1:min(top_n, .N)]
  
  if (is.null(varnames)) varnames <- sort(unique(c(shap_dt$var1, shap_dt$var2)))
  R_case <- vec_to_corr(x_raw, schema = schema, varnames = varnames)
  
  p_dim <- length(varnames)
  SHAP_mat <- matrix(0, p_dim, p_dim, dimnames = list(varnames, varnames))
  for (i in seq_len(nrow(shap_dt))) {
    SHAP_mat[shap_dt$var1[i], shap_dt$var2[i]] <- shap_dt$shap_logit[i]
    SHAP_mat[shap_dt$var2[i], shap_dt$var1[i]] <- shap_dt$shap_logit[i]
  }
  
  list(
    p_raw = p_raw, p_cal = p_cal, bias = bias,
    shap_table = shap_dt, shap_top = shap_top,
    R_case = R_case, SHAP_mat = SHAP_mat, varnames = varnames
  )
}

detect_anchors <- function(res_case, global_dist,
                           min_shap = 0, min_anchor_strength = 0.05,
                           min_percentile = 0.50, max_anchors = 10) {
  dt <- data.table::copy(res_case$shap_table)
  dt <- dt[shap_logit > min_shap]
  if (nrow(dt) == 0) return(dt[0])
  
  dt[, percentile := stats::ecdf(global_dist)(abs_shap_logit)]
  dt[, anchor_strength := shap_logit * percentile]
  dt <- dt[percentile >= min_percentile & anchor_strength >= min_anchor_strength]
  if (nrow(dt) == 0) return(dt[0])
  
  data.table::setorder(dt, -anchor_strength)
  if (!is.null(max_anchors) && nrow(dt) > max_anchors) dt <- dt[1:max_anchors]
  
  dt[, .(feature, var1, var2, shap_logit, abs_shap_logit, percentile, anchor_strength)]
}


# =============================================================================
# 9. Target correlation matrix construction
# =============================================================================

build_target_R <- function(model, S, nobs, cor = 0.2, reliability = 0.8) {
  fit_dummy <- lavaan::sem(model, sample.cov = S, sample.nobs = nobs, do.fit = FALSE)
  g <- 1L
  S_g <- fit_dummy@SampleStats@cov[[g]]
  lambda.idx <- which(names(fit_dummy@Model@GLIST) == "lambda")[g]
  LAMBDA   <- fit_dummy@Model@GLIST[[lambda.idx]]
  ov.names <- fit_dummy@Model@dimNames[[lambda.idx]][[1]]
  nvar <- nrow(LAMBDA); nfac <- ncol(LAMBDA)
  
  Lambda1 <- LAMBDA
  Lambda1[fit_dummy@Model@m.free.idx[[lambda.idx]]] <- 1
  
  COR.ss <- stats::cov2cor(t(Lambda1) %*% S_g %*% Lambda1)
  small.idx <- which(abs(COR.ss) < 0.1)
  if (length(small.idx) > 0L) COR.ss[small.idx] <- 0
  
  COR.u <- matrix(cor, nfac, nfac); diag(COR.u) <- 1
  VETA <- COR.u * sign(COR.ss)
  L07 <- Lambda1 * 0.7
  
  if (reliability >= 1.0) {
    THETA <- matrix(0, nvar, nvar)
  } else {
    tmp <- diag(L07 %*% VETA %*% t(L07))
    td <- tmp / reliability - tmp
    if (any(td <= 0)) { warning("Adjusting non-positive residual variances."); td[td <= 0] <- 0.01 }
    THETA <- diag(td)
  }
  
  Sigma <- L07 %*% VETA %*% t(L07) + THETA
  S.sd <- sqrt(diag(S_g))
  Sigma.r <- t(stats::cov2cor(Sigma) * S.sd) * S.sd
  R_target <- stats::cov2cor(Sigma.r)
  rownames(R_target) <- colnames(R_target) <- ov.names
  R_target
}


# =============================================================================
# 10. Heuristic suspect detection (fallback when no detector)
# =============================================================================

detect_suspects_heuristic <- function(R_obs, R_target,
                                      max_suspects = 10, min_residual = 0.01) {
  p <- nrow(R_obs); varnames <- colnames(R_obs)
  idx <- which(lower.tri(R_obs), arr.ind = TRUE)
  dt <- data.table(
    var1 = varnames[idx[,1]], var2 = varnames[idx[,2]],
    r_obs = R_obs[lower.tri(R_obs)], r_target = R_target[lower.tri(R_target)]
  )
  dt[, residual := abs(r_obs - r_target)]
  dt <- dt[residual >= min_residual]
  if (nrow(dt) == 0L) return(dt[0])
  dt[, percentile := rank(residual) / .N]
  data.table::setorder(dt, -residual)
  if (!is.null(max_suspects) && nrow(dt) > max_suspects) dt <- dt[1:max_suspects]
  dt
}


# =============================================================================
# 11. Delta computation with smeared updates
# =============================================================================

compute_delta_R <- function(R_obs, R_target, suspects, lambda0,
                            varnames = colnames(R_obs),
                            smear_center = 0.65, smear_sharpness = 10,
                            eps = 1e-3, cap_step = Inf) {
  p <- nrow(R_obs)
  Delta <- matrix(0, p, p); Weight <- matrix(0, p, p)
  if (is.null(suspects) || nrow(suspects) == 0L)
    return(list(Delta = Delta, Weight = Weight))
  
  add_sym <- function(M, i, j, val) {
    M[i,j] <- M[i,j] + val; M[j,i] <- M[j,i] + val; M
  }
  clamp <- function(x, cap) { if (is.infinite(cap)) x else pmax(pmin(x, cap), -cap) }
  
  for (k in seq_len(nrow(suspects))) {
    i <- match(suspects$var1[k], varnames)
    j <- match(suspects$var2[k], varnames)
    if (is.na(i) || is.na(j) || i == j) next
    w_base <- suspects$percentile[k]
    step_ij <- clamp(R_target[i,j] - R_obs[i,j], cap_step)
    Delta <- add_sym(Delta, i, j, lambda0 * w_base * step_ij)
    Weight <- add_sym(Weight, i, j, w_base)
    for (m in seq_len(p)) {
      if (m == i || m == j) next
      w_i <- sigmoid_weight(R_obs[i,m], center=smear_center, sharpness=smear_sharpness)
      if (w_i > eps) {
        wi <- w_i * w_base
        Delta <- add_sym(Delta, i, m, lambda0 * wi * clamp(R_target[i,m]-R_obs[i,m], cap_step))
        Weight <- add_sym(Weight, i, m, wi)
      }
      w_j <- sigmoid_weight(R_obs[j,m], center=smear_center, sharpness=smear_sharpness)
      if (w_j > eps) {
        wj <- w_j * w_base
        Delta <- add_sym(Delta, j, m, lambda0 * wj * clamp(R_target[j,m]-R_obs[j,m], cap_step))
        Weight <- add_sym(Weight, j, m, wj)
      }
    }
  }
  list(Delta = Delta, Weight = Weight)
}


# =============================================================================
# 12. Repair engine: forward search + bisection
# =============================================================================

repair_with_forwardsearch_then_bisect <- function(
    R_obs, R_target, suspects, model, nobs,
    varnames = colnames(R_obs), lambda_min = 1e-4, lambda_max = 0.5,
    grow1 = 1.45, grow2 = 1.60, switch_at = 1e-1, max_forward = 40,
    tol = 1e-4, max_bisect = 20, eig_tol = -1e-8,
    fallback_nearPD = TRUE, verbose = FALSE,
    smear_center = 0.60, smear_sharpness = 8, eps = 1e-3, cap_step = Inf) {
  
  build_R_try <- function(lambda) {
    d <- compute_delta_R(R_obs, R_target, suspects, lambda, varnames,
                         smear_center, smear_sharpness, eps, cap_step)
    R_pre <- R_obs; idx <- d$Weight > 0
    R_pre[idx] <- R_obs[idx] + d$Delta[idx] / pmax(d$Weight[idx], 1e-12)
    diag(R_pre) <- 1
    list(R_pre = R_pre, delta_obj = d)
  }
  
  is_success <- function(R_pre) {
    me <- min_eig(R_pre)
    if (me <= eig_tol) return(list(ok = FALSE, min_eig = me))
    s <- fit_sem_on_R(R_pre, model, nobs)
    list(ok = isTRUE(s$converged), min_eig = me)
  }
  
  if (is.null(suspects) || nrow(suspects) == 0L) {
    s0 <- fit_sem_on_R(R_obs, model, nobs); me0 <- min_eig(R_obs)
    return(list(R_rep=R_obs, ls_success=isTRUE(s0$converged), used_nearPD=FALSE,
                lambda_used=0, min_eig_prePD=me0, min_eig_postPD=me0,
                nearPD_extra_max=0, nearPD_extra_mean=0,
                sem_converged_postPD=isTRUE(s0$converged)))
  }
  
  lambda <- lambda_min
  low_l <- NA_real_; high_l <- NA_real_
  low_R <- NULL; high_R <- NULL
  low_me <- NA_real_; high_me <- NA_real_
  
  for (k in seq_len(max_forward)) {
    obj <- build_R_try(lambda); chk <- is_success(obj$R_pre)
    if (verbose) message(sprintf("FWD %02d | l=%.4g | eig=%.2e | ok=%s", k, lambda, chk$min_eig, chk$ok))
    if (chk$ok) { high_l <- lambda; high_R <- obj$R_pre; high_me <- chk$min_eig; break }
    low_l <- lambda; low_R <- obj$R_pre; low_me <- chk$min_eig
    lambda <- lambda * (if (lambda < switch_at) grow1 else grow2)
    if (lambda > lambda_max) break
  }
  
  if (is.na(high_l)) {
    if (!fallback_nearPD || is.null(low_R))
      return(list(R_rep=R_obs, ls_success=FALSE, used_nearPD=FALSE,
                  lambda_used=NA, min_eig_prePD=NA, min_eig_postPD=NA,
                  nearPD_extra_max=NA, nearPD_extra_mean=NA, sem_converged_postPD=FALSE))
    R_post <- as.matrix(Matrix::nearPD(low_R, keepDiag=TRUE)$mat); diag(R_post) <- 1
    D_pd <- R_post - low_R; off <- lower.tri(D_pd)
    sp <- fit_sem_on_R(R_post, model, nobs)
    return(list(R_rep=R_post, ls_success=FALSE, used_nearPD=TRUE, lambda_used=low_l,
                min_eig_prePD=low_me, min_eig_postPD=min_eig(R_post),
                nearPD_extra_max=max(abs(D_pd[off])), nearPD_extra_mean=mean(abs(D_pd[off])),
                sem_converged_postPD=isTRUE(sp$converged)))
  }
  
  if (is.na(low_l))
    return(list(R_rep=high_R, ls_success=TRUE, used_nearPD=FALSE, lambda_used=high_l,
                min_eig_prePD=high_me, min_eig_postPD=high_me,
                nearPD_extra_max=0, nearPD_extra_mean=0, sem_converged_postPD=TRUE))
  
  lo <- low_l; hi <- high_l
  best_l <- high_l; best_R <- high_R; best_me <- high_me
  
  for (b in seq_len(max_bisect)) {
    if ((hi - lo) <= max(tol, tol * hi)) break
    mid <- 0.5 * (lo + hi); obj <- build_R_try(mid); chk <- is_success(obj$R_pre)
    if (verbose) message(sprintf("BIS %02d | lo=%.3g hi=%.3g mid=%.3g | ok=%s", b, lo, hi, mid, chk$ok))
    if (chk$ok) { hi <- mid; best_l <- mid; best_R <- obj$R_pre; best_me <- chk$min_eig }
    else lo <- mid
  }
  
  list(R_rep=best_R, ls_success=TRUE, used_nearPD=FALSE, lambda_used=best_l,
       min_eig_prePD=best_me, min_eig_postPD=best_me,
       nearPD_extra_max=0, nearPD_extra_mean=0, sem_converged_postPD=TRUE)
}


# =============================================================================
# 13. Jackknife bounds (optional validation)
# =============================================================================

jackknife_cor_bounds <- function(Data, probs = c(0.025, 0.975)) {
  X <- as.matrix(Data); n <- nrow(X); p <- ncol(X)
  stopifnot(n >= 3)
  s <- colSums(X); T_mat <- crossprod(X)
  R_array <- array(NA_real_, dim = c(p, p, n))
  for (i in 1:n) {
    xi <- X[i,]; n1 <- n-1L; s_i <- s-xi; mu_i <- s_i/n1
    T_i <- T_mat - tcrossprod(xi)
    R_array[,,i] <- stats::cov2cor((T_i - n1*tcrossprod(mu_i))/(n-2))
  }
  R_lo <- apply(R_array, c(1,2), quantile, probs=probs[1], na.rm=TRUE)
  R_hi <- apply(R_array, c(1,2), quantile, probs=probs[2], na.rm=TRUE)
  diag(R_lo) <- 1; diag(R_hi) <- 1
  list(R_lower=(R_lo+t(R_lo))/2, R_upper=(R_hi+t(R_hi))/2)
}


# =============================================================================
# 14. train_detector() -- train XGBoost for a specific SEM model
#     Accepts pre-generated R_list to avoid regenerating adversarial VCOVs.
#     Only the convergence labels (y) change per SEM model; X stays the same.
# =============================================================================

train_detector <- function(
    model,
    R_list,
    nobs           = 200,
    seed           = 42,
    verbose        = TRUE
) {
  # Extract observed variable names and dimensionality from model
  ov_names <- extract_ov_names(model)
  p <- length(ov_names)
  if (verbose) message(sprintf("Model has %d observed variables: %s",
                               p, paste(ov_names, collapse = ", ")))
  
  # Validate pre-generated VCOVs
  stopifnot(!is.null(R_list), is.list(R_list), length(R_list) > 0)
  p_adv <- nrow(R_list[[1]])
  if (p_adv != p) {
    stop(sprintf("R_list dimensionality (%d) does not match model (%d observed variables).",
                 p_adv, p))
  }
  if (verbose) message(sprintf("[1/3] Using %d pre-generated VCOVs (p=%d)",
                               length(R_list), p_adv))
  
  # Rename columns to match model variable names
  R_list <- lapply(R_list, function(R) {
    colnames(R) <- rownames(R) <- ov_names; R
  })
  
  # Fit SEM on each VCOV -- this is the model-specific step
  if (verbose) message("[2/3] Fitting SEM on adversarial VCOVs (model-specific)...")
  labels <- fit_sem_batch(R_list, model, nobs, verbose = verbose)
  y <- as.integer(labels == "nonconverged")
  
  if (verbose) {
    n_nc <- sum(y == 1); n_c <- sum(y == 0)
    message(sprintf("  Converged: %d (%.1f%%) | Non-converged: %d (%.1f%%)",
                    n_c, 100*n_c/length(y), n_nc, 100*n_nc/length(y)))
  }
  
  if (sum(y == 1) < 10 || sum(y == 0) < 10) {
    warning("Extreme class imbalance -- detector may not train well. ",
            "Consider adjusting severity_probs or n_sims.")
  }
  
  # Build features (X does NOT change across models -- same lower-triangle values)
  schema <- build_feature_schema(ov_names)
  X <- build_feature_matrix(R_list, schema)
  
  if (verbose) message(sprintf("[3/3] Training XGBoost (%d features, 5-fold CV)...",
                               length(schema)))
  xgb_result <- train_xgb_detector(X, y, schema,
                                   seed = seed, verbose = verbose)
  
  detector <- list(
    model         = xgb_result$model,
    schema        = schema,
    global_dist   = xgb_result$global_dist,
    ov_names      = ov_names,
    n_sims        = length(R_list),
    nobs          = nobs,
    best_params   = xgb_result$best_params,
    class_balance = table(labels)
  )
  
  if (verbose) message("Detector training complete.")
  detector
}


# =============================================================================
# 15. repair_sem_vcov() -- main entry point
# =============================================================================

repair_sem_vcov <- function(
    data               = NULL,
    sample_cov         = NULL,
    sample_nobs        = NULL,
    model,
    detector           = NULL,
    target_cor         = 0.2,
    target_reliability = 0.8,
    max_suspects       = 10,
    check_bounds       = FALSE,
    verbose            = TRUE,
    ...
) {
  # --- Input validation ---
  if (is.null(data) && is.null(sample_cov))
    stop("Provide either 'data' or 'sample_cov' + 'sample_nobs'.")
  
  if (!is.null(data)) {
    data <- as.data.frame(data)
    S <- cov(data); nobs <- nrow(data); R_obs <- cor(data)
  } else {
    stopifnot(!is.null(sample_nobs))
    S <- sample_cov; nobs <- sample_nobs; R_obs <- stats::cov2cor(S)
  }
  
  varnames <- colnames(R_obs)
  if (is.null(varnames)) {
    varnames <- paste0("V", seq_len(ncol(R_obs)))
    colnames(R_obs) <- rownames(R_obs) <- varnames
    colnames(S) <- rownames(S) <- varnames
  }
  
  # --- 1. Check observed convergence ---
  if (verbose) message("[1/5] Fitting SEM on observed correlation matrix...")
  sem0 <- fit_sem_on_R(R_obs, model, nobs)
  
  if (sem0$converged) {
    if (verbose) message("  >> SEM already converges. No repair needed.")
    return(list(R_repaired=R_obs, S_repaired=S, R_observed=R_obs,
                converged_before=TRUE, converged_after=TRUE,
                repair_applied=FALSE, fit_before=sem0))
  }
  if (verbose) message("  >> SEM does NOT converge. Starting repair pipeline...")
  
  # --- 2. Build target ---
  if (verbose) message("[2/5] Constructing target correlation matrix...")
  R_target <- tryCatch(
    build_target_R(model, S, nobs, cor=target_cor, reliability=target_reliability),
    error = function(e) {
      if (verbose) message("  >> Target failed: ", e$message, " -> nearPD fallback.")
      R_pd <- as.matrix(Matrix::nearPD(R_obs, keepDiag=TRUE)$mat)
      diag(R_pd) <- 1; rownames(R_pd) <- colnames(R_pd) <- varnames; R_pd
    }
  )
  
  # --- 3. Identify suspects ---
  if (verbose) message("[3/5] Detecting suspect correlations...")
  if (!is.null(detector)) {
    # ML/SHAP-based detection
    if (verbose) message("  >> Using trained detector (SHAP-based)")
    one_case <- R_to_one_case(R_obs, detector$schema)
    res_case <- explain_vcov_case(
      final_model = detector$model, new_row = one_case,
      schema = detector$schema, varnames = varnames
    )
    suspects <- detect_anchors(res_case, detector$global_dist,
                               max_anchors = max_suspects)
    if (verbose) message(sprintf("  >> p(nonconv) = %.3f | %d anchor(s) detected",
                                 res_case$p_raw, nrow(suspects)))
  } else {
    # Heuristic fallback
    if (verbose) message("  >> No detector provided -> heuristic fallback")
    suspects <- detect_suspects_heuristic(R_obs, R_target,
                                          max_suspects = max_suspects)
  }
  
  if (verbose && nrow(suspects) > 0) {
    message(sprintf("  >> Top suspect: %s <-> %s",
                    suspects$var1[1], suspects$var2[1]))
  }
  
  # --- 4. Repair ---
  if (verbose) message("[4/5] Running forward-search + bisection repair...")
  rep_result <- repair_with_forwardsearch_then_bisect(
    R_obs=R_obs, R_target=R_target, suspects=suspects,
    model=model, nobs=nobs, verbose=verbose, ...
  )
  R_rep <- rep_result$R_rep
  
  # --- 5. Verify convergence ---
  if (verbose) message("[5/5] Verifying convergence on repaired matrix...")
  sem1 <- fit_sem_on_R(R_rep, model, nobs)
  if (verbose) message(sprintf("  >> Repair %s | lambda=%.4g | nearPD=%s",
                               if(sem1$converged) "SUCCEEDED" else "FAILED",
                               rep_result$lambda_used, rep_result$used_nearPD))
  
  # --- Diagnostics ---
  D <- R_rep - R_obs; idx <- lower.tri(D)
  diagnostics <- list(
    delta_max=max(abs(D[idx])), delta_mean=mean(abs(D[idx])),
    delta_frob=sqrt(sum(D[idx]^2)), lambda_used=rep_result$lambda_used,
    used_nearPD=rep_result$used_nearPD,
    min_eig_obs=min_eig(R_obs), min_eig_rep=min_eig(R_rep)
  )
  
  # --- Optional: jackknife bounds ---
  bounds_check <- NULL
  if (check_bounds && !is.null(data)) {
    jk <- jackknife_cor_bounds(data)
    within <- (R_rep >= jk$R_lower) & (R_rep <= jk$R_upper); diag(within) <- TRUE
    bounds_check <- list(R_lower=jk$R_lower, R_upper=jk$R_upper,
                         within_bounds=within,
                         pct_within=mean(within[lower.tri(within)]))
    if (verbose) message(sprintf("  >> %.1f%% within jackknife bounds.",
                                 100*bounds_check$pct_within))
  }
  
  # --- Rescale to covariance ---
  sd_vec <- sqrt(diag(S))
  S_rep <- t(R_rep * sd_vec) * sd_vec
  rownames(S_rep) <- colnames(S_rep) <- varnames
  
  list(
    R_repaired=R_rep, S_repaired=S_rep, R_observed=R_obs, R_target=R_target,
    converged_before=FALSE, converged_after=sem1$converged,
    repair_applied=TRUE, suspects=suspects, diagnostics=diagnostics,
    bounds_check=bounds_check, fit_before=sem0, fit_after=sem1
  )
}


t_total_done <- proc.time()["elapsed"] - t_total
cat(sprintf("\n========== TOTAL PIPELINE TIME: %.1f sec ==========\n", t_total_done))
