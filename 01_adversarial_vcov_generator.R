# =============================================================================
# Title   : Simulation of clean and pathological correlation matrices
# Author  : Leonard Vanbrabant (Ghent University/GGD West-Brabant)
# Date    : 2026-03-16
# Purpose : Generate correlation-matrix templates, inject realistic pathologies,
#           compute diagnostics, and run large-scale simulation studies.
# =============================================================================


# =============================================================================
# Helper functions
# =============================================================================

# Convert a covariance-like matrix to a correlation matrix in a numerically
# stable way. Non-finite or non-positive diagonal entries are replaced by a
# small positive constant before rescaling.
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

# Sanitize a correlation matrix by:
# 1. replacing non-finite entries with 0,
# 2. enforcing symmetry,
# 3. resetting the diagonal to 1.
sanitize_R <- function(R) {
  R[!is.finite(R)] <- 0
  R <- (R + t(R)) / 2
  diag(R) <- 1
  
  R
}


# =============================================================================
# Template generation
# =============================================================================

# Generate a baseline correlation matrix from one of several template families.
#
# Args:
#   p        : Number of variables.
#   template : Template family used to generate the matrix.
#
# Returns:
#   A p x p correlation matrix.
generate_R_template <- function(
    p,
    template = c(
      "wishart", "block", "toeplitz",
      "lowrank", "uniform_random"
    )) {
  
  template <- match.arg(template)
  
  # Random positive-definite covariance matrix based on Gaussian distributions,
  # subsequently rescaled to a correlation matrix.
  if (template == "wishart") {
    A <- matrix(rnorm(p * p), p, p)
    S <- crossprod(A)
    return(safe_cov2cor(S))
  }
  
  # Block-structured correlation matrix with stronger within-block dependence
  # and weaker random between-block dependence.
  if (template == "block") {
    K <- sample(2:min(4, p), 1)
    cuts <- sort(sample(1:(p - 1), K - 1))
    block_id <- cut(1:p, breaks = c(0, cuts, p), labels = FALSE)
    
    R <- matrix(0, p, p)
    
    for (k in seq_len(K)) {
      members <- which(block_id == k)
      rho_k <- runif(1, 0.2, 0.7)
      R[members, members] <- rho_k
    }
    
    R[R == 0] <- runif(sum(R == 0), -0.3, 0.3)
    diag(R) <- 1
    
    return(R)
  }
  
  # Toeplitz-like structure with correlations that decay as variables become
  # further apart in index distance.
  if (template == "toeplitz") {
    lags <- 1:(p - 1)
    rhos <- runif(length(lags), -0.6, 0.6) *
      exp(-runif(1, 0.1, 0.6) * lags)
    
    R <- matrix(0, p, p)
    
    for (i in 1:p) {
      for (j in 1:p) {
        if (i == j) {
          R[i, j] <- 1
        } else {
          k <- abs(i - j)
          R[i, j] <- rhos[k]
        }
      }
    }
    
    return(R)
  }
  
  # Low-rank covariance structure with added uniquenesses, then rescaled to
  # a correlation matrix.
  if (template == "lowrank") {
    r <- sample(1:(p - 1), 1)
    L <- matrix(rnorm(p * r), p, r)
    S <- L %*% t(L) + diag(runif(p, 0.05, 0.15))
    
    return(safe_cov2cor(S))
  }
  
  # Fully random symmetric matrix, then rescaled to a correlation matrix.
  if (template == "uniform_random") {
    A <- matrix(runif(p * p, -1, 1), p, p)
    S <- (A + t(A)) / 2
    diag(S) <- 1
    
    return(safe_cov2cor(S))
  }
}


# =============================================================================
# Unhappy injection
# =============================================================================

# Inject near-singularity by shrinking the smallest eigenvalue.
#
# Severity interpretation:
#   1 = mild     : 1e-2 to 1e-1
#   2 = moderate : 1e-3 to 1e-2
#   3 = severe   : 1e-4 to 1e-3
#   4 = extreme  : 1e-5 to 1e-4
inject_near_singular <- function(R, severity = 2) {
  ev <- eigen(R, symmetric = TRUE)
  k <- length(ev$values)  # Smallest eigenvalue is stored last.
  
  target <- switch(
    as.character(severity),
    "1" = 10^runif(1, -2, -1),
    "2" = 10^runif(1, -3, -2),
    "3" = 10^runif(1, -4, -3),
    "4" = 10^runif(1, -5, -4),
    10^runif(1, -3, -2)   # Default: moderate
  )
  
  ev$values[k] <- target
  R2 <- ev$vectors %*% diag(ev$values) %*% t(ev$vectors)
  
  safe_cov2cor(R2)
}

# Inject indefiniteness by making the smallest eigenvalue negative.
#
# Severity interpretation:
#   1 = mild     : -0.01 to -0.05
#   2 = moderate : -0.05 to -0.10
#   3 = severe   : -0.10 to -0.25
#   4 = extreme  : -0.25 to -0.50
inject_indefinite <- function(R, severity = 2) {
  ev <- eigen(R, symmetric = TRUE)
  k <- length(ev$values)
  
  neg_val <- switch(
    as.character(severity),
    "1" = -runif(1, 0.01, 0.05),
    "2" = -runif(1, 0.05, 0.10),
    "3" = -runif(1, 0.10, 0.25),
    "4" = -runif(1, 0.25, 0.50),
    -runif(1, 0.05, 0.10)  # Default: moderate
  )
  
  ev$values[k] <- neg_val
  R2 <- ev$vectors %*% diag(ev$values) %*% t(ev$vectors)
  
  sanitize_R(R2)
}

# Inject a local cluster of high correlations among a subset of variables.
#
# Severity interpretation:
#   1 = mild     : 0.60 to 0.80
#   2 = moderate : 0.75 to 0.90
#   3 = severe   : 0.85 to 0.95
#   4 = extreme  : 0.95 to 0.98
inject_corr_cluster <- function(R, p, severity = 2) {
  k <- sample(2:min(5, p), 1)
  idx <- sample(1:p, k)
  
  vals <- switch(
    as.character(severity),
    "1" = runif(k * (k - 1) / 2, 0.60, 0.80),
    "2" = runif(k * (k - 1) / 2, 0.75, 0.90),
    "3" = runif(k * (k - 1) / 2, 0.85, 0.95),
    "4" = runif(k * (k - 1) / 2, 0.95, 0.98),
    runif(k * (k - 1) / 2, 0.75, 0.90)  # Default: moderate
  )
  
  sub <- diag(1, k)
  sub[upper.tri(sub)] <- vals
  sub[lower.tri(sub)] <- vals
  
  R[idx, idx] <- sub
  
  sanitize_R(R)
}

# Inject mixed pathologies by combining a correlation cluster with either
# near-singularity or indefiniteness, depending on the severity level.
inject_mixed <- function(R, p, severity = 2) {
  sev <- as.integer(severity)
  sev <- max(1L, min(4L, sev))
  
  if (sev == 1L) {
    R <- inject_corr_cluster(R, p, severity = 1)
    R <- inject_near_singular(R, severity = 1)
  } else if (sev == 2L) {
    R <- inject_corr_cluster(R, p, severity = 2)
    R <- inject_near_singular(R, severity = 2)
  } else if (sev == 3L) {
    R <- inject_corr_cluster(R, p, severity = 3)
    R <- inject_indefinite(R, severity = 1)
  } else if (sev == 4L) {
    R <- inject_corr_cluster(R, p, severity = 3)
    R <- inject_indefinite(R, severity = 2)
    R <- inject_near_singular(R, severity = 3)
  }
  
  safe_cov2cor(R)
}


# =============================================================================
# Diagnostics
# =============================================================================

# Compute a compact set of diagnostic measures for a correlation matrix.
#
# Returns:
#   A named list with:
#   - min_eig       : Smallest eigenvalue
#   - max_eig       : Largest eigenvalue
#   - cond          : Condition number based on positive eigenvalues only
#   - max_abs_cor   : Maximum absolute off-diagonal correlation
#   - neg_eig_count : Number of negative eigenvalues
#   - dist_nearPD   : Frobenius distance to the nearest positive-definite matrix
diagnose_R <- function(R) {
  R <- sanitize_R((R + t(R)) / 2)
  
  eig <- try(
    eigen(R, symmetric = TRUE, only.values = TRUE)$values,
    silent = TRUE
  )
  
  if (inherits(eig, "try-error")) {
    return(list(
      min_eig = NA_real_,
      max_eig = NA_real_,
      cond = Inf,
      max_abs_cor = NA_real_,
      neg_eig_count = NA_integer_,
      dist_nearPD = NA_real_
    ))
  }
  
  min_ev <- min(eig)
  max_ev <- max(eig)
  
  pos <- eig[eig > 0]
  pos_min <- if (length(pos) == 0L) NA_real_ else min(pos)
  cond <- if (is.na(pos_min)) Inf else max_ev / pos_min
  
  R_pd <- as.matrix(Matrix::nearPD(R, keepDiag = TRUE)$mat)
  dist_pd <- norm(R - R_pd, type = "F")
  
  off <- R[upper.tri(R)]
  
  list(
    min_eig = min_ev,
    max_eig = max_ev,
    cond = cond,
    max_abs_cor = max(abs(off)),
    neg_eig_count = sum(eig < 0),
    dist_nearPD = dist_pd
  )
}


# =============================================================================
# Main simulation function
# =============================================================================

# Simulate a collection of clean and pathological correlation matrices.
#
# Args:
#   n_sims         : Number of simulated matrices.
#   p              : Number of variables.
#   seed           : Random seed for reproducibility.
#   templates      : Template families used as baseline generators.
#   severity_probs : Named vector with probabilities for the severity classes:
#                    clean, mild, moderate, severe, extreme.
#
# Returns:
#   A list containing:
#   - R_list          : Simulated correlation matrices
#   - R_vech          : Lower-triangle values as a matrix (n_sims x p*(p-1)/2)
#   - diag_df         : Diagnostic summary per matrix
#   - severity_label  : Severity class per matrix
#   - pathology_label : Pathology type per matrix
run_vcov_sim <- function(
    n_sims = 5000,
    p = 6,
    seed = 42,
    templates = c(
      "wishart", "block", "toeplitz",
      "lowrank", "uniform_random"
    ),
    severity_probs = c(
      clean = 0.45,
      mild = 0.25,
      moderate = 0.15,
      severe = 0.10,
      extreme = 0.05
    )) {
  
  set.seed(seed)
  templates <- match.arg(templates, several.ok = TRUE)
  
  severity_levels <- c("clean", "mild", "moderate", "severe", "extreme")
  
  if (is.null(names(severity_probs))) {
    stop(
      "severity_probs must be a named vector with names: ",
      paste(severity_levels, collapse = ", ")
    )
  }
  
  severity_probs <- severity_probs[severity_levels]
  
  if (any(is.na(severity_probs))) {
    stop(
      "severity_probs must include entries for all of: ",
      paste(severity_levels, collapse = ", ")
    )
  }
  
  severity_probs <- severity_probs / sum(severity_probs)
  
  severity_to_int <- function(sev) {
    switch(
      sev,
      "clean" = 0L,
      "mild" = 1L,
      "moderate" = 2L,
      "severe" = 3L,
      "extreme" = 4L
    )
  }
  
  varnames <- paste0("V", seq_len(p))
  
  R_list <- vector("list", n_sims)
  diag_list <- vector("list", n_sims)
  labels_sev <- character(n_sims)
  labels_type <- character(n_sims)
  
  # Conservative rule set to define a genuinely "clean" matrix.
  # Thresholds for max_eig and cond scale with p to avoid rejecting
  # legitimate matrices at higher dimensionalities.
  is_clean_R <- function(d) {
    d$min_eig > 1e-3 &&
      d$cond < max(200, 50 * p) &&
      d$max_eig < max(5, 0.7 * p) &&
      d$max_abs_cor < 0.70 &&
      d$dist_nearPD < 1e-4 &&
      d$neg_eig_count == 0
  }
  
  for (i in seq_len(n_sims)) {
    severity <- sample(severity_levels, 1, prob = severity_probs)
    sev_int <- severity_to_int(severity)
    pathology_type <- "none"
    
    tpl <- sample(templates, 1)
    R <- generate_R_template(p, tpl)
    
    # -------------------------------------------------------------------------
    # Clean matrices
    # -------------------------------------------------------------------------
    if (severity == "clean") {
      templates_clean <- intersect(templates, c("wishart", "block", "toeplitz"))
      if (length(templates_clean) == 0L) {
        templates_clean <- templates
      }
      
      max_attempts <- 50L
      ok <- FALSE
      
      for (attempt in seq_len(max_attempts)) {
        tpl_clean <- sample(templates_clean, 1)
        R0 <- generate_R_template(p, tpl_clean)
        R0 <- sanitize_R(R0)
        d0 <- diagnose_R(R0)
        
        if (is_clean_R(d0)) {
          R <- R0
          ok <- TRUE
          break
        }
      }
      
      # Fallback: repair the current matrix to the nearest positive-definite
      # matrix if no sufficiently clean candidate was found.
      if (!ok) {
        R_pd <- Matrix::nearPD(R, keepDiag = TRUE)$mat
        R <- safe_cov2cor(as.matrix(R_pd))
      }
      
      pathology_type <- "none"
    }
    
    # -------------------------------------------------------------------------
    # Mild pathology
    # -------------------------------------------------------------------------
    if (severity == "mild") {
      pathology_type <- sample(c("near_singular", "corr_cluster"), 1)
      
      if (pathology_type == "near_singular") {
        R <- inject_near_singular(R, severity = 1)
      } else if (pathology_type == "corr_cluster") {
        R <- inject_corr_cluster(R, p, severity = 1)
      }
    }
    
    # -------------------------------------------------------------------------
    # Moderate pathology
    # -------------------------------------------------------------------------
    if (severity == "moderate") {
      pathology_type <- sample(
        c("near_singular", "indefinite", "corr_cluster"), 1)
      
      if (pathology_type == "near_singular") {
        R <- inject_near_singular(R, severity = 2)
      } else if (pathology_type == "indefinite") {
        R <- inject_indefinite(R, severity = 1)
      } else if (pathology_type == "corr_cluster") {
        R <- inject_corr_cluster(R, p, severity = 2)
      }
    }
    
    # -------------------------------------------------------------------------
    # Severe pathology
    # -------------------------------------------------------------------------
    if (severity == "severe") {
      pathology_type <- sample(
        c("near_singular", "indefinite", "corr_cluster", "mixed"), 1)
      
      if (pathology_type == "near_singular") {
        R <- inject_near_singular(R, severity = 3)
      } else if (pathology_type == "indefinite") {
        R <- inject_indefinite(R, severity = 2)
      } else if (pathology_type == "corr_cluster") {
        R <- inject_corr_cluster(R, p, severity = 3)
      } else if (pathology_type == "mixed") {
        R <- inject_mixed(R, p, severity = 3)
      }
    }
    
    # -------------------------------------------------------------------------
    # Extreme pathology
    # -------------------------------------------------------------------------
    if (severity == "extreme") {
      pathology_type <- "mixed"
      R <- inject_mixed(R, p, severity = 4)
    }
    
    R <- sanitize_R(R)
    rownames(R) <- varnames
    colnames(R) <- varnames
    
    d <- diagnose_R(R)
    
    R_list[[i]] <- R
    diag_list[[i]] <- d
    labels_sev[i] <- severity
    labels_type[i] <- pathology_type
  }
  
  diag_df <- as.data.frame(do.call(rbind, lapply(diag_list, as.data.frame)))
  diag_df$severity_level <- labels_sev
  diag_df$pathology_type <- labels_type
  
  # Build R_vech: lower-triangle values as a matrix (n_sims x p*(p-1)/2)
  R_vech <- do.call(rbind, lapply(R_list, function(R) R[lower.tri(R)]))
  
  list(
    R_list = R_list,
    R_vech = R_vech,
    diag_df = diag_df,
    severity_label = labels_sev,
    pathology_label = labels_type
  )
}
