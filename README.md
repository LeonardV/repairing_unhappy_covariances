# Local Repair of Unhappy Covariances
R toolkit for repairing non-converging SEM covariance matrices using XGBoost-guided SHAP diagnostics.

When structural equation models fail to converge on observed covariance matrices, this tool identifies the unhappy covariances and applies minimal, targeted corrections. The pipeline trains a binary classifier on adversarial covariance matrices, uses SHAP values to pinpoint problematic covariances, and repairs the matrix via forward-search with bisection — preserving as much of the original structure as possible.

## How it works

1. **Adversarial VCOV generation** — Generate perturbed covariance matrices that span the space from well-behaved to pathological. These are model-independent (only depend on the number of observed variables *p*) and can be reused across different SEM specifications. Once generated, the adversarial VCOVs do not need to be regenerated unless new pathology types are added to the perturbation scheme.

2. **Convergence labelling** — Fit the target SEM model on each adversarial VCOV and label it as converged or non-converged.

3. **XGBoost detector** — Train a binary classifier on Fisher-z transformed lower-triangle correlations (features) with convergence labels (target). Uses 5-fold CV with early stopping to find the optimal number of boosting rounds (`eta = 0.1`, all other hyperparameters at defaults).

4. **SHAP anchor detection** — For a given non-converging matrix, compute SHAP values to identify which correlation pairs contribute most to predicted non-convergence. These "unhappy correlations" are the repair targets.

5. **Forward-search + bisection repair** — Incrementally adjust the suspect correlations toward a theory-informed target matrix, using the smallest step size (lambda) that restores convergence. Falls back to `nearPD` projection if needed.

## Installation

```r
# Install dependencies
install.packages(c("lavaan", "Matrix", "data.table", "xgboost"))
```

## Quick start

```r
source("repair_sem_vcov.R")

# Step 1: Load pre-generated adversarial VCOVs (generated once per p)
load("R_list_p6.RData")

# Step 2: Define your SEM model
my_model <- '
  Y =~ V1 + V2 + V3
  X =~ V4 + V5 + V6
  Y ~ X
'

# Step 3: Train detector (labels + XGBoost, model-specific)
detector <- train_detector(
  model  = my_model,
  R_list = R_list,
  nobs   = 30
)

# Example VCOV with unhappy covariances
R_unhappy <- matrix(
  c(
    2.8951255,  1.09782955,  0.43006788, -0.3735530, -0.20881548, -0.1563188,
    1.0978296,  1.80545776,  0.03860447, -0.7993932,  0.03764216, -0.3784308,
    0.4300679,  0.03860447,  1.32534142, -0.3897087, -0.09359977, -0.2777123,
   -0.3735530, -0.79939319, -0.38970872,  2.3855418,  0.38998205,  0.6353415,
   -0.2088155,  0.03764216, -0.09359977,  0.3899820,  1.24926755,  0.3003272,
   -0.1563188, -0.37843082, -0.27771225,  0.6353415,  0.30032718,  1.2999353
  ),
  nrow = 6,
  byrow = TRUE,
  dimnames = list(
    c("V1", "V2", "V3", "V4", "V5", "V6"),
    c("V1", "V2", "V3", "V4", "V5", "V6")
  )
)

# Step 4: Locally repair a non-converging covariance matrix
result <- repair_sem_vcov(
  sample_cov  = R_unhappy,
  sample_nobs = 30,
  model       = my_model,
  detector    = detector,
  verbose     = TRUE
)

# Inspect results
cat("\n========== RESULTS ==========\n")
  cat(sprintf("Converged BEFORE repair: %s\n", result$converged_before))
  cat(sprintf("Converged AFTER  repair: %s\n", result$converged_after))
  cat(sprintf("Repair time: %.2f sec\n", t2_done))
  cat(sprintf("Lambda used: %.4g\n", result$diagnostics$lambda_used))
  cat(sprintf("Used nearPD fallback: %s\n", result$diagnostics$used_nearPD))
  cat(sprintf("Delta max: %.4f | Delta mean: %.4f\n",
              result$diagnostics$delta_max, result$diagnostics$delta_mean))
  cat(sprintf("Min eigenvalue: obs=%.4f | repaired=%.4f\n",
              result$diagnostics$min_eig_obs, result$diagnostics$min_eig_rep))

  if (nrow(result$suspects) > 0) {
    cat("\nTop suspects (SHAP anchors):\n")
    print(result$suspects[, .(feature, shap_logit, anchor_strength)])
  }
}
```

## Generating adversarial VCOVs for a different *p*

The included `R_list_p6.RData` contains 50,000 adversarial covariance matrices for *p* = 6. If your SEM model has a different number of observed variables, generate a new set:

```r
source("adversarial_vcov_generator.R")

sim_res <- run_vcov_sim(
  n_sims = 50000,
  p = 10 # set to your number of observed variables
)

R_list <- sim_res$R_list
save(R_list, file = "R_list_p10.RData")
```

This only needs to be done once per dimensionality. The resulting `R_list` can be reused across all SEM models with that same *p*, and does not need to be regenerated unless new pathology types are added to the perturbation scheme.

## Reuse across models

The adversarial VCOVs only depend on dimensionality (*p*), not on the SEM specification. Generate them once and reuse:

```r
# Same R_list, different model
other_model <- '
  F1 =~ V1 + V2 + V3
  F2 =~ V4 + V5 + V6
  F1 ~~ F2
'
detector2 <- train_detector(model = other_model, R_list = R_list, nobs = 200)
```

## Key features

- **Model-agnostic adversarial generation** — VCOVs reusable across SEM specifications with the same *p*; only regenerate when new pathology types are added
- **No hyperparameter tuning** — Fixed `eta = 0.1` with XGBoost defaults; only `nrounds` is determined via 5-fold CV
- **SHAP-based anchor detection** — Interpretable identification of destabilizing correlations
- **Minimal repair** — Forward-search + bisection finds the smallest correction that restores convergence
- **nearPD fallback** — Guaranteed positive-definite output even when targeted repair is insufficient
- **Jackknife bounds** — Optional validation that repaired correlations stay within sampling variability

## Dependencies

- [lavaan](https://lavaan.ugent.be/) — SEM fitting
- [xgboost](https://xgboost.readthedocs.io/) — Gradient boosting classifier
- [Matrix](https://cran.r-project.org/package=Matrix) — nearPD projection
- [data.table](https://rdatatable.gitlab.io/data.table/) — Fast data manipulation

## Author

Leonard Vanbrabant (Ghent University / GGD West-Brabant)

## License

This project is licensed under the [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html).
