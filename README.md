# Local Repair of Unhappy Covariances
R toolkit for repairing non-converging SEM covariance matrices using XGBoost-guided SHAP diagnostics.

When structural equation models fail to converge on observed covariance matrices, this tool identifies the destabilizing correlations and applies minimal, targeted corrections. The pipeline trains a binary classifier on adversarial covariance matrices, uses SHAP values to pinpoint problematic correlation pairs, and repairs the matrix via forward-search with bisection — preserving as much of the original structure as possible.

## How it works

1. **Adversarial VCOV generation** — Generate perturbed covariance matrices that span the space from well-behaved to pathological. These are model-independent (only depend on the number of observed variables *p*) and can be reused across different SEM specifications. Once generated, the adversarial VCOVs do not need to be regenerated unless new pathology types are added to the perturbation scheme.

2. **Convergence labelling** — Fit the target SEM model on each adversarial VCOV and label it as converged or non-converged.

3. **XGBoost detector** — Train a binary classifier on Fisher-z transformed lower-triangle correlations (features) with convergence labels (target). Uses 5-fold CV with early stopping to find the optimal number of boosting rounds (`eta = 0.1`, all other hyperparameters at defaults).

4. **SHAP anchor detection** — For a given non-converging matrix, compute SHAP values to identify which correlation pairs contribute most to predicted non-convergence. These "anchors" are the repair targets.

5. **Forward-search + bisection repair** — Incrementally adjust the suspect correlations toward a theory-informed target matrix, using the smallest step size (lambda) that restores convergence. Falls back to `nearPD` projection if needed.

## Installation

```r
# Install dependencies
install.packages(c("lavaan", "Matrix", "data.table", "xgboost"))

# Source the toolkit
source("repair_sem_vcov.R")
```

## Quick start

```r
source("repair_sem_vcov.R")

# Step 1: Load pre-generated adversarial VCOVs (generated once per p)
load("R_list.RData")

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
  nobs   = 200
)

# Step 4: Repair a non-converging covariance matrix
result <- repair_sem_vcov(
  data     = my_data,
  model    = my_model,
  detector = detector
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
