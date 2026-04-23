#!/usr/bin/env Rscript

# Average Treatment Effect on Patients' Outcome
# R script corresponding to AverageTreatmentEffect.ipynb

# Import required libraries.
suppressPackageStartupMessages({
  library(Matching)
})

# Load the dataset and provide contextual information on column types.
df <- read.csv("mydata.csv", stringsAsFactors = FALSE)

binary_columns <- c("X5", "W", "Y")
categorical_columns <- c("X6", "X8")
numeric_columns <- c("X1", "X2", "X3", "X4", "X7", "X9")

# Define covariates and treatment indicator.
covariates <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9")
treatment_indicator <- "W"

# Create preprocessing analogous to sklearn's ColumnTransformer:
# - standardize numeric columns
# - one-hot encode categorical columns
# - passthrough remaining columns
X_raw <- df[, covariates]

# Standard scaling for numeric columns
X_num_scaled <- scale(X_raw[, numeric_columns])

# One-hot encode categorical columns (drop intercept)
X_cat_ohe <- model.matrix(~ X6 + X8 - 1, data = X_raw)

# Passthrough columns (binary column X5)
X_passthrough <- as.matrix(X_raw[, setdiff(covariates, c(numeric_columns, categorical_columns)), drop = FALSE])

# Final covariate matrix for matching
X <- cbind(X_num_scaled, X_cat_ohe, X_passthrough)

# Configure vectors for outcome and treatment
Y <- df$Y
D <- df[[treatment_indicator]]

# Estimate average treatment effect (ATE) using nearest-neighbour matching
# (Abadie & Imbens style inference as implemented in Matching::Match).
match_fit <- Match(
  Y = Y,
  Tr = D,
  X = X,
  M = 1,
  estimand = "ATE"
)

# Display estimates and confidence interval.
print(summary(match_fit))

cat("\nPoint estimate (ATE):", match_fit$est, "\n")
cat("Standard error:", match_fit$se.standard, "\n")
cat("95% CI:", match_fit$est - 1.96 * match_fit$se.standard,
    "to", match_fit$est + 1.96 * match_fit$se.standard, "\n")
