# Binary Classifier to Predict Outcome Y with 5-fold Cross-Validation

# Import required libraries
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(rsample)
  library(recipes)
  library(workflows)
  library(parsnip)
  library(tune)
  library(yardstick)
  library(themis)
  library(dials)
})

# Load the dataset
# Assumes mydata.csv is in the current working directory.
df <- read_csv("mydata.csv", show_col_types = FALSE)

# Contextual information
binary_columns <- c("X5", "W", "Y")
categorical_columns <- c("X6", "X8")
numeric_columns <- c("X1", "X2", "X3", "X4", "X7", "X9")

# Split into features and target
# Keep column names aligned with the notebook.
X <- df %>% select(-Y)
y <- df %>% pull(Y)

# Ensure outcome has classification semantics for metrics and downsampling.
df <- df %>%
  mutate(
    Y = factor(as.character(Y), levels = c("0", "1")),
    across(all_of(categorical_columns), as.factor)
  )

# Preprocessing pipeline:
# - standard scaling of numeric columns
# - one-hot encoding of categorical columns
# - passthrough for remaining columns
# - random undersampling of the majority class
preprocessor <- recipe(Y ~ ., data = df) %>%
  step_normalize(all_of(numeric_columns)) %>%
  step_dummy(all_of(categorical_columns), one_hot = TRUE) %>%
  step_downsample(Y)

# Gradient boosting classifier (close equivalent to sklearn's GradientBoostingClassifier)
classifier_spec <- boost_tree(mode = "classification") %>%
  set_engine("gbm")

model_workflow <- workflow() %>%
  add_recipe(preprocessor) %>%
  add_model(classifier_spec)

# 5-fold cross-validation
set.seed(1)
folds <- vfold_cv(df, v = 5, strata = Y)

# Metrics to match notebook output
metric_map <- list(
  balanced_accuracy = metric_set(bal_accuracy),
  accuracy = metric_set(accuracy),
  f1 = metric_set(f_meas),
  roc_auc = metric_set(roc_auc)
)

# Run CV and print mean +/- std. dev. per metric
for (metric_name in names(metric_map)) {
  cv_results <- fit_resamples(
    model_workflow,
    resamples = folds,
    metrics = metric_map[[metric_name]],
    control = control_resamples(save_pred = TRUE)
  )

  metric_tbl <- collect_metrics(cv_results) %>%
    filter(.metric %in% c("bal_accuracy", "accuracy", "f_meas", "roc_auc"))

  mean_score <- metric_tbl$mean[[1]]
  std_score <- metric_tbl$std_err[[1]] * sqrt(nrow(folds$splits))

  cat(
    paste0(
      metric_name,
      " mean +/- std. dev.: ",
      sprintf("%.3f", mean_score),
      " +/- ",
      sprintf("%.3f", std_score),
      "\n"
    )
  )
}
