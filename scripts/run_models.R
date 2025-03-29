# Main workflow script to run all models
source("scripts/utils/setup.R")

# Load and preprocess data
source("scripts/02_data_preprocessing.R")

# Run all models
message("\n=== Running All Models ===\n")

# Decision Tree
source("scripts/03_models/decision_tree.R")
dt_results <- run_decision_tree(train_data, test_data)

# Naive Bayes
source("scripts/03_models/naive_bayes.R")
nb_results <- run_naive_bayes(train_data, test_data)

# XGBoost
source("scripts/03_models/xgboost.R")
xgb_results <- run_xgboost(train_data, test_data)

# SVM
source("scripts/03_models/svm.R")
svm_results <- run_svm(train_data, test_data)

# Compare model performances
compare_models <- function(dt_results, nb_results, xgb_results, svm_results) {
  models <- c("Decision Tree", "Naive Bayes", "XGBoost", "SVM")
  performances <- list(dt_results$performance, 
                      nb_results$performance, 
                      xgb_results$performance,
                      svm_results$performance)
  
  # Create comparison table
  metrics <- c("accuracy", "precision", "recall", "f1", "auc")
  comparison <- data.frame(Model = models)
  
  for(metric in metrics) {
    comparison[[metric]] <- sapply(performances, function(p) round(p[[metric]], 4))
  }
  
  return(comparison)
}

# Print model comparison
model_comparison <- compare_models(dt_results, nb_results, xgb_results, svm_results)
print(model_comparison)

# Save comparison results
write.csv(model_comparison, "results/model_comparison.csv", row.names = FALSE)
message("\nModel comparison saved to results/model_comparison.csv") 