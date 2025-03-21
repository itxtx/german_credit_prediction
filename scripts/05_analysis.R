# ======= 05_analysis.R =======
# This is the main script that sources and runs all other scripts
# for the German Credit Risk Analysis project.
# It executes the full workflow from data import to model comparison
# and provides a final analysis and conclusion.

# Record start time
start_time <- Sys.time()

# Display header
cat("\n")
cat("==================================================================\n")
cat("                 GERMAN CREDIT RISK ANALYSIS                      \n")
cat("==================================================================\n")
cat("Start time:", format(start_time, "%Y-%m-%d %H:%M:%S"), "\n\n")

# Create a function to run a script and measure its execution time
run_script <- function(script_path, script_name) {
  cat("\n")
  cat("------------------------------------------------------------------\n")
  cat("EXECUTING:", script_name, "\n")
  cat("------------------------------------------------------------------\n")
  
  # Record script start time
  script_start_time <- Sys.time()
  
  # Run the script
  result <- tryCatch({
    source(script_path)
    "SUCCESS"
  }, error = function(e) {
    cat("ERROR in", script_name, ":", e$message, "\n")
    "FAILED"
  }, warning = function(w) {
    cat("WARNING in", script_name, ":", w$message, "\n")
    "COMPLETED WITH WARNINGS"
  })
  
  # Record script end time and calculate duration
  script_end_time <- Sys.time()
  script_duration <- script_end_time - script_start_time
  
  cat("\n")
  cat(script_name, "execution:", result, "\n")
  cat("Duration:", format(script_duration, digits = 2), "\n")
  cat("------------------------------------------------------------------\n")
  
  return(list(
    script = script_name,
    result = result,
    duration = script_duration
  ))
}

# Initialize execution log
execution_log <- list()

# Step 1: Set up environment
cat("\nSetting up environment and loading utilities...\n")
execution_log$setup <- run_script("scripts/utils/setup.R", "setup.R")
execution_log$preprocessing <- run_script("scripts/utils/preprocessing.R", "preprocessing.R")
execution_log$evaluation <- run_script("scripts/utils/evaluation.R", "evaluation.R")

# Step 2: Data Import and Preprocessing
cat("\nRunning data import and preprocessing...\n")
execution_log$data_import <- run_script("scripts/01_data_import.R", "01_data_import.R")
execution_log$data_preprocessing <- run_script("scripts/02_data_preprocessing.R", "02_data_preprocessing.R")

# Step 3: Train and evaluate all models
cat("\nTraining and evaluating models...\n")
model_scripts <- list(
  logistic_regression = "scripts/03_models/logistic_regression.R",
  naive_bayes = "scripts/03_models/naive_bayes.R",
  decision_tree = "scripts/03_models/decision_tree.R",
  random_forest = "scripts/03_models/random_forest.R",
  xgboost = "scripts/03_models/xgboost_model.R",
  svm = "scripts/03_models/svm_model.R"
)

# Run each model script
for(model_name in names(model_scripts)) {
  execution_log[[model_name]] <- run_script(model_scripts[[model_name]], basename(model_scripts[[model_name]]))
}

# Step 4: Model Comparison
cat("\nComparing models...\n")
execution_log$model_comparison <- run_script("scripts/04_model_comparison.R", "04_model_comparison.R")

# Record end time
end_time <- Sys.time()
total_duration <- end_time - start_time

# Generate summary of execution
cat("\n")
cat("==================================================================\n")
cat("                   EXECUTION SUMMARY                              \n")
cat("==================================================================\n")

# Create a summary table of execution times
summary_table <- data.frame(
  Script = character(),
  Result = character(),
  Duration = numeric(),
  stringsAsFactors = FALSE
)

for(item in execution_log) {
  summary_table <- rbind(summary_table, data.frame(
    Script = item$script,
    Result = item$result,
    Duration = as.numeric(item$duration, units = "secs"),
    stringsAsFactors = FALSE
  ))
}

# Sort by duration
summary_table <- summary_table[order(summary_table$Duration, decreasing = TRUE), ]

# Print summary table
print(summary_table)

cat("\nTotal execution time:", format(total_duration, digits = 2), "\n")

# Check if all scripts succeeded
all_success <- all(sapply(execution_log, function(x) x$result == "SUCCESS"))
if(all_success) {
  cat("All scripts executed successfully!\n")
} else {
  cat("Some scripts failed or completed with warnings. Check the logs for details.\n")
}

# Final Analysis and Conclusion
cat("\n")
cat("==================================================================\n")
cat("                   FINAL ANALYSIS AND CONCLUSION                  \n")
cat("==================================================================\n")

# Try to load best model info if available
best_model_name <- tryCatch({
  readLines("results/model_comparison/best_model_name.txt")[1]
}, error = function(e) {
  cat("Could not load best model name:", e$message, "\n")
  NULL
})

comparison_table <- tryCatch({
  # Check if comparison_results is already in environment
  if(exists("comparison_results") && !is.null(comparison_results$comparison_table)) {
    comparison_results$comparison_table
  } else {
    # Try to recreate it
    all_results <- load_model_results()
    create_comparison_table(all_results)
  }
}, error = function(e) {
  cat("Could not load comparison table:", e$message, "\n")
  NULL
})

# Print key findings
cat("\n=== Key Findings ===\n\n")

if(!is.null(best_model_name) && !is.null(comparison_table)) {
  # Find best model in comparison table
  best_model_row <- comparison_table[comparison_table$Model == best_model_name, ]
  
  cat("Best Model: ", gsub("_", " ", best_model_name), "\n")
  
  # Print metrics for best model
  metrics <- c("accuracy", "precision", "recall", "f1", "auc")
  for(metric in metrics) {
    if(metric %in% names(best_model_row) && !is.na(best_model_row[[metric]])) {
      cat(toupper(substr(metric, 1, 1)), substr(metric, 2, nchar(metric)), ": ", 
          round(best_model_row[[metric]], 4), "\n")
    }
  }
} else {
  cat("Best model information not available.\n")
}

# Load and analyze feature importance if available
features_importance <- tryCatch({
  if(best_model_name == "random_forest" || best_model_name == "xgboost") {
    if(best_model_name == "random_forest") {
      imp_file <- "results/models/random_forest/variable_importance.csv"
      if(file.exists(imp_file)) {
        read.csv(imp_file)
      } else {
        NULL
      }
    } else if(best_model_name == "xgboost") {
      imp_file <- "results/models/xgboost/feature_importance.csv"
      if(file.exists(imp_file)) {
        read.csv(imp_file)
      } else {
        NULL
      }
    }
  } else {
    NULL
  }
}, error = function(e) {
  cat("Could not load feature importance:", e$message, "\n")
  NULL
})

if(!is.null(features_importance) && nrow(features_importance) > 0) {
  cat("\n=== Top 5 Important Features ===\n\n")
  top_features <- head(features_importance, 5)
  for(i in 1:nrow(top_features)) {
    cat(i, ". ", top_features$Variable[i], " (", round(top_features$Importance[i], 4), ")\n", sep = "")
  }
}

# Generate comprehensive conclusion
cat("\n=== Conclusion ===\n\n")

cat("This German Credit Risk Analysis project systematically evaluated multiple machine learning\n")
cat("models to predict credit risk. The analysis pipeline included data preprocessing, model\n")
cat("training with cross-validation, and comprehensive performance evaluation.\n\n")

if(!is.null(best_model_name)) {
  cat("The ", gsub("_", " ", best_model_name), " model achieved the best overall performance,\n")
  cat("indicating its suitability for credit risk prediction on this dataset. ")
  
  # Add model-specific insights
  if(best_model_name == "random_forest" || best_model_name == "xgboost") {
    cat("The success of this ensemble method\n")
    cat("highlights the importance of capturing complex relationships and interactions between\n")
    cat("features in credit risk assessment.\n\n")
  } else if(best_model_name == "logistic_regression") {
    cat("The effectiveness of this\n")
    cat("linear model suggests that credit risk in this dataset may have strong linear\n")
    cat("relationships with the predictors.\n\n")
  } else if(best_model_name == "svm") {
    cat("The strength of SVM in this\n")
    cat("context demonstrates the value of margin-based approaches in separating good and bad\n")
    cat("credit risks.\n\n")
  } else {
    cat("\n\n")
  }
} else {
  cat("Model comparison results were not available for the final analysis. For detailed\n")
  cat("performance metrics, please refer to the model comparison report.\n\n")
}

if(!is.null(features_importance) && nrow(features_importance) > 0) {
  cat("Feature importance analysis revealed that variables such as ")
  cat(paste(head(features_importance$Variable, 3), collapse = ", "))
  cat("\nare particularly influential in predicting credit risk. These findings align with\n")
  cat("financial domain knowledge, where factors like checking account status, loan duration,\n")
  cat("and credit history are known to be strong predictors of default risk.\n\n")
}

cat("Recommendations for practical implementation:\n\n")

cat("1. Model Selection: ")
if(!is.null(best_model_name)) {
  cat("Deploy the ", gsub("_", " ", best_model_name), " model for credit risk assessment,\n")
  cat("   but consider maintaining alternatives for robustness.\n")
} else {
  cat("Select a model based on both predictive performance and\n")
  cat("   interpretability requirements for the specific use case.\n")
}

cat("2. Feature Focus: Prioritize data collection and quality assurance for the most\n")
cat("   influential features identified in the analysis.\n")

cat("3. Model Monitoring: Implement routine monitoring to detect concept drift and\n")
cat("   ensure model performance remains stable over time.\n")

cat("4. Balanced Approach: Consider business costs of false positives vs. false negatives\n")
cat("   when setting classification thresholds in production.\n\n")

cat("Future work could explore more advanced techniques such as neural networks,\n")
cat("stacked ensembles, or automated machine learning approaches. Additionally,\n")
cat("incorporating more domain-specific features or external data sources might\n")
cat("further enhance predictive performance.\n\n")

cat("This analysis provides a solid foundation for credit risk modeling and\n")
cat("demonstrates the effectiveness of machine learning approaches in this\n")
cat("financial application domain.\n")

# Save summary and conclusion to file
output_dir <- "results"
if(!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Create summary file
summary_file <- file.path(output_dir, "analysis_summary.md")
sink(summary_file)

cat("# German Credit Risk Analysis - Summary Report\n\n")
cat("Date: ", format(Sys.Date(), "%B %d, %Y"), "\n\n")

cat("## Execution Summary\n\n")
cat("| Script | Result | Duration (seconds) |\n")
cat("|---|---|---|\n")
for(i in 1:nrow(summary_table)) {
  cat("| ", summary_table$Script[i], " | ", summary_table$Result[i], " | ", 
      round(summary_table$Duration[i], 2), " |\n")
}

cat("\n\nTotal execution time: ", format(total_duration, digits = 2), "\n\n")

cat("## Key Findings\n\n")

if(!is.null(best_model_name) && !is.null(comparison_table)) {
  # Find best model in comparison table
  best_model_row <- comparison_table[comparison_table$Model == best_model_name, ]
  
  cat("**Best Model:** ", gsub("_", " ", best_model_name), "\n\n")
  
  cat("**Performance Metrics:**\n\n")
  cat("| Metric | Value |\n")
  cat("|---|---|\n")
  
  # Print metrics for best model
  metrics <- c("accuracy", "precision", "recall", "f1", "auc")
  for(metric in metrics) {
    if(metric %in% names(best_model_row) && !is.na(best_model_row[[metric]])) {
      cat("| ", toupper(substr(metric, 1, 1)), substr(metric, 2, nchar(metric)), " | ", 
          round(best_model_row[[metric]], 4), " |\n")
    }
  }
  cat("\n")
}

if(!is.null(features_importance) && nrow(features_importance) > 0) {
  cat("**Top 5 Important Features:**\n\n")
  cat("| Rank | Feature | Importance |\n")
  cat("|---|---|---|\n")
  top_features <- head(features_importance, 5)
  for(i in 1:nrow(top_features)) {
    cat("| ", i, " | ", top_features$Variable[i], " | ", 
        round(top_features$Importance[i], 4), " |\n")
  }
  cat("\n")
}

cat("## Conclusion\n\n")

cat("This German Credit Risk Analysis project systematically evaluated multiple machine learning ")
cat("models to predict credit risk. The analysis pipeline included data preprocessing, model ")
cat("training with cross-validation, and comprehensive performance evaluation.\n\n")

if(!is.null(best_model_name)) {
  cat("The ", gsub("_", " ", best_model_name), " model achieved the best overall performance, ")
  cat("indicating its suitability for credit risk prediction on this dataset. ")
  
  # Add model-specific insights
  if(best_model_name == "random_forest" || best_model_name == "xgboost") {
    cat("The success of this ensemble method ")
    cat("highlights the importance of capturing complex relationships and interactions between ")
    cat("features in credit risk assessment.\n\n")
  } else if(best_model_name == "logistic_regression") {
    cat("The effectiveness of this ")
    cat("linear model suggests that credit risk in this dataset may have strong linear ")
    cat("relationships with the predictors.\n\n")
  } else if(best_model_name == "svm") {
    cat("The strength of SVM in this ")
    cat("context demonstrates the value of margin-based approaches in separating good and bad ")
    cat("credit risks.\n\n")
  } else {
    cat("\n\n")
  }
} else {
  cat("Model comparison results were not available for the final analysis. For detailed ")
  cat("performance metrics, please refer to the model comparison report.\n\n")
}

if(!is.null(features_importance) && nrow(features_importance) > 0) {
  cat("Feature importance analysis revealed that variables such as ")
  cat(paste(head(features_importance$Variable, 3), collapse = ", "))
  cat(" are particularly influential in predicting credit risk. These findings align with ")
  cat("financial domain knowledge, where factors like checking account status, loan duration, ")
  cat("and credit history are known to be strong predictors of default risk.\n\n")
}

cat("**Recommendations for practical implementation:**\n\n")

cat("1. **Model Selection:** ")
if(!is.null(best_model_name)) {
  cat("Deploy the ", gsub("_", " ", best_model_name), " model for credit risk assessment, ")
  cat("but consider maintaining alternatives for robustness.\n")
} else {
  cat("Select a model based on both predictive performance and ")
  cat("interpretability requirements for the specific use case.\n")
}

cat("2. **Feature Focus:** Prioritize data collection and quality assurance for the most ")
cat("influential features identified in the analysis.\n")

cat("3. **Model Monitoring:** Implement routine monitoring to detect concept drift and ")
cat("ensure model performance remains stable over time.\n")

cat("4. **Balanced Approach:** Consider business costs of false positives vs. false negatives ")
cat("when setting classification thresholds in production.\n\n")

cat("Future work could explore more advanced techniques such as neural networks, ")
cat("stacked ensembles, or automated machine learning approaches. Additionally, ")
cat("incorporating more domain-specific features or external data sources might ")
cat("further enhance predictive performance.\n\n")

cat("This analysis provides a solid foundation for credit risk modeling and ")
cat("demonstrates the effectiveness of machine learning approaches in this ")
cat("financial application domain.\n")

# Close the file
sink()

cat("\nSummary report saved to:", summary_file, "\n")

cat("\n")
cat("==================================================================\n")
cat("                   ANALYSIS COMPLETED                             \n")
cat("==================================================================\n")
cat("End time:", format(end_time, "%Y-%m-%d %H:%M:%S"), "\n")
cat("Total duration:", format(total_duration, digits = 2), "\n\n")