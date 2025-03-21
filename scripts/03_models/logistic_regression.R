# ======= logistic_regression.R =======
# This script implements a logistic regression model for the German Credit dataset.
# The script handles:
# - Data preparation specific to logistic regression
# - Model training with cross-validation
# - Prediction generation
# - Performance evaluation

# Source utility scripts
source("scripts/utils/setup.R")
source("scripts/utils/evaluation.R")

# Load preprocessed data if not already in environment
if(!exists("train_data") || !exists("test_data")) {
  source("scripts/02_data_preprocessing.R")
}

# Function to prepare data specifically for logistic regression
prepare_for_logistic_regression <- function(train_data, test_data) {
  message("\n=== Preparing Data for Logistic Regression ===")
  
  # Create copies to avoid modifying the originals
  train_lr <- train_data
  test_lr <- test_data
  
  # For logistic regression, we should handle:
  # 1. Multicollinearity: Check for highly correlated predictors
  # 2. Potential dummy variable trap: One-hot encoding with care
  
  # Check for multicollinearity among numeric predictors
  numeric_cols <- names(train_lr)[sapply(train_lr, is.numeric)]
  if(length(numeric_cols) > 1) {
    message("Checking for multicollinearity among numeric predictors...")
    cor_matrix <- cor(train_lr[, numeric_cols], use = "pairwise.complete.obs")
    high_cor <- findCorrelation(cor_matrix, cutoff = 0.8)
    
    if(length(high_cor) > 0) {
      message("Found ", length(high_cor), " highly correlated predictors. Consider removing them for better model stability.")
      # Optionally, we could remove them:
      # train_lr <- train_lr[, -high_cor]
      # test_lr <- test_lr[, -high_cor]
    } else {
      message("No severe multicollinearity detected among numeric predictors.")
    }
  }
  
  # For factors with many levels, we might want to combine rare levels
  # This helps prevent perfect separation issues in logistic regression
  factor_cols <- names(train_lr)[sapply(train_lr, is.factor)]
  factor_cols <- setdiff(factor_cols, "class")  # Exclude target variable
  
  for(col in factor_cols) {
    # Get frequency table
    freq_table <- table(train_lr[[col]])
    rare_levels <- names(freq_table)[freq_table < 10]  # Levels with fewer than 10 observations
    
    if(length(rare_levels) > 0) {
      message("Column '", col, "' has ", length(rare_levels), " rare levels (< 10 observations)")
      
      # Optionally combine rare levels
      # train_lr[[col]] <- as.factor(ifelse(train_lr[[col]] %in% rare_levels, "OTHER", as.character(train_lr[[col]])))
      # test_lr[[col]] <- as.factor(ifelse(test_lr[[col]] %in% rare_levels, "OTHER", as.character(test_lr[[col]])))
    }
  }
  
  # Create a formula that includes all predictors except the class
  predictors <- setdiff(names(train_lr), "class")
  formula_string <- paste("class ~", paste(predictors, collapse = " + "))
  model_formula <- as.formula(formula_string)
  
  message("Created model formula with ", length(predictors), " predictors")
  
  return(list(
    train = train_lr,
    test = test_lr,
    formula = model_formula
  ))
}

# Function to train logistic regression model with cross-validation
train_logistic_regression <- function(prepared_data, k_folds = 5, seed_value = 123) {
  message("\n=== Training Logistic Regression Model with Cross-Validation ===")
  
  # Set seed for reproducibility
  set.seed(seed_value)
  
  # Extract prepared data
  train_data <- prepared_data$train
  model_formula <- prepared_data$formula
  
  # Set up cross-validation
  ctrl <- trainControl(
    method = "cv",            # Cross-validation
    number = k_folds,         # Number of folds
    classProbs = TRUE,        # Calculate class probabilities
    summaryFunction = twoClassSummary,  # Use ROC summary
    savePredictions = "final" # Save final predictions
  )
  
  # Train the model with error handling
  tryCatch({
    message("Training logistic regression model...")
    
    # Start timing
    start_time <- proc.time()
    
    # Train the model
    logistic_model <- train(
      model_formula, 
      data = train_data,
      method = "glm",
      family = "binomial",
      trControl = ctrl,
      metric = "ROC"
    )
    
    # End timing
    end_time <- proc.time()
    train_time <- end_time - start_time
    
    message("Model training completed in ", round(train_time[3], 2), " seconds")
    
    # Print model summary
    message("\nModel Summary:")
    print(logistic_model)
    
    # Print cross-validation results
    message("\nCross-Validation Results:")
    print(logistic_model$results)
    
    # Print variable importance if available
    if(!is.null(logistic_model$finalModel)) {
      message("\nVariable Coefficients:")
      coef_summary <- summary(logistic_model$finalModel)
      print(coef_summary$coefficients)
    }
    
    return(logistic_model)
    
  }, error = function(e) {
    message("ERROR training logistic regression: ", e$message)
    
    # Try with a simpler formula if the original fails
    message("Attempting with a simplified model...")
    
    # Create a simpler formula with fewer predictors
    simple_formula <- as.formula("class ~ checking_status + duration + credit_amount + age")
    
    # Train a simpler model
    simple_model <- train(
      simple_formula, 
      data = train_data,
      method = "glm",
      family = "binomial",
      trControl = ctrl,
      metric = "ROC"
    )
    
    message("Simplified model training completed")
    return(simple_model)
  })
}

# Function to generate predictions using the trained model
generate_predictions <- function(model, test_data) {
  message("\n=== Generating Predictions ===")
  
  # Generate class predictions
  pred_class <- predict(model, newdata = test_data)
  
  # Generate probability predictions
  pred_prob <- predict(model, newdata = test_data, type = "prob")
  
  # Extract probability for the positive class (assuming "Good" is positive)
  if("Good" %in% colnames(pred_prob)) {
    pos_class_prob <- pred_prob[, "Good"]
  } else {
    # If "Good" not found, use second column (typically the positive class in binary)
    pos_class_prob <- pred_prob[, 2]
    message("Used second probability column as positive class")
  }
  
  message("Generated predictions for ", length(pred_class), " test samples")
  
  return(list(
    class = pred_class,
    prob = pos_class_prob,
    all_probs = pred_prob
  ))
}

# Function to evaluate model performance
evaluate_logistic_regression <- function(predictions, actual, output_dir = "results/models/logistic_regression") {
  message("\n=== Evaluating Model Performance ===")
  
  # Create output directory if it doesn't exist
  if(!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    message("Created directory: ", output_dir)
  }
  
  # Evaluate model using the evaluation.R utility
  performance <- evaluate_model(
    pred = predictions$class,
    actual = actual,
    pred_prob = predictions$prob,
    positive_class = "Good"
  )
  
  # Print performance metrics
  message("\nPerformance Metrics:")
  metrics <- c("accuracy", "precision", "recall", "f1", "auc")
  for(metric in metrics) {
    if(!is.null(performance[[metric]])) {
      message(paste0(toupper(substring(metric, 1, 1)), substring(metric, 2), ": ", 
                    round(performance[[metric]], 4)))
    }
  }
  
  # Print confusion matrix
  message("\nConfusion Matrix:")
  print(performance$confusion_matrix$table)
  
  # Plot ROC curve
  png(file.path(output_dir, "roc_curve.png"), width = 800, height = 600)
  plot_roc_curve(
    pred_prob = predictions$prob,
    actual = actual,
    positive_class = "Good",
    model_name = "Logistic Regression"
  )
  dev.off()
  message("ROC curve saved to: ", file.path(output_dir, "roc_curve.png"))
  
  # Plot confusion matrix
  if(requireNamespace("ggplot2", quietly = TRUE)) {
    confusion_plot <- plot_confusion_matrix(
      performance$confusion_matrix,
      title = "Logistic Regression Confusion Matrix"
    )
    ggplot2::ggsave(
      file.path(output_dir, "confusion_matrix.png"),
      confusion_plot,
      width = 8,
      height = 6
    )
    message("Confusion matrix plot saved to: ", file.path(output_dir, "confusion_matrix.png"))
  }
  
  # If available, save variable importance plot
  if(requireNamespace("ggplot2", quietly = TRUE) && !is.null(logistic_model$finalModel)) {
    # Extract coefficients
    coefs <- coef(logistic_model$finalModel)[-1]  # Remove intercept
    importance_df <- data.frame(
      Variable = names(coefs),
      Importance = abs(coefs)
    )
    importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]
    
    # Create and save plot
    importance_plot <- plot_variable_importance(
      importance_df,
      title = "Logistic Regression - Variable Importance",
      max_vars = 20
    )
    ggplot2::ggsave(
      file.path(output_dir, "variable_importance.png"),
      importance_plot,
      width = 10,
      height = 8
    )
    message("Variable importance plot saved to: ", file.path(output_dir, "variable_importance.png"))
  }
  
  # Save performance results
  performance_file <- file.path(output_dir, "performance_metrics.RData")
  save(performance, file = performance_file)
  message("Performance metrics saved to: ", performance_file)
  
  # Save a summary text file
  summary_file <- file.path(output_dir, "model_summary.txt")
  sink(summary_file)
  cat("=== LOGISTIC REGRESSION MODEL SUMMARY ===\n\n")
  cat("Date: ", as.character(Sys.Date()), "\n\n")
  
  cat("PERFORMANCE METRICS:\n")
  for(metric in metrics) {
    if(!is.null(performance[[metric]])) {
      cat(paste0(toupper(substring(metric, 1, 1)), substring(metric, 2), ": ", 
                round(performance[[metric]], 4), "\n"))
    }
  }
  
  cat("\nCONFUSION MATRIX:\n")
  print(performance$confusion_matrix$table)
  
  cat("\nDETAILED STATISTICS:\n")
  print(performance$confusion_matrix$byClass)
  
  sink()
  message("Model summary saved to: ", summary_file)
  
  return(performance)
}

# Main function to run the entire logistic regression workflow
run_logistic_regression <- function(train_data, test_data, k_folds = 5, seed_value = 123) {
  message("\n====== Running Logistic Regression Workflow ======\n")
  
  # Step 1: Prepare data for logistic regression
  prepared_data <- prepare_for_logistic_regression(train_data, test_data)
  
  # Step 2: Train logistic regression model
  logistic_model <- train_logistic_regression(prepared_data, k_folds, seed_value)
  
  # Step 3: Generate predictions
  predictions <- generate_predictions(logistic_model, test_data)
  
  # Step 4: Evaluate model performance
  performance <- evaluate_logistic_regression(predictions, test_data$class)
  
  message("\n====== Logistic Regression Workflow Complete ======\n")
  
  # Return model and performance metrics
  return(list(
    model = logistic_model,
    predictions = predictions,
    performance = performance
  ))
}

# Run the model if this script is being run directly
if(!exists("LOGISTIC_REGRESSION_SOURCED") || !LOGISTIC_REGRESSION_SOURCED) {
  # Check if required data is available
  if(!exists("train_data") || !exists("test_data")) {
    source("scripts/02_data_preprocessing.R")
  }
  
  # Run logistic regression
  logistic_results <- run_logistic_regression(train_data, test_data)
  
  # Save model for later use
  saveRDS(logistic_results$model, "results/models/logistic_regression/logistic_model.rds")
  
  LOGISTIC_REGRESSION_SOURCED <- TRUE
} else {
  message("logistic_regression.R has been sourced. Use run_logistic_regression() to train and evaluate the model.")
}