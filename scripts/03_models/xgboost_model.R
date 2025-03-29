# ======= xgboost_model.R =======
# This script implements an XGBoost model for the German Credit dataset.
# The script handles:
# - Data preparation specific to XGBoost
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

library(xgboost)
library(caret)

# Function to prepare data specifically for XGBoost
prepare_for_xgboost <- function(train_data, test_data) {
  message("\n=== Preparing Data for XGBoost ===")
  
  # Create copies to avoid modifying the originals
  train_xgb <- train_data
  test_xgb <- test_data
  
  # For XGBoost, we need:
  # 1. Convert categorical variables to numeric (one-hot encoding)
  # 2. Convert target to 0/1 numeric format
  # 3. Create DMatrix objects
  
  # Step 1: Identify categorical variables and perform one-hot encoding
  factor_cols <- names(train_xgb)[sapply(train_xgb, is.factor)]
  factor_cols <- setdiff(factor_cols, "class")  # Exclude target variable
  
  message("One-hot encoding ", length(factor_cols), " categorical variables...")
  
  # Create dummy variables for each categorical column
  for(col in factor_cols) {
    # Create dummy variables using model.matrix
    train_dummies <- model.matrix(~ 0 + get(col), data = train_xgb)
    test_dummies <- model.matrix(~ 0 + get(col), data = test_xgb)
    
    # Fix column names (remove "get(col)" prefix)
    colnames(train_dummies) <- gsub("^get\\(col\\)", col, colnames(train_dummies))
    colnames(test_dummies) <- gsub("^get\\(col\\)", col, colnames(test_dummies))
    
    # Add to datasets
    train_xgb <- cbind(train_xgb, train_dummies)
    test_xgb <- cbind(test_xgb, test_dummies)
    
    # Remove original categorical column
    train_xgb[[col]] <- NULL
    test_xgb[[col]] <- NULL
  }
  
  # Step 2: Convert target to 0/1 format
  # For XGBoost, the positive class is labeled as 1
  train_xgb$target <- as.numeric(train_xgb$class == "Good")
  test_xgb$target <- as.numeric(test_xgb$class == "Good")
  
  # Remove original class column
  train_xgb$class <- NULL
  test_xgb$class <- NULL
  
  # Step 3: Create feature matrices and label vectors
  features <- setdiff(names(train_xgb), "target")
  
  # Create matrices
  train_matrix <- as.matrix(train_xgb[, features])
  test_matrix <- as.matrix(test_xgb[, features])
  
  train_label <- train_xgb$target
  test_label <- test_xgb$target
  
  message("Data preparation completed: ", ncol(train_matrix), " features after one-hot encoding")
  
  # Return prepared data
  return(list(
    train_matrix = train_matrix,
    test_matrix = test_matrix,
    train_label = train_label,
    test_label = test_label,
    features = features
  ))
}

# Function to train XGBoost model with cross-validation
train_xgboost_model <- function(prepared_data, k_folds = 5, seed_value = 123) {
  message("\n=== Training XGBoost Model with Cross-Validation ===")
  
  # Set seed for reproducibility
  set.seed(seed_value)
  
  # Extract prepared data
  train_matrix <- prepared_data$train_matrix
  train_label <- prepared_data$train_label
  
  # Check if xgboost package is available
  if(!requireNamespace("xgboost", quietly = TRUE)) {
    message("XGBoost package not found. Attempting to install...")
    
    # Try to install and load xgboost
    xgboost_available <- resolve_xgboost_issues()
    
    if(!xgboost_available) {
      stop("Failed to install XGBoost. Cannot continue with XGBoost model.")
    }
  }
  
  # Load xgboost
  library(xgboost)
  
  # Create DMatrix objects for XGBoost
  dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
  
  # Set XGBoost parameters
  params <- list(
    objective = "binary:logistic",   # Binary classification with logistic regression
    eval_metric = "auc",             # AUC for evaluation
    booster = "gbtree",              # Tree-based model
    eta = 0.1,                       # Learning rate
    max_depth = 6,                   # Maximum tree depth
    min_child_weight = 1,            # Minimum sum of instance weight needed in a child
    subsample = 0.8,                 # Subsample ratio of the training instances
    colsample_bytree = 0.8,          # Subsample ratio of columns when constructing each tree
    scale_pos_weight = sum(train_label == 0) / sum(train_label == 1)  # Handle class imbalance
  )
  
  # Define parameter grid for cross-validation
  param_grid <- expand.grid(
    eta = c(0.01, 0.05, 0.1),
    max_depth = c(3, 6, 9),
    min_child_weight = c(1, 3, 5),
    subsample = c(0.6, 0.8, 1.0),
    colsample_bytree = c(0.6, 0.8, 1.0),
    nrounds = c(100, 200, 300)
  )
  
  # Limit to a smaller subset of combinations for reasonable training time
  set.seed(seed_value)
  param_grid <- param_grid[sample(nrow(param_grid), min(nrow(param_grid), 5)), ]
  
  message("Performing cross-validation with ", nrow(param_grid), " parameter combinations...")
  
  # Train the model with error handling
  tryCatch({
    # Start timing
    start_time <- proc.time()
    
    # Initialize variables to track best model
    best_score <- 0
    best_params <- NULL
    best_nrounds <- 0
    
    # Perform simplified grid search with cross-validation
    for(i in 1:nrow(param_grid)) {
      # Get current parameter set
      current_params <- list(
        objective = "binary:logistic",
        eval_metric = "auc",
        eta = param_grid$eta[i],
        max_depth = param_grid$max_depth[i],
        min_child_weight = param_grid$min_child_weight[i],
        subsample = param_grid$subsample[i],
        colsample_bytree = param_grid$colsample_bytree[i],
        scale_pos_weight = sum(train_label == 0) / sum(train_label == 1)
      )
      
      # Perform cross-validation
      cv_result <- xgb.cv(
        params = current_params,
        data = dtrain,
        nrounds = param_grid$nrounds[i],
        nfold = k_folds,
        early_stopping_rounds = 10,
        verbose = 0,
        stratified = TRUE
      )
      
      # Extract best score and iteration
      best_iteration <- which.max(cv_result$evaluation_log$test_auc_mean)
      best_iteration_score <- cv_result$evaluation_log$test_auc_mean[best_iteration]
      
      message("Params ", i, "/", nrow(param_grid), " - AUC: ", 
              round(best_iteration_score, 4), 
              " at iteration ", best_iteration)
      
      # Update best params if current is better
      if(best_iteration_score > best_score) {
        best_score <- best_iteration_score
        best_params <- current_params
        best_nrounds <- best_iteration
      }
    }
    
    message("Best parameters found - AUC: ", round(best_score, 4))
    message("Best hyperparameters: ")
    print(best_params)
    message("Best number of rounds: ", best_nrounds)
    
    # Train final model with best parameters
    final_model <- xgboost(
      params = best_params,
      data = dtrain,
      nrounds = best_nrounds,
      verbose = 0
    )
    
    # End timing
    end_time <- proc.time()
    train_time <- end_time - start_time
    
    message("Model training completed in ", round(train_time[3], 2), " seconds")
    
    # Get feature importance
    importance <- xgb.importance(feature_names = prepared_data$features, model = final_model)
    message("\nTop 10 features by importance:")
    print(head(importance, 10))
    
    # Add best parameters and metrics to model
    final_model$best_params <- best_params
    final_model$best_nrounds <- best_nrounds
    final_model$best_score <- best_score
    final_model$feature_names <- prepared_data$features
    
    return(final_model)
    
  }, error = function(e) {
    message("ERROR training XGBoost: ", e$message)
    
    # Try with a simpler approach if the original fails
    message("Attempting with a simplified model...")
    
    # Simple parameters
    simple_params <- list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = 0.1,
      max_depth = 3,
      min_child_weight = 1
    )
    
    # Train simple model
    simple_model <- xgboost(
      params = simple_params,
      data = dtrain,
      nrounds = 100,
      verbose = 0
    )
    
    message("Simplified model training completed")
    
    # Add parameters to model
    simple_model$best_params <- simple_params
    simple_model$best_nrounds <- 100
    simple_model$feature_names <- prepared_data$features
    
    return(simple_model)
  })
}

# Function to generate predictions using the trained model
generate_predictions <- function(model, test_data) {
  # Prepare test data
  test_prepared <- prepare_for_xgboost(test_data)
  
  # Generate predictions
  pred_prob <- predict(model, test_prepared$data)
  pred_class <- ifelse(pred_prob > 0.5, "Good", "Bad")
  
  return(list(
    class = pred_class,
    prob = pred_prob,
    all_probs = data.frame(Bad = 1 - pred_prob, Good = pred_prob)
  ))
}

# Function to evaluate model performance
evaluate_xgboost <- function(predictions, actual_label, model, output_dir = "results/models/xgboost") {
  message("\n=== Evaluating Model Performance ===")
  
  # Create output directory if it doesn't exist
  if(!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    message("Created directory: ", output_dir)
  }
  
  # Convert numeric actual labels (0/1) to factor (Bad/Good)
  actual_class <- factor(ifelse(actual_label == 1, "Good", "Bad"), levels = c("Bad", "Good"))
  
  # Evaluate model using the evaluation.R utility
  performance <- evaluate_model(
    pred = predictions$class,
    actual = actual_class,
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
    actual = actual_class,
    positive_class = "Good",
    model_name = "XGBoost"
  )
  dev.off()
  message("ROC curve saved to: ", file.path(output_dir, "roc_curve.png"))
  
  # Plot confusion matrix
  if(requireNamespace("ggplot2", quietly = TRUE)) {
    confusion_plot <- plot_confusion_matrix(
      performance$confusion_matrix,
      title = "XGBoost Confusion Matrix"
    )
    ggplot2::ggsave(
      file.path(output_dir, "confusion_matrix.png"),
      confusion_plot,
      width = 8,
      height = 6
    )
    message("Confusion matrix plot saved to: ", file.path(output_dir, "confusion_matrix.png"))
  }
  
  # Get feature importance
  importance <- xgb.importance(model = model)
  
  # Check if importance data exists
  if (!is.null(importance) && nrow(importance) > 0) {
    # Create importance plot
    importance_plot <- plot_variable_importance(
      importance_df = data.frame(
        Feature = importance$Feature,
        Gain = importance$Gain
      ),
      title = "XGBoost - Feature Importance (Gain)",
      max_vars = 20
    )
    
    # Save the plot if it was created successfully
    if (!is.null(importance_plot)) {
      ggsave(
        file.path(output_dir, "feature_importance.png"),
        importance_plot,
        width = 10,
        height = 8
      )
    }
  } else {
    warning("No feature importance data available")
  }
  
  # Save performance results
  performance_file <- file.path(output_dir, "performance_metrics.RData")
  save(performance, file = performance_file)
  message("Performance metrics saved to: ", performance_file)
  
  # Save a summary text file
  summary_file <- file.path(output_dir, "model_summary.txt")
  sink(summary_file)
  cat("=== XGBOOST MODEL SUMMARY ===\n\n")
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
  
  cat("\nMODEL PARAMETERS:\n")
  for(param_name in names(model$best_params)) {
    cat(param_name, ": ", model$best_params[[param_name]], "\n")
  }
  cat("nrounds: ", model$best_nrounds, "\n")
  
  cat("\nTOP 10 FEATURES BY IMPORTANCE (GAIN):\n")
  if(requireNamespace("xgboost", quietly = TRUE)) {
    importance <- xgboost::xgb.importance(feature_names = model$feature_names, model = model)
    print(head(importance, 10))
  }
  
  sink()
  message("Model summary saved to: ", summary_file)
  
  return(performance)
}

# Function to generate and plot SHAP values for model explainability
plot_shap_values <- function(model, test_matrix, output_dir = "results/models/xgboost") {
  message("\n=== Generating SHAP Values for Model Explainability ===")
  
  # Check if required packages are available
  if(!requireNamespace("xgboost", quietly = TRUE) || !requireNamespace("ggplot2", quietly = TRUE)) {
    message("Required packages not available for SHAP analysis. Skipping...")
    return(NULL)
  }
  
  # Create SHAP directory if it doesn't exist
  shap_dir <- file.path(output_dir, "shap")
  if(!dir.exists(shap_dir)) {
    dir.create(shap_dir, recursive = TRUE)
    message("Created directory: ", shap_dir)
  }
  
  tryCatch({
    # Convert test_matrix to DMatrix
    dtest <- xgboost::xgb.DMatrix(test_matrix)
    
    # Get feature names from the model
    feature_names <- model$feature_names
    
    # Ensure test_matrix has the same number of columns as feature names
    if(ncol(test_matrix) != length(feature_names)) {
      stop(sprintf("Mismatch in dimensions: test_matrix has %d columns but there are %d feature names", 
                  ncol(test_matrix), length(feature_names)))
    }
    
    # Compute SHAP values using xgboost's predict function
    shap_values <- xgboost::predict(model, dtest, predcontrib = TRUE)
    
    # Take a sample of records to visualize (to avoid cluttered plots)
    set.seed(123)
    sample_size <- min(100, nrow(test_matrix))
    sample_idx <- sample(1:nrow(test_matrix), sample_size)
    
    # Get top 10 most important features for plotting
    importance <- xgboost::xgb.importance(feature_names = feature_names, model = model)
    top_features <- head(importance$Feature, 10)
    
    # Create summary plot for SHAP values
    shap_df <- data.frame(
      Feature = rep(feature_names, each = sample_size),
      SHAP_Value = as.vector(shap_values[sample_idx, 1:length(feature_names)]),
      Feature_Value = as.vector(sapply(feature_names, function(f) {
        col_idx <- which(feature_names == f)
        return(test_matrix[sample_idx, col_idx])
      }))
    )
    
    # Filter to top features
    shap_df <- shap_df[shap_df$Feature %in% top_features, ]
    
    # Calculate absolute SHAP values for sorting
    shap_df$SHAP_Abs <- abs(shap_df$SHAP_Value)
    
    # Order features by importance
    shap_df$Feature <- factor(shap_df$Feature, levels = rev(top_features))
    
    # Create summary plot
    summary_plot <- ggplot2::ggplot(shap_df, ggplot2::aes(x = SHAP_Value, y = Feature, color = Feature_Value)) +
      ggplot2::geom_jitter(width = 0, height = 0.2, alpha = 0.7, size = 2) +
      ggplot2::scale_color_gradient(low = "blue", high = "red", name = "Feature Value") +
      ggplot2::labs(
        title = "SHAP Values for Top 10 Features",
        x = "SHAP Value (Impact on Model Output)",
        y = "Feature"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(size = 16, face = "bold"),
        axis.title = ggplot2::element_text(size = 12),
        axis.text = ggplot2::element_text(size = 10)
      )
    
    # Save the plot
    ggplot2::ggsave(
      file.path(shap_dir, "shap_summary.png"),
      summary_plot,
      width = 12,
      height = 8
    )
    message("SHAP summary plot saved to: ", file.path(shap_dir, "shap_summary.png"))
    
    # Generate individual feature plots for top 5 features
    for(i in 1:min(5, length(top_features))) {
      feature <- top_features[i]
      feature_data <- shap_df[shap_df$Feature == feature, ]
      
      feature_plot <- ggplot2::ggplot(feature_data, ggplot2::aes(x = Feature_Value, y = SHAP_Value)) +
        ggplot2::geom_point(color = "blue", alpha = 0.6) +
        ggplot2::geom_smooth(method = "loess", se = TRUE, color = "red") +
        ggplot2::labs(
          title = paste("SHAP Values for", feature),
          x = "Feature Value",
          y = "SHAP Value (Impact on Model Output)"
        ) +
        ggplot2::theme_minimal()
      
      # Save the plot
      ggplot2::ggsave(
        file.path(shap_dir, paste0("shap_", feature, ".png")),
        feature_plot,
        width = 10,
        height = 6
      )
      message("Feature SHAP plot saved for: ", feature)
    }
    
    return(shap_df)
    
  }, error = function(e) {
    message("Error generating SHAP values: ", e$message)
    return(NULL)
  })
}

# Main function to run the entire XGBoost workflow
run_xgboost <- function(train_data, test_data, k_folds = 5, seed_value = 123) {
  message("\n====== Running XGBoost Workflow ======\n")
  
  # Step 1: Prepare data for XGBoost
  prepared_data <- prepare_for_xgboost(train_data, test_data)
  
  # Step 2: Train XGBoost model
  xgb_model <- train_xgboost_model(prepared_data, k_folds, seed_value)
  
  # Step 3: Generate predictions
  predictions <- generate_predictions(xgb_model, test_data)
  
  # Step 4: Evaluate model performance
  performance <- evaluate_xgboost(predictions, prepared_data$test_label, xgb_model)
  
  # Step 5: Generate SHAP values for explainability
  shap_data <- plot_shap_values(xgb_model, prepared_data$test_matrix)
  
  message("\n====== XGBoost Workflow Complete ======\n")
  
  # Return model and performance metrics
  return(list(
    model = xgb_model,
    predictions = predictions,
    performance = performance,
    shap_data = shap_data
  ))
}

# Run the model if this script is being run directly
if(!exists("XGBOOST_SOURCED") || !XGBOOST_SOURCED) {
  # Check if required data is available
  if(!exists("train_data") || !exists("test_data")) {
    source("scripts/02_data_preprocessing.R")
  }
  
  # Run XGBoost
  xgb_results <- run_xgboost(train_data, test_data)
  
  # Save model for later use
  saveRDS(xgb_results$model, "results/models/xgboost/xgboost_model.rds")
  
  XGBOOST_SOURCED <- TRUE
} else {
  message("xgboost_model.R has been sourced. Use run_xgboost() to train and evaluate the model.")
}

# Prepare data for XGBoost
prepare_xgb_data <- function(data) {
  # Convert factors to dummy variables
  dummies <- model.matrix(~.-1, data[, !names(data) %in% "class", drop = FALSE])
  
  # Convert class to numeric (0/1)
  labels <- as.numeric(data$class) - 1
  
  return(list(data = dummies, label = labels))
}

# Train XGBoost model
train_xgb <- function(train_data) {
  prepared_data <- prepare_xgb_data(train_data)
  
  xgb_params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.1,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  xgb_model <- xgboost(
    data = prepared_data$data,
    label = prepared_data$label,
    params = xgb_params,
    nrounds = 100,
    verbose = 0
  )
  
  return(xgb_model)
}

# Generate predictions
predict_xgb <- function(model, test_data) {
  prepared_test <- prepare_xgb_data(test_data)
  predictions <- predict(model, prepared_test$data)
  return(predictions)
}