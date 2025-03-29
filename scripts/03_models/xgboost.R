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
  message("Loading data from CSV files...")
  train_data <- read.csv("data/processed/train_data.csv", stringsAsFactors = TRUE)
  test_data <- read.csv("data/processed/test_data.csv", stringsAsFactors = TRUE)
  
  # Ensure class is a factor
  train_data$class <- factor(train_data$class, levels = c("Bad", "Good"))
  test_data$class <- factor(test_data$class, levels = c("Bad", "Good"))
}

library(xgboost)
library(caret)

# Function to prepare data specifically for XGBoost
prepare_for_xgboost <- function(train_data = NULL, test_data) {
  message("\n=== Preparing Data for XGBoost ===")
  
  # Create copy of test data to avoid modifying the original
  test_xgb <- test_data
  
  # If train_data is provided, use it for one-hot encoding reference
  if (!is.null(train_data)) {
    train_xgb <- train_data
    # Identify categorical variables from training data
    factor_cols <- names(train_xgb)[sapply(train_xgb, is.factor)]
    factor_cols <- setdiff(factor_cols, "class")  # Exclude target variable
    
    message("One-hot encoding ", length(factor_cols), " categorical variables...")
    
    # Process both train and test data
    for(col in factor_cols) {
      # Create dummy variables using model.matrix
      train_dummies <- model.matrix(~ 0 + get(col), data = train_xgb)
      test_dummies <- model.matrix(~ 0 + get(col), data = test_xgb)
      
      # Fix column names
      colnames(train_dummies) <- gsub("^get\\(col\\)", col, colnames(train_dummies))
      colnames(test_dummies) <- gsub("^get\\(col\\)", col, colnames(test_dummies))
      
      # Add to datasets
      train_xgb <- cbind(train_xgb, train_dummies)
      test_xgb <- cbind(test_xgb, test_dummies)
      
      # Remove original categorical column
      train_xgb[[col]] <- NULL
      test_xgb[[col]] <- NULL
    }
    
    # Process training data
    train_xgb$target <- as.numeric(train_xgb$class == "Good")
    train_xgb$class <- NULL
    
    features <- setdiff(names(train_xgb), "target")
    train_matrix <- as.matrix(train_xgb[, features])
    train_label <- train_xgb$target
  } else {
    # If no training data, process test data independently
    factor_cols <- names(test_xgb)[sapply(test_xgb, is.factor)]
    factor_cols <- setdiff(factor_cols, "class")
    
    for(col in factor_cols) {
      test_dummies <- model.matrix(~ 0 + get(col), data = test_xgb)
      colnames(test_dummies) <- gsub("^get\\(col\\)", col, colnames(test_dummies))
      test_xgb <- cbind(test_xgb, test_dummies)
      test_xgb[[col]] <- NULL
    }
    
    train_matrix <- NULL
    train_label <- NULL
    features <- setdiff(names(test_xgb), "class")
  }
  
  # Process test data
  test_xgb$target <- as.numeric(test_xgb$class == "Good")
  test_xgb$class <- NULL
  
  test_matrix <- as.matrix(test_xgb[, features])
  test_label <- test_xgb$target
  
  # Ensure all matrices are numeric
  if (!is.null(train_matrix)) storage.mode(train_matrix) <- "numeric"
  storage.mode(test_matrix) <- "numeric"
  
  message("Data preparation completed: ", ncol(test_matrix), " features after one-hot encoding")
  
  return(list(
    train_matrix = train_matrix,
    test_matrix = test_matrix,
    train_label = train_label,
    test_label = test_label,
    features = features
  ))
}

# Function to train XGBoost model with cross-validation
train_xgboost_model <- function(train_data, target_col = "class") {
  message("\n=== Training XGBoost Model ===")
  
  # Create a copy of the data to avoid modifying the original
  data <- train_data
  
  # Convert target to numeric (0/1)
  y <- as.numeric(data[[target_col]] == "Good")
  
  # Remove target column from features
  X <- data[, !names(data) %in% target_col, drop = FALSE]
  
  # Identify numeric and factor columns
  factor_cols <- names(X)[sapply(X, is.factor)]
  numeric_cols <- names(X)[sapply(X, is.numeric)]
  
  # Create dummy variables for factor columns
  if(length(factor_cols) > 0) {
    # Create model matrix for categorical variables
    factor_matrix <- model.matrix(~ . - 1, data = X[, factor_cols, drop = FALSE])
    
    # If there are numeric columns, combine them with the dummy variables
    if(length(numeric_cols) > 0) {
      X_processed <- cbind(as.matrix(X[, numeric_cols, drop = FALSE]), factor_matrix)
    } else {
      X_processed <- factor_matrix
    }
  } else {
    # If no factor columns, just convert numeric columns to matrix
    X_processed <- as.matrix(X[, numeric_cols, drop = FALSE])
  }
  
  # Ensure all data is numeric
  storage.mode(X_processed) <- "numeric"
  
  # Create DMatrix
  dtrain <- xgb.DMatrix(data = X_processed, label = y)
  
  # Set parameters
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.1,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  # Train model
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 100,
    verbose = 0
  )
  
  # Store feature names in the model for later use
  model$feature_names <- colnames(X_processed)
  
  return(model)
}

# Function to generate predictions using the trained model
generate_predictions <- function(model, test_data) {
  message("\n=== Generating Predictions ===")
  
  # If test_data is not a matrix, convert it
  if(!inherits(test_data, "xgb.DMatrix")) {
    if(is.data.frame(test_data)) {
      test_data <- prepare_for_xgboost(NULL, test_data)$test_matrix
    }
  }
  
  # Generate probabilities
  pred_prob <- predict(model, test_data)
  
  # Convert to class predictions
  pred_class <- ifelse(pred_prob > 0.5, "Good", "Bad")
  pred_class <- factor(pred_class, levels = c("Bad", "Good"))
  
  return(list(
    class = pred_class,
    prob = pred_prob
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
    importance_df <- data.frame(
      Feature = importance$Feature,
      Gain = importance$Gain
    )
    
    importance_plot <- plot_variable_importance(
      importance_df,
      title = "XGBoost - Feature Importance (Gain)",
      max_vars = 20
    )
    
    # Save the plot if it was created successfully
    if (!is.null(importance_plot)) {
      ggplot2::ggsave(
        file.path(output_dir, "feature_importance.png"),
        importance_plot,
        width = 10,
        height = 8
      )
      message("Variable importance plot saved to: ", file.path(output_dir, "feature_importance.png"))
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
  
  tryCatch({
    # Get feature names from the model
    feature_names <- model$feature_names
    
    # Ensure test_matrix has the same features as the model
    if(ncol(test_matrix) != length(feature_names)) {
      message("Adjusting test matrix to match model features...")
      
      # Find common features
      common_features <- intersect(colnames(test_matrix), feature_names)
      
      # Check if we have enough matching features
      if(length(common_features) < length(feature_names) * 0.8) {
        warning("Less than 80% of features match between model and test data")
      }
      
      # Subset test_matrix to only include model features
      test_matrix <- test_matrix[, common_features, drop = FALSE]
      feature_names <- common_features
    }
    
    # Convert test_matrix to DMatrix
    dtest <- xgboost::xgb.DMatrix(test_matrix)
    
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
      file.path(output_dir, "shap_summary.png"),
      summary_plot,
      width = 12,
      height = 8
    )
    message("SHAP summary plot saved to: ", file.path(output_dir, "shap_summary.png"))
    
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
        file.path(output_dir, paste0("shap_", feature, ".png")),
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
  xgb_model <- train_xgboost_model(train_data)
  
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

# Update the plot_variable_importance function
plot_variable_importance <- function(importance_df, title = "Variable Importance", max_vars = 20) {
  if(!requireNamespace("ggplot2", quietly = TRUE)) {
    message("ggplot2 package is required for plotting variable importance")
    return(NULL)
  }
  
  # Ensure importance_df has the required columns
  if(!all(c("Feature", "Gain") %in% names(importance_df))) {
    stop("importance_df must contain 'Feature' and 'Gain' columns")
  }
  
  # Sort by importance and take top variables
  importance_df <- importance_df[order(-importance_df$Gain), ]
  if(nrow(importance_df) > max_vars) {
    importance_df <- importance_df[1:max_vars, ]
  }
  
  # Create the plot
  plot <- ggplot2::ggplot(importance_df, 
                         ggplot2::aes(x = reorder(Feature, Gain), y = Gain)) +
    ggplot2::geom_bar(stat = "identity", fill = "steelblue") +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title = title,
      x = "Features",
      y = "Importance (Gain)"
    ) +
    ggplot2::theme_minimal()
  
  return(plot)
}

# Update the SHAP calculation function
calculate_shap_values <- function(model, data) {
  if(!requireNamespace("xgboost", quietly = TRUE)) {
    message("xgboost package not available. Skipping SHAP values...")
    return(NULL)
  }
  
  tryCatch({
    # Convert data to DMatrix if needed
    if(!inherits(data, "xgb.DMatrix")) {
      data <- xgb.DMatrix(data = as.matrix(data))
    }
    
    # Calculate SHAP values using xgboost's built-in function
    shap_values <- predict(model, data, predcontrib = TRUE)
    
    return(shap_values)
  }, error = function(e) {
    message("Error generating SHAP values: ", e$message)
    return(NULL)
  })
}