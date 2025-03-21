# ======= svm_model.R =======
# This script implements a Support Vector Machine model for the German Credit dataset.
# The script handles:
# - Data preparation specific to SVM
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

# Function to prepare data specifically for SVM
prepare_for_svm <- function(train_data, test_data) {
  message("\n=== Preparing Data for SVM ===")
  
  # Create copies to avoid modifying the originals
  train_svm <- train_data
  test_svm <- test_data
  
  # For SVM, we need:
  # 1. Scale all numeric features (essential for SVM)
  # 2. Convert categorical variables to numeric (one-hot encoding)
  # 3. Consider dimensionality reduction for large datasets
  
  # Step 1: Scale numeric features
  numeric_cols <- names(train_svm)[sapply(train_svm, is.numeric)]
  if(length(numeric_cols) > 0) {
    message("Scaling ", length(numeric_cols), " numeric features...")
    
    # Calculate scaling parameters from training data
    scaling_params <- list()
    for(col in numeric_cols) {
      scaling_params[[col]] <- list(
        mean = mean(train_svm[[col]], na.rm = TRUE),
        sd = sd(train_svm[[col]], na.rm = TRUE)
      )
      
      # Apply scaling to training data
      train_svm[[col]] <- (train_svm[[col]] - scaling_params[[col]]$mean) / 
                          scaling_params[[col]]$sd
      
      # Apply same scaling to test data
      test_svm[[col]] <- (test_svm[[col]] - scaling_params[[col]]$mean) / 
                          scaling_params[[col]]$sd
    }
  }
  
  # Step 2: One-hot encode categorical variables
  factor_cols <- names(train_svm)[sapply(train_svm, is.factor)]
  factor_cols <- setdiff(factor_cols, "class")  # Exclude target variable
  
  if(length(factor_cols) > 0) {
    message("One-hot encoding ", length(factor_cols), " categorical variables...")
    
    for(col in factor_cols) {
      # Create dummy variables using model.matrix
      train_dummies <- model.matrix(~ 0 + get(col), data = train_svm)
      test_dummies <- model.matrix(~ 0 + get(col), data = test_svm)
      
      # Fix column names (remove "get(col)" prefix)
      colnames(train_dummies) <- gsub("^get\\(col\\)", col, colnames(train_dummies))
      colnames(test_dummies) <- gsub("^get\\(col\\)", col, colnames(test_dummies))
      
      # Add to datasets
      train_svm <- cbind(train_svm, train_dummies)
      test_svm <- cbind(test_svm, test_dummies)
      
      # Remove original categorical column
      train_svm[[col]] <- NULL
      test_svm[[col]] <- NULL
    }
  }
  
  # Step 3: Feature selection (optional for very high-dimensional data)
  total_features <- ncol(train_svm) - 1  # Excluding class column
  if(total_features > 100) {
    message("High dimensional data detected (", total_features, " features). Consider feature selection.")
    # Feature selection would be implemented here if needed
  }
  
  message("Data preparation completed: ", ncol(train_svm) - 1, " features after processing")
  
  # Create a formula that includes all predictors except the class
  predictors <- setdiff(names(train_svm), "class")
  formula_string <- paste("class ~", paste(predictors, collapse = " + "))
  model_formula <- as.formula(formula_string)
  
  return(list(
    train = train_svm,
    test = test_svm,
    formula = model_formula,
    scaling_params = scaling_params
  ))
}

# Function to train SVM model with cross-validation
train_svm_model <- function(prepared_data, k_folds = 5, seed_value = 123) {
  message("\n=== Training SVM Model with Cross-Validation ===")
  
  # Set seed for reproducibility
  set.seed(seed_value)
  
  # Extract prepared data
  train_data <- prepared_data$train
  model_formula <- prepared_data$formula
  
  # Check if required packages are available
  if(!requireNamespace("e1071", quietly = TRUE) || !requireNamespace("kernlab", quietly = TRUE)) {
    message("Installing required packages for SVM...")
    install.packages(c("e1071", "kernlab"), repos = "https://cloud.r-project.org")
  }
  
  # Set up cross-validation
  ctrl <- trainControl(
    method = "cv",            # Cross-validation
    number = k_folds,         # Number of folds
    classProbs = TRUE,        # Calculate class probabilities
    summaryFunction = twoClassSummary,  # Use ROC summary
    savePredictions = "final" # Save final predictions
  )
  
  # Create a tuning grid for SVM parameters
  # For RBF kernel: Cost (C) and sigma
  tuning_grid <- expand.grid(
    sigma = c(0.01, 0.05, 0.1),  # Kernel parameter
    C = c(0.1, 1, 10)            # Cost parameter
  )
  
  # Train the model with error handling
  tryCatch({
    message("Training SVM model with Radial Basis Function kernel...")
    
    # Start timing
    start_time <- proc.time()
    
    # Train the model with cross-validation and parameter tuning
    # Use svmRadial method from kernlab through caret
    svm_model <- train(
      model_formula, 
      data = train_data,
      method = "svmRadial",     # RBF kernel
      trControl = ctrl,
      tuneGrid = tuning_grid,
      metric = "ROC",
      preProcess = NULL,        # Data already preprocessed
      prob.model = TRUE         # Generate class probabilities
    )
    
    # End timing
    end_time <- proc.time()
    train_time <- end_time - start_time
    
    message("Model training completed in ", round(train_time[3], 2), " seconds")
    
    # Print model summary
    message("\nModel Summary:")
    print(svm_model)
    
    # Print cross-validation results
    message("\nCross-Validation Results:")
    print(svm_model$results)
    
    # Print best tuning parameters
    message("\nBest Tuning Parameters:")
    print(svm_model$bestTune)
    
    return(svm_model)
    
  }, error = function(e) {
    message("ERROR training SVM: ", e$message)
    
    # Try with a simpler approach if the original fails
    message("Attempting with a simplified model...")
    
    # Train a simpler linear SVM model directly with e1071
    if(requireNamespace("e1071", quietly = TRUE)) {
      simple_model <- e1071::svm(
        formula = model_formula,
        data = train_data,
        kernel = "linear",
        cost = 1,
        probability = TRUE,
        scale = FALSE  # Data already scaled
      )
      
      message("Simplified linear SVM model training completed")
      return(simple_model)
    } else {
      stop("Could not train SVM model with either method")
    }
  })
}

# Function to generate predictions using the trained model
generate_predictions <- function(model, test_data) {
  message("\n=== Generating Predictions ===")
  
  # Handle different model types
  if(inherits(model, "train")) {
    # Caret's train object
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
  } else if(inherits(model, "svm")) {
    # Direct svm object from e1071
    # Generate class predictions
    pred_class <- predict(model, newdata = test_data, type = "class")
    
    # Generate probability predictions if model was trained with probability=TRUE
    if(model$prob) {
      pred_prob_attr <- predict(model, newdata = test_data, probability = TRUE)
      prob_matrix <- attr(pred_prob_attr, "probabilities")
      
      # Find the column for the positive class
      if("Good" %in% colnames(prob_matrix)) {
        pos_class_prob <- prob_matrix[, "Good"]
      } else {
        # If "Good" not found, use second column
        pos_class_prob <- prob_matrix[, 2]
        message("Used second probability column as positive class")
      }
    } else {
      # If probabilities not available, use decision values (less reliable)
      message("WARNING: SVM model was not trained with probability=TRUE")
      decision_values <- predict(model, newdata = test_data, decision.values = TRUE)
      attr_decision <- attr(decision_values, "decision.values")
      # Convert decision values to pseudo-probabilities with sigmoid function
      pos_class_prob <- 1 / (1 + exp(-attr_decision))
    }
  } else {
    stop("Unknown model type. Cannot generate predictions.")
  }
  
  # Ensure pred_class is a factor with the right levels
  if(!is.factor(pred_class)) {
    pred_class <- factor(pred_class, levels = c("Bad", "Good"))
  }
  
  message("Generated predictions for ", length(pred_class), " test samples")
  
  return(list(
    class = pred_class,
    prob = pos_class_prob
  ))
}

# Function to evaluate model performance
evaluate_svm <- function(predictions, actual, output_dir = "results/models/svm") {
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
    model_name = "SVM"
  )
  dev.off()
  message("ROC curve saved to: ", file.path(output_dir, "roc_curve.png"))
  
  # Plot confusion matrix
  if(requireNamespace("ggplot2", quietly = TRUE)) {
    confusion_plot <- plot_confusion_matrix(
      performance$confusion_matrix,
      title = "SVM Confusion Matrix"
    )
    ggplot2::ggsave(
      file.path(output_dir, "confusion_matrix.png"),
      confusion_plot,
      width = 8,
      height = 6
    )
    message("Confusion matrix plot saved to: ", file.path(output_dir, "confusion_matrix.png"))
  }
  
  # Create precision-recall curve (an additional useful visualization for imbalanced data)
  if(requireNamespace("PRROC", quietly = TRUE) && requireNamespace("ggplot2", quietly = TRUE)) {
    # Convert actual to numeric for PRROC (1 for positive class)
    actual_numeric <- as.numeric(actual == "Good")
    
    # Calculate precision-recall curve
    pr_curve <- PRROC::pr.curve(
      scores.class0 = predictions$prob,
      weights.class0 = actual_numeric,
      curve = TRUE
    )
    
    # Create data frame for plotting
    pr_df <- data.frame(
      Recall = pr_curve$curve[, 1],
      Precision = pr_curve$curve[, 2]
    )
    
    # Plot with ggplot2
    pr_plot <- ggplot2::ggplot(pr_df, ggplot2::aes(x = Recall, y = Precision)) +
      ggplot2::geom_line(color = "blue", size = 1) +
      ggplot2::geom_area(fill = "skyblue", alpha = 0.3) +
      ggplot2::labs(
        title = paste("Precision-Recall Curve - AUPRC =", round(pr_curve$auc.integral, 4)),
        x = "Recall",
        y = "Precision"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(size = 14, face = "bold"),
        axis.title = ggplot2::element_text(size = 12),
        axis.text = ggplot2::element_text(size = 10)
      )
    
    # Save the plot
    ggplot2::ggsave(
      file.path(output_dir, "precision_recall_curve.png"),
      pr_plot,
      width = 8,
      height = 6
    )
    message("Precision-recall curve saved to: ", file.path(output_dir, "precision_recall_curve.png"))
  }
  
  # Save performance results
  performance_file <- file.path(output_dir, "performance_metrics.RData")
  save(performance, file = performance_file)
  message("Performance metrics saved to: ", performance_file)
  
  # Save a summary text file
  summary_file <- file.path(output_dir, "model_summary.txt")
  sink(summary_file)
  cat("=== SVM MODEL SUMMARY ===\n\n")
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

# Function to visualize decision boundary (for 2D feature space only)
visualize_decision_boundary <- function(model, train_data, output_dir = "results/models/svm") {
  message("\n=== Visualizing SVM Decision Boundary ===")
  
  # Check if required packages are available
  if(!requireNamespace("ggplot2", quietly = TRUE)) {
    message("ggplot2 package not available. Skipping visualization...")
    return(NULL)
  }
  
  # This visualization only works well for 2 features
  # So we'll select the 2 most important numeric features
  numeric_cols <- names(train_data)[sapply(train_data, is.numeric)]
  
  if(length(numeric_cols) < 2) {
    message("Need at least 2 numeric features for decision boundary visualization. Skipping...")
    return(NULL)
  }
  
  # If we have many numeric features, select 2 important ones
  # For simplicity, we'll use duration and credit_amount if available
  if("duration" %in% numeric_cols && "credit_amount" %in% numeric_cols) {
    feature1 <- "duration"
    feature2 <- "credit_amount"
  } else {
    # Otherwise just use the first two numeric features
    feature1 <- numeric_cols[1]
    feature2 <- numeric_cols[2]
  }
  
  message("Visualizing decision boundary using features: ", feature1, " and ", feature2)
  
  # Create a grid for prediction
  grid_size <- 100
  x_range <- range(train_data[[feature1]])
  y_range <- range(train_data[[feature2]])
  
  # Expand ranges slightly for better visualization
  x_margin <- (x_range[2] - x_range[1]) * 0.05
  y_margin <- (y_range[2] - y_range[1]) * 0.05
  x_range <- c(x_range[1] - x_margin, x_range[2] + x_margin)
  y_range <- c(y_range[1] - y_margin, y_range[2] + y_margin)
  
  # Create grid
  grid_x <- seq(x_range[1], x_range[2], length.out = grid_size)
  grid_y <- seq(y_range[1], y_range[2], length.out = grid_size)
  grid_data <- expand.grid(x = grid_x, y = grid_y)
  
  # Set column names to match the original data
  names(grid_data) <- c(feature1, feature2)
  
  # Create a data frame for prediction that matches the training data structure
  # This is tricky as we need to include all columns that the model expects
  # For simplicity, we'll use a subset of the model with just the two features
  
  # Train a simple model just for visualization
  formula <- as.formula(paste("class ~", feature1, "+", feature2))
  
  tryCatch({
    if(requireNamespace("e1071", quietly = TRUE)) {
      simple_model <- e1071::svm(
        formula = formula,
        data = train_data,
        kernel = "radial",
        gamma = 0.1,
        cost = 1,
        probability = FALSE
      )
      
      # Predict on the grid
      grid_data$prediction <- predict(simple_model, newdata = grid_data)
      
      # Create a data frame for the original points
      plot_data <- data.frame(
        x = train_data[[feature1]],
        y = train_data[[feature2]],
        class = train_data$class
      )
      
      # Create the plot
      boundary_plot <- ggplot2::ggplot() +
        # Add decision boundary as a filled contour
        ggplot2::geom_tile(data = grid_data, 
                   ggplot2::aes(x = .data[[feature1]], y = .data[[feature2]], fill = prediction),
                   alpha = 0.3) +
        # Add original data points
        ggplot2::geom_point(data = plot_data,
                   ggplot2::aes(x = x, y = y, color = class, shape = class),
                   size = 3, alpha = 0.7) +
        # Set colors
        ggplot2::scale_fill_manual(values = c("Bad" = "red", "Good" = "green")) +
        ggplot2::scale_color_manual(values = c("Bad" = "red", "Good" = "green")) +
        # Labels
        ggplot2::labs(
          title = "SVM Decision Boundary",
          subtitle = paste("Features:", feature1, "vs", feature2),
          x = feature1,
          y = feature2,
          fill = "Predicted Class",
          color = "Actual Class",
          shape = "Actual Class"
        ) +
        ggplot2::theme_minimal() +
        ggplot2::theme(
          plot.title = ggplot2::element_text(size = 16, face = "bold"),
          axis.title = ggplot2::element_text(size = 14),
          legend.title = ggplot2::element_text(size = 12),
          legend.text = ggplot2::element_text(size = 10)
        )
      
      # Save the plot
      ggplot2::ggsave(
        file.path(output_dir, "decision_boundary.png"),
        boundary_plot,
        width = 10,
        height = 8
      )
      message("Decision boundary plot saved to: ", file.path(output_dir, "decision_boundary.png"))
      
      return(boundary_plot)
    } else {
      message("e1071 package not available. Skipping decision boundary visualization...")
      return(NULL)
    }
  }, error = function(e) {
    message("Error visualizing decision boundary: ", e$message)
    return(NULL)
  })
}

# Main function to run the entire SVM workflow
run_svm <- function(train_data, test_data, k_folds = 5, seed_value = 123) {
  message("\n====== Running SVM Workflow ======\n")
  
  # Step 1: Prepare data for SVM
  prepared_data <- prepare_for_svm(train_data, test_data)
  
  # Step 2: Train SVM model
  svm_model <- train_svm_model(prepared_data, k_folds, seed_value)
  
  # Step 3: Generate predictions
  predictions <- generate_predictions(svm_model, prepared_data$test)
  
  # Step 4: Evaluate model performance
  performance <- evaluate_svm(predictions, test_data$class)
  
  # Step 5: Visualize decision boundary (if possible)
  boundary_plot <- visualize_decision_boundary(svm_model, prepared_data$train)
  
  message("\n====== SVM Workflow Complete ======\n")
  
  # Return model and performance metrics
  return(list(
    model = svm_model,
    predictions = predictions,
    performance = performance,
    boundary_plot = boundary_plot
  ))
}

# Run the model if this script is being run directly
if(!exists("SVM_SOURCED") || !SVM_SOURCED) {
  # Check if required data is available
  if(!exists("train_data") || !exists("test_data")) {
    source("scripts/02_data_preprocessing.R")
  }
  
  # Run SVM
  svm_results <- run_svm(train_data, test_data)
  
  # Save model for later use
  saveRDS(svm_results$model, "results/models/svm/svm_model.rds")
  
  SVM_SOURCED <- TRUE
} else {
  message("svm_model.R has been sourced. Use run_svm() to train and evaluate the model.")
}