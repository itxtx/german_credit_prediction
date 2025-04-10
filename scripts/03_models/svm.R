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
  
  # Store original column names and dummy variables info
  feature_info <- list(
    original_cols = names(train_svm),
    dummy_cols = list()
  )
  
  # Ensure numeric columns are properly typed
  numeric_cols <- c("duration", "credit_amount", "installment_commitment", 
                   "residence_since", "age", "existing_credits", "num_dependents")
  
  for(col in numeric_cols) {
    if(col %in% names(train_svm)) {
      # Convert to numeric, handling any non-numeric values
      train_svm[[col]] <- as.numeric(as.character(train_svm[[col]]))
      test_svm[[col]] <- as.numeric(as.character(test_svm[[col]]))
      
      # Replace any NA values with 0
      train_svm[[col]][is.na(train_svm[[col]])] <- 0
      test_svm[[col]][is.na(test_svm[[col]])] <- 0
    }
  }
  
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
      
      # Apply scaling
      train_svm[[col]] <- scale(train_svm[[col]], 
                               center = scaling_params[[col]]$mean,
                               scale = scaling_params[[col]]$sd)
      test_svm[[col]] <- scale(test_svm[[col]], 
                              center = scaling_params[[col]]$mean,
                              scale = scaling_params[[col]]$sd)
      
      # Convert scaled values to numeric (removes attributes)
      train_svm[[col]] <- as.numeric(train_svm[[col]])
      test_svm[[col]] <- as.numeric(test_svm[[col]])
    }
  }
  
  # Step 2: One-hot encode categorical variables
  factor_cols <- names(train_svm)[sapply(train_svm, is.factor)]
  factor_cols <- setdiff(factor_cols, "class")  # Exclude target variable
  
  if(length(factor_cols) > 0) {
    message("One-hot encoding ", length(factor_cols), " categorical variables...")
    
    # Create empty data frames for the encoded data
    train_encoded <- data.frame(row.names = 1:nrow(train_svm))
    test_encoded <- data.frame(row.names = 1:nrow(test_svm))
    
    # Store dummy columns for each factor
    for(col in factor_cols) {
      # Get all possible levels from both datasets
      all_levels <- unique(c(levels(train_svm[[col]]), levels(test_svm[[col]])))
      
      # Update factor levels in both datasets
      train_svm[[col]] <- factor(train_svm[[col]], levels = all_levels)
      test_svm[[col]] <- factor(test_svm[[col]], levels = all_levels)
      
      # Create dummy variables
      train_matrix <- model.matrix(~ get(col) - 1, data = train_svm)
      test_matrix <- model.matrix(~ get(col) - 1, data = test_svm)
      
      # Fix column names
      col_names <- gsub("^get\\(col\\)", col, colnames(train_matrix))
      colnames(train_matrix) <- col_names
      colnames(test_matrix) <- col_names
      
      # Store dummy column names
      feature_info$dummy_cols[[col]] <- col_names
      
      # Add to encoded data frames
      train_encoded <- cbind(train_encoded, as.data.frame(train_matrix))
      test_encoded <- cbind(test_encoded, as.data.frame(test_matrix))
    }
    
    # Add numeric columns to encoded data frames
    for(col in numeric_cols) {
      train_encoded[[col]] <- train_svm[[col]]
      test_encoded[[col]] <- test_svm[[col]]
    }
    
    # Add class column
    train_encoded$class <- train_svm$class
    test_encoded$class <- test_svm$class
    
    # Update data frames
    train_svm <- train_encoded
    test_svm <- test_encoded
  }
  
  # Store final column names
  feature_info$final_cols <- names(train_svm)
  feature_info$scaling_params <- scaling_params
  
  # Ensure all columns are in the same order
  train_svm <- train_svm[, feature_info$final_cols]
  test_svm <- test_svm[, feature_info$final_cols]
  
  # Ensure all numeric columns are properly typed
  for(col in setdiff(names(train_svm), "class")) {
    train_svm[[col]] <- as.numeric(train_svm[[col]])
    test_svm[[col]] <- as.numeric(test_svm[[col]])
  }
  
  return(list(
    train = train_svm,
    test = test_svm,
    feature_info = feature_info
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
  if(!requireNamespace("e1071", quietly = TRUE)) {
    message("Installing required packages for SVM...")
    install.packages("e1071", repos = "https://cloud.r-project.org")
  }
  
  # Ensure all numeric columns are properly typed
  numeric_cols <- c("duration", "credit_amount", "installment_commitment", 
                   "residence_since", "age", "existing_credits", "num_dependents")
  
  for(col in numeric_cols) {
    if(col %in% names(train_data)) {
      train_data[[col]] <- as.numeric(as.character(train_data[[col]]))
    }
  }
  
  # Train a simpler SVM model directly with e1071
  message("Training SVM model with Radial Basis Function kernel...")
  start_time <- proc.time()
  
  # Train model with cross-validation
  cv_folds <- e1071::tune.control(cross = k_folds)
  
  # Tune SVM parameters
  tuned_model <- e1071::tune.svm(
    x = train_data[, setdiff(names(train_data), "class")],
    y = train_data$class,
    gamma = c(0.01, 0.05, 0.1),
    cost = c(0.1, 1, 10),
    tunecontrol = cv_folds,
    kernel = "radial",
    probability = TRUE,
    scale = FALSE  # Data already scaled
  )
  
  # Get best parameters
  best_params <- tuned_model$best.parameters
  
  # Train final model with best parameters
  final_model <- e1071::svm(
    x = train_data[, setdiff(names(train_data), "class")],
    y = train_data$class,
    kernel = "radial",
    gamma = best_params$gamma,
    cost = best_params$cost,
    probability = TRUE,
    scale = FALSE  # Data already scaled
  )
  
  # End timing
  end_time <- proc.time()
  train_time <- end_time - start_time
  
  message("Model training completed in ", round(train_time[3], 2), " seconds")
  
  # Print model summary
  message("\nModel Summary:")
  print(final_model)
  
  # Print cross-validation results
  message("\nCross-Validation Results:")
  print(tuned_model$performances)
  
  # Print best tuning parameters
  message("\nBest Tuning Parameters:")
  print(best_params)
  
  # Store feature info in the model
  attr(final_model, "feature_info") <- prepared_data$feature_info
  
  return(final_model)
}

# Function to generate predictions using the trained model
generate_predictions <- function(model, test_data) {
  message("\n=== Generating Predictions ===")
  
  # Get feature info from model
  feature_info <- attr(model, "feature_info")
  if(is.null(feature_info)) {
    stop("No feature info found in model. Cannot generate predictions.")
  }
  
  # Create a data frame with the same structure as training data
  pred_data <- data.frame(matrix(0, nrow = nrow(test_data), 
                                ncol = length(feature_info$final_cols) - 1))  # -1 for class
  names(pred_data) <- setdiff(feature_info$final_cols, "class")
  
  # Copy numeric columns
  numeric_cols <- c("duration", "credit_amount", "installment_commitment", 
                   "residence_since", "age", "existing_credits", "num_dependents")
  
  for(col in numeric_cols) {
    if(col %in% names(test_data) && col %in% names(pred_data)) {
      pred_data[[col]] <- as.numeric(as.character(test_data[[col]]))
      pred_data[[col]][is.na(pred_data[[col]])] <- 0
    }
  }
  
  # Apply scaling to numeric columns
  scaling_params <- feature_info$scaling_params
  if(!is.null(scaling_params)) {
    for(col in names(scaling_params)) {
      if(col %in% names(pred_data)) {
        pred_data[[col]] <- scale(pred_data[[col]], 
                                center = scaling_params[[col]]$mean,
                                scale = scaling_params[[col]]$sd)
        pred_data[[col]] <- as.numeric(pred_data[[col]])
      }
    }
  }
  
  # Handle dummy variables
  for(col in names(feature_info$dummy_cols)) {
    dummy_cols <- feature_info$dummy_cols[[col]]
    # Create dummy variables for the current factor
    if(col %in% names(test_data)) {
      # Get the current value
      current_value <- as.character(test_data[[col]])
      # Set all dummy columns to 0
      for(dummy_col in dummy_cols) {
        pred_data[[dummy_col]] <- 0
      }
      # Set the appropriate dummy column to 1
      for(i in seq_along(current_value)) {
        matching_col <- paste0(col, current_value[i])
        if(matching_col %in% dummy_cols) {
          pred_data[i, matching_col] <- 1
        }
      }
    }
  }
  
  # Ensure all columns are numeric
  for(col in names(pred_data)) {
    pred_data[[col]] <- as.numeric(pred_data[[col]])
  }
  
  # Ensure columns are in the same order as training data
  pred_data <- pred_data[, setdiff(feature_info$final_cols, "class")]
  
  # Generate probabilities
  message("Generating predictions...")
  pred_prob <- predict(model, pred_data, probability = TRUE)
  pred_prob <- attr(pred_prob, "probabilities")[, "Good"]
  
  # Convert to class predictions
  pred_class <- ifelse(pred_prob > 0.5, "Good", "Bad")
  pred_class <- factor(pred_class, levels = c("Bad", "Good"))
  
  return(list(
    class = pred_class,
    prob = pred_prob
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

library(e1071)
library(caret)

# Train SVM model
train_svm <- function(train_data) {
  # Train SVM with radial kernel
  svm_model <- svm(
    class ~ .,
    data = train_data,
    kernel = "radial",
    probability = TRUE,
    scale = TRUE
  )
  
  return(svm_model)
}

# Generate predictions
predict_svm <- function(model, test_data) {
  predictions <- predict(model, test_data, probability = TRUE)
  prob_predictions <- attr(predictions, "probabilities")[, "Good"]
  return(prob_predictions)
}