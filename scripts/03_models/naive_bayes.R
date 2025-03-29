# ======= naive_bayes.R =======
# This script implements a naive Bayes model for the German Credit dataset.
# The script handles:
# - Data preparation specific to naive Bayes
# - Model training with cross-validation
# - Prediction generation
# - Performance evaluation

# Source utility scripts
source("scripts/utils/setup.R")
source("scripts/utils/evaluation.R")

# Load required packages
library(naivebayes)  # Change from e1071 to naivebayes package

# Load preprocessed data if not already in environment
if(!exists("train_data") || !exists("test_data")) {
  source("scripts/02_data_preprocessing.R")
}

# Function to create binned features
create_bins <- function(data) {
  # Create a copy to avoid modifying the original
  binned_data <- data
  
  # Define bins for duration
  duration_breaks <- c(0, 12, 24, 36, 48, 60, Inf)
  duration_labels <- c("0-12", "12-24", "24-36", "36-48", "48-60", "60+")
  
  # Apply duration binning
  binned_data$duration_bin <- cut(binned_data$duration, 
                                 breaks = duration_breaks,
                                 labels = duration_labels)
  
  # Define bins for credit_amount
  amount_breaks <- c(0, 1000, 2000, 5000, 10000, Inf)
  amount_labels <- c("0-1K", "1K-2K", "2K-5K", "5K-10K", "10K+")
  
  # Apply credit_amount binning
  binned_data$credit_amount_bin <- cut(binned_data$credit_amount, 
                                      breaks = amount_breaks,
                                      labels = amount_labels)
  
  return(binned_data)
}

# Function to prepare data specifically for naive Bayes
prepare_for_naive_bayes <- function(train_data, test_data) {
  message("\n=== Preparing Data for Naive Bayes ===")
  
  # Create copies to avoid modifying the originals
  train_nb <- train_data
  test_nb <- test_data
  
  # Create binned versions
  train_nb <- create_bins(train_nb)
  test_nb <- create_bins(test_nb)
  
  # Debug: Print initial structure
  message("\nInitial data structure:")
  message("Training data:")
  print(str(train_nb))
  message("\nTest data:")
  print(str(test_nb))
  
  # Store feature information
  feature_info <- list()
  
  # First convert all character columns to factors
  char_cols <- names(train_nb)[sapply(train_nb, is.character)]
  for(col in char_cols) {
    train_nb[[col]] <- as.factor(train_nb[[col]])
    if(col %in% names(test_nb)) {
      test_nb[[col]] <- as.factor(test_nb[[col]])
    }
  }
  
  # Process all factor columns with careful level handling
  factor_cols <- names(train_nb)[sapply(train_nb, is.factor)]
  for(col in factor_cols) {
    if(col %in% names(test_nb)) {
      # Convert to character first to preserve all values
      train_values <- as.character(train_nb[[col]])
      test_values <- as.character(test_nb[[col]])
      
      # Get all unique values
      all_levels <- unique(c(train_values, test_values))
      
      # Store levels for later use
      feature_info[[col]] <- all_levels
      
      # Debug output before conversion
      message("\nProcessing column: ", col)
      message("Unique values in training: ", paste(unique(train_values), collapse=", "))
      message("Unique values in test: ", paste(unique(test_values), collapse=", "))
      message("Combined levels: ", paste(all_levels, collapse=", "))
      
      # Convert both datasets using the same levels
      train_nb[[col]] <- factor(train_values, levels = all_levels)
      test_nb[[col]] <- factor(test_values, levels = all_levels)
      
      # Verify conversion
      message("After conversion:")
      message("Training levels (", length(levels(train_nb[[col]])), "): ", 
              paste(levels(train_nb[[col]]), collapse=", "))
      message("Test levels (", length(levels(test_nb[[col]])), "): ", 
              paste(levels(test_nb[[col]]), collapse=", "))
      
      # Verify no NAs were introduced
      train_nas <- sum(is.na(train_nb[[col]]))
      test_nas <- sum(is.na(test_nb[[col]]))
      if(train_nas > 0 || test_nas > 0) {
        message("WARNING: NAs detected - Train: ", train_nas, ", Test: ", test_nas)
      }
    }
  }
  
  # Store numeric column names
  numeric_cols <- names(train_nb)[sapply(train_nb, is.numeric)]
  feature_info[["numeric_cols"]] <- numeric_cols
  
  # Create formula excluding raw numeric columns that have binned versions
  predictors <- setdiff(names(train_nb), 
                       c("class", "duration", "credit_amount"))
  formula_string <- paste("class ~", paste(predictors, collapse = " + "))
  model_formula <- as.formula(formula_string)
  
  message("\nFinal data structure:")
  message("Training data:")
  print(str(train_nb))
  message("\nTest data:")
  print(str(test_nb))
  
  return(list(
    train = train_nb,
    test = test_nb,
    formula = model_formula,
    feature_info = feature_info
  ))
}

# Function to train naive Bayes model
train_naive_bayes <- function(prepared_data, k_folds = 5, seed_value = 123) {
  message("\n=== Training Naive Bayes Model ===")
  
  # Extract prepared data
  train_data <- prepared_data$train
  model_formula <- prepared_data$formula
  
  # Debug: Print training data structure
  message("\nTraining data structure:")
  print(str(train_data))
  
  # Train the model with Laplace smoothing
  nb_model <- tryCatch({
    model <- naive_bayes(
      formula = model_formula,
      data = train_data,
      laplace = 1,  # Add Laplace smoothing
      usekernel = FALSE  # Disable kernel density estimation for numeric variables
    )
    
    # Store feature information
    attr(model, "feature_info") <- prepared_data$feature_info
    
    # Verify model structure
    message("\nModel structure:")
    print(str(model))
    
    model
  }, error = function(e) {
    message("Error during model training:")
    message(e$message)
    message("\nTraining data summary:")
    print(summary(train_data))
    stop("Failed to train Naive Bayes model")
  })
  
  return(nb_model)
}

# Function to generate predictions using the trained model
generate_predictions <- function(model, test_data) {
  message("\n=== Generating Predictions ===")
  
  # Debug: Print model information
  message("\nModel Information:")
  message("Model class: ", class(model))
  model_feature_names <- names(model$tables)
  message("Model features (", length(model_feature_names), "): ", 
          paste(model_feature_names, collapse=", "))
  
  # Create copy of test data to avoid modifying original
  test_subset <- test_data
  
  # Debug: Print test data information
  message("\nTest Data Information:")
  test_feature_names <- names(test_subset)
  message("Test features (", length(test_feature_names), "): ", 
          paste(test_feature_names, collapse=", "))
  
  # Find matching features
  matching_features <- intersect(model_feature_names, test_feature_names)
  message("\nMatching features (", length(matching_features), "): ", 
          paste(matching_features, collapse=", "))
  
  # Find missing features
  missing_features <- setdiff(model_feature_names, test_feature_names)
  if (length(missing_features) > 0) {
    message("\nWARNING: Missing features in test data: ", 
            paste(missing_features, collapse=", "))
  }
  
  # Extra features in test data
  extra_features <- setdiff(test_feature_names, model_feature_names)
  if (length(extra_features) > 0) {
    message("\nNOTE: Extra features in test data: ", 
            paste(extra_features, collapse=", "))
  }
  
  # Ensure test data has exactly the same features as the model
  test_subset <- test_subset[, matching_features, drop = FALSE]
  
  # Debug: Print feature levels for categorical variables
  message("\nFeature Levels Comparison:")
  for (feature in matching_features) {
    if (is.factor(test_subset[[feature]])) {
      model_levels <- levels(model$tables[[feature]])
      test_levels <- levels(test_subset[[feature]])
      message("\nFeature: ", feature)
      message("Model levels: ", paste(model_levels, collapse=", "))
      message("Test levels: ", paste(test_levels, collapse=", "))
      
      # Ensure test data has same levels as model
      if (!identical(model_levels, test_levels)) {
        message("Aligning levels for feature: ", feature)
        test_subset[[feature]] <- factor(test_subset[[feature]], levels = model_levels)
      }
    }
  }
  
  # Generate predictions
  tryCatch({
    # Debug: Print final dimensions
    message("\nFinal data dimensions:")
    message("Number of features used for prediction: ", ncol(test_subset))
    message("Number of test samples: ", nrow(test_subset))
    
    pred_class <- predict(model, test_subset)
    pred_prob <- predict(model, test_subset, type = "prob")
    
    if (is.null(pred_prob)) {
      # If probabilities are not available, create dummy probabilities
      warning("Probabilities not available from model, using dummy values")
      pred_prob <- ifelse(pred_class == "Good", 0.75, 0.25)
    }
    
    return(list(
      class = pred_class,
      prob = pred_prob,
      all_probs = if(is.matrix(pred_prob)) {
        as.data.frame(pred_prob)
      } else {
        data.frame(Bad = 1 - pred_prob, Good = pred_prob)
      }
    ))
    
  }, error = function(e) {
    message("Error details:")
    message("Error message: ", e$message)
    message("Model features: ", paste(model_feature_names, collapse=", "))
    message("Test features: ", paste(names(test_subset), collapse=", "))
    stop("Prediction error: ", e$message)
  })
}

# Function to evaluate model performance
evaluate_naive_bayes <- function(predictions, actual, output_dir = "results/models/naive_bayes") {
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
    model_name = "Naive Bayes"
  )
  dev.off()
  message("ROC curve saved to: ", file.path(output_dir, "roc_curve.png"))
  
  # Plot confusion matrix
  if(requireNamespace("ggplot2", quietly = TRUE)) {
    confusion_plot <- plot_confusion_matrix(
      performance$confusion_matrix,
      title = "Naive Bayes Confusion Matrix"
    )
    ggplot2::ggsave(
      file.path(output_dir, "confusion_matrix.png"),
      confusion_plot,
      width = 8,
      height = 6
    )
    message("Confusion matrix plot saved to: ", file.path(output_dir, "confusion_matrix.png"))
  }
  
  # Save performance results
  performance_file <- file.path(output_dir, "performance_metrics.RData")
  save(performance, file = performance_file)
  message("Performance metrics saved to: ", performance_file)
  
  # Save a summary text file
  summary_file <- file.path(output_dir, "model_summary.txt")
  sink(summary_file)
  cat("=== NAIVE BAYES MODEL SUMMARY ===\n\n")
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

# Main function to run the entire naive Bayes workflow
run_naive_bayes <- function(train_data, test_data, k_folds = 5, seed_value = 123) {
  message("\n====== Running Naive Bayes Workflow ======\n")
  
  # Check if test_data is valid
  if(is.null(test_data) || nrow(test_data) == 0) {
    # Try to load test data from file
    test_file <- "data/processed/test_data.csv"
    if(file.exists(test_file)) {
      test_data <- read.csv(test_file, stringsAsFactors = TRUE)
      message("Loaded test data from file with ", nrow(test_data), " rows")
    }
    
    if(is.null(test_data) || nrow(test_data) == 0) {
      stop("Test data is empty or could not be loaded")
    }
  }
  
  # Step 1: Prepare data for naive Bayes
  prepared_data <- prepare_for_naive_bayes(train_data, test_data)
  
  # Step 2: Train naive Bayes model
  nb_model <- train_naive_bayes(prepared_data, k_folds, seed_value)
  
  # Step 3: Generate predictions
  predictions <- generate_predictions(nb_model, test_data)
  
  # Step 4: Evaluate model performance
  performance <- evaluate_naive_bayes(predictions, test_data$class)
  
  message("\n====== Naive Bayes Workflow Complete ======\n")
  
  # Return model and performance metrics
  return(list(
    model = nb_model,
    predictions = predictions,
    performance = performance
  ))
}

# Add this new function at the end of the file, before the final if-block
predict_naive_bayes <- function(model, newdata, type = "both") {
  if(!inherits(model, "naiveBayes")) {
    stop("Model must be a naive Bayes model")
  }
  
  if(!attr(model, "model_type") == "gaussian") {
    stop("Invalid model type. Expected 'gaussian'")
  }
  
  return(generate_predictions(model, newdata, pred_type = type))
}

# Run the model if this script is being run directly
if(!exists("NAIVE_BAYES_SOURCED") || !NAIVE_BAYES_SOURCED) {
  # Check if required data is available
  if(!exists("train_data") || !exists("test_data")) {
    source("scripts/02_data_preprocessing.R")
  }
  
  # Run naive Bayes
  nb_results <- run_naive_bayes(train_data, test_data)
  
  # Save model for later use
  saveRDS(nb_results$model, "results/models/naive_bayes/naive_bayes_model.rds")
  
  NAIVE_BAYES_SOURCED <- TRUE
} else {
  message("naive_bayes.R has been sourced. Use run_naive_bayes() to train and evaluate the model.")
}