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

# Function to generate predictions
generate_predictions <- function(model, test_data, pred_type = "both") {
  message("\n=== Generating Predictions ===")
  
  # First apply binning to test data
  test_subset <- create_bins(test_data)
  
  message("\nTest data structure after binning:")
  print(str(test_subset))
  
  # Get the feature information stored during training
  feature_info <- attr(model, "feature_info")
  if(is.null(feature_info)) {
    stop("Model is missing feature information. Please retrain the model.")
  }
  
  # Process features and generate predictions
  tryCatch({
    # Get model features from the correct attribute
    model_features <- names(attr(model, "tables"))
    
    message("\nProcessing features for prediction:")
    # Process each feature according to its type
    for(col in model_features) {
      if(!col %in% names(test_subset)) {
        stop(paste("Missing required feature in test data:", col))
      }
      
      if(is.factor(test_subset[[col]])) {
        # Get expected levels from the model
        expected_levels <- levels(attr(model, "tables")[[col]])
        
        # Convert to character then back to factor with expected levels
        test_subset[[col]] <- factor(as.character(test_subset[[col]]), 
                                   levels = expected_levels)
        
        # Debug output
        message("\nColumn: ", col)
        message("Expected levels: ", paste(expected_levels, collapse=", "))
        message("Actual levels: ", paste(levels(test_subset[[col]]), collapse=", "))
        message("NA count: ", sum(is.na(test_subset[[col]])))
      }
    }
    
    # Keep only the features used in the model
    test_subset <- test_subset[, model_features, drop = FALSE]
    
    # Generate predictions
    results <- list()
    if(pred_type %in% c("both", "class")) {
      results$class <- predict(model, newdata = test_subset)
    }
    if(pred_type %in% c("both", "raw")) {
      results$all_probs <- predict(model, newdata = test_subset, type = "prob")
      results$prob <- results$all_probs[, "Good"]
    }
    
    message("Generated predictions for ", nrow(test_subset), " test samples")
    return(results)
    
  }, error = function(e) {
    message("\nDetailed error information:")
    message("Test data structure:")
    print(str(test_subset))
    message("\nModel features:")
    print(model_features)
    stop(paste("Error generating predictions:", e$message))
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