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
  
  # Apply binning to both training and test data
  train_nb <- create_bins(train_nb)
  test_nb <- create_bins(test_nb)
  
  # Store original feature information
  feature_info <- list()
  
  # Ensure all categorical variables are factors
  factor_cols <- names(train_nb)[sapply(train_nb, is.character)]
  if(length(factor_cols) > 0) {
    message("Converting character columns to factors...")
    for(col in factor_cols) {
      train_nb[[col]] <- as.factor(train_nb[[col]])
      if(col %in% names(test_nb)) {
        test_nb[[col]] <- as.factor(test_nb[[col]])
      }
    }
  }
  
  # Process all factor columns and store their levels
  factor_cols <- names(train_nb)[sapply(train_nb, is.factor)]
  for(col in factor_cols) {
    if(col %in% names(test_nb)) {
      # Get all levels from both datasets
      all_levels <- unique(c(levels(train_nb[[col]]), levels(test_nb[[col]])))
      
      # Store levels for later use
      feature_info[[col]] <- all_levels
      
      # Set the levels for both datasets
      train_nb[[col]] <- factor(train_nb[[col]], levels = all_levels)
      test_nb[[col]] <- factor(test_nb[[col]], levels = all_levels)
    }
  }
  
  # Store numeric column names
  numeric_cols <- names(train_nb)[sapply(train_nb, is.numeric)]
  feature_info[["numeric_cols"]] <- numeric_cols
  
  # Create a formula that includes all predictors except the class
  predictors <- setdiff(names(train_nb), "class")
  formula_string <- paste("class ~", paste(predictors, collapse = " + "))
  model_formula <- as.formula(formula_string)
  
  message("Created model formula with ", length(predictors), " predictors")
  
  return(list(
    train = train_nb,
    test = test_nb,
    formula = model_formula,
    feature_info = feature_info
  ))
}

# Function to train naive Bayes model with cross-validation
train_naive_bayes <- function(prepared_data, k_folds = 5, seed_value = 123) {
  message("\n=== Training Naive Bayes Model with Cross-Validation ===")
  
  # Set seed for reproducibility
  set.seed(seed_value)
  
  # Extract prepared data
  train_data <- prepared_data$train
  model_formula <- prepared_data$formula
  
  # Add check for missing values before training
  train_data <- na.omit(train_data)  # Remove any rows with missing values
  
  # Now train the model with Laplace smoothing
  nb_model <- naive_bayes(
    formula = model_formula,
    data = train_data,
    laplace = 1  # Add Laplace smoothing
  )
  
  # Store feature information in the model
  nb_model$feature_info <- prepared_data$feature_info
  
  return(nb_model)
}

# Function to generate predictions using the trained model
generate_predictions <- function(model, test_data) {
  message("\n=== Generating Predictions ===")
  
  # First apply binning to test data
  test_subset <- create_bins(test_data)
  
  # Get the feature information stored during training
  feature_info <- model$feature_info
  if(is.null(feature_info)) {
    stop("Model is missing feature information. Please retrain the model.")
  }
  
  # Ensure test data has exactly the same features as training data
  model_features <- names(model$tables)
  
  # Process each feature according to its type
  for(col in model_features) {
    if(col %in% names(feature_info)) {  # If it's a factor column
      if(!col %in% names(test_subset)) {
        stop(paste("Missing required feature in test data:", col))
      }
      
      # Convert to factor with exactly the same levels as in training
      test_subset[[col]] <- factor(test_subset[[col]], 
                                  levels = feature_info[[col]])
      
      message("Aligned levels for feature: ", col)
    } else if(col %in% feature_info$numeric_cols) {  # If it's a numeric column
      if(!is.numeric(test_subset[[col]])) {
        test_subset[[col]] <- as.numeric(as.character(test_subset[[col]]))
      }
    }
  }
  
  # Keep only the features used in the model
  test_subset <- test_subset[, model_features, drop = FALSE]
  
  # Generate predictions
  pred_class <- predict(model, newdata = test_subset, type = "class")
  pred_prob <- predict(model, newdata = test_subset, type = "prob")
  
  # Extract probability for positive class
  pos_class_prob <- pred_prob[, "Good"]
  
  message("Generated predictions for ", length(pred_class), " test samples")
  
  return(list(
    class = pred_class,
    prob = pos_class_prob,
    all_probs = pred_prob
  ))
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