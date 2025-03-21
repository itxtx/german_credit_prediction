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

# Function to prepare data specifically for naive Bayes
prepare_for_naive_bayes <- function(train_data, test_data) {
  message("\n=== Preparing Data for Naive Bayes ===")
  
  # Create copies to avoid modifying the originals
  train_nb <- train_data
  test_nb <- test_data
  
  # For naive Bayes, we should:
  # 1. Ensure all categorical variables are properly factorized
  # 2. Avoid transformations that might affect probability distributions
  # 3. Discretize continuous variables if needed (optional)
  
  # Ensure all categorical variables are factors
  factor_cols <- names(train_nb)[sapply(train_nb, is.character)]
  if(length(factor_cols) > 0) {
    message("Converting character columns to factors...")
    for(col in factor_cols) {
      train_nb[[col]] <- as.factor(train_nb[[col]])
      test_nb[[col]] <- as.factor(test_nb[[col]])
    }
  }
  
  # Align factor levels between training and test sets
  factor_cols <- names(train_nb)[sapply(train_nb, is.factor)]
  for(col in factor_cols) {
    # Get all levels from both datasets
    all_levels <- unique(c(levels(train_nb[[col]]), levels(test_nb[[col]])))
    
    # Set the levels for both datasets
    levels(train_nb[[col]]) <- all_levels
    levels(test_nb[[col]]) <- all_levels
  }
  
  # Naive Bayes performs better when continuous variables are discretized (optional)
  # Here we'll check if discretization might be beneficial
  numeric_cols <- names(train_nb)[sapply(train_nb, is.numeric)]
  if(length(numeric_cols) > 0) {
    message("Found ", length(numeric_cols), " numeric columns that could potentially be discretized")
    
    # For demonstration, we'll discretize one of the most important numeric variables: duration
    if("duration" %in% numeric_cols) {
      message("Discretizing 'duration' into bins...")
      
      # Define bins for duration
      duration_breaks <- c(0, 12, 24, 36, 48, 60, Inf)
      duration_labels <- c("0-12", "12-24", "24-36", "36-48", "48-60", "60+")
      
      # Apply discretization
      train_nb$duration_bin <- cut(train_nb$duration, 
                                  breaks = duration_breaks,
                                  labels = duration_labels)
      test_nb$duration_bin <- cut(test_nb$duration, 
                                 breaks = duration_breaks,
                                 labels = duration_labels)
      
      # Note: We're keeping the original duration as well
    }
    
    # Similarly for credit_amount
    if("credit_amount" %in% numeric_cols) {
      message("Discretizing 'credit_amount' into bins...")
      
      # Define bins for credit_amount
      amount_breaks <- c(0, 1000, 2000, 5000, 10000, Inf)
      amount_labels <- c("0-1K", "1K-2K", "2K-5K", "5K-10K", "10K+")
      
      # Apply discretization
      train_nb$credit_amount_bin <- cut(train_nb$credit_amount, 
                                      breaks = amount_breaks,
                                      labels = amount_labels)
      test_nb$credit_amount_bin <- cut(test_nb$credit_amount, 
                                     breaks = amount_breaks,
                                     labels = amount_labels)
    }
  }
  
  # Create a formula that includes all predictors except the class
  predictors <- setdiff(names(train_nb), "class")
  formula_string <- paste("class ~", paste(predictors, collapse = " + "))
  model_formula <- as.formula(formula_string)
  
  message("Created model formula with ", length(predictors), " predictors")
  
  return(list(
    train = train_nb,
    test = test_nb,
    formula = model_formula
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
    message("Training naive Bayes model...")
    
    # Start timing
    start_time <- proc.time()
    
    # Check if e1071 is available
    if(!requireNamespace("e1071", quietly = TRUE)) {
      message("Installing e1071 package...")
      install.packages("e1071", repos = "https://cloud.r-project.org")
    }
    
    # Train the model
    nb_model <- train(
      model_formula, 
      data = train_data,
      method = "naive_bayes",
      trControl = ctrl,
      metric = "ROC",
      tuneLength = 5,         # Try 5 different Laplace smoothing values
      preProcess = NULL       # No preprocessing needed for Naive Bayes
    )
    
    # End timing
    end_time <- proc.time()
    train_time <- end_time - start_time
    
    message("Model training completed in ", round(train_time[3], 2), " seconds")
    
    # Print model summary
    message("\nModel Summary:")
    print(nb_model)
    
    # Print cross-validation results
    message("\nCross-Validation Results:")
    print(nb_model$results)
    
    # Print best tuning parameters
    message("\nBest Tuning Parameters:")
    print(nb_model$bestTune)
    
    return(nb_model)
    
  }, error = function(e) {
    message("ERROR training naive Bayes: ", e$message)
    
    # Try with a simpler formula if the original fails
    message("Attempting with a simplified model...")
    
    # Create a simpler formula with fewer predictors
    simple_formula <- as.formula("class ~ checking_status + duration + credit_history + purpose + credit_amount")
    
    # Train a simpler model
    simple_model <- train(
      simple_formula, 
      data = train_data,
      method = "naive_bayes",
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