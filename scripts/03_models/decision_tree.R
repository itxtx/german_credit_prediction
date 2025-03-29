# ======= decision_tree.R =======
# This script implements a decision tree model for the German Credit dataset.
# The script handles:
# - Data preparation specific to decision trees
# - Model training with cross-validation
# - Prediction generation
# - Performance evaluation

# Source utility scripts
source("scripts/utils/setup.R")
source("scripts/utils/evaluation.R")

# Load and validate data
load_and_validate_data <- function() {
  message("Loading and validating data...")
  
  # Try to load data from CSV if not in environment
  if(!exists("train_data") || !exists("test_data")) {
    message("Loading data from CSV files...")
    train_data <- read.csv("data/processed/train_data.csv", stringsAsFactors = TRUE)
    test_data <- read.csv("data/processed/test_data.csv", stringsAsFactors = TRUE)
    
    # Validate required columns
    required_columns <- c("checking_status", "class")
    missing_columns <- required_columns[!required_columns %in% names(train_data)]
    
    if(length(missing_columns) > 0) {
      stop("Missing required columns in training data: ", 
           paste(missing_columns, collapse = ", "))
    }
    
    missing_columns <- required_columns[!required_columns %in% names(test_data)]
    if(length(missing_columns) > 0) {
      stop("Missing required columns in test data: ", 
           paste(missing_columns, collapse = ", "))
    }
    
    message("Data loaded successfully")
    message("Training data dimensions: ", paste(dim(train_data), collapse = " x "))
    message("Test data dimensions: ", paste(dim(test_data), collapse = " x "))
    
    # Return the loaded data
    return(list(train_data = train_data, test_data = test_data))
  } else {
    message("Using existing data from environment")
    return(list(train_data = train_data, test_data = test_data))
  }
}

# Replace the existing data loading check with:
data_list <- load_and_validate_data()
train_data <- data_list$train_data
test_data <- data_list$test_data

# Add this new function near the top of the file
engineer_features <- function(data) {
  data_prepared <- data
  
  # Create duration-to-amount ratio
  if("duration" %in% names(data) && "credit_amount" %in% names(data)) {
    message("Creating duration-to-amount ratio feature...")
    data_prepared$duration_to_amount <- data$duration / (data$credit_amount + 1)
  }
  
  # Create age categories
  if("age" %in% names(data)) {
    message("Creating age category feature...")
    data_prepared$age_cat <- cut(data$age,
                                breaks = c(0, 25, 40, 60, 100),
                                labels = c("Young", "Adult", "Middle", "Senior"))
  }
  
  return(data_prepared)
}

# Simplify prepare_for_decision_tree function
prepare_for_decision_tree <- function(train_data, test_data) {
  message("\n=== Preparing Data for Decision Tree ===")
  
  # Validate that all columns in train_data exist in test_data
  train_cols <- names(train_data)
  test_cols <- names(test_data)
  missing_cols <- setdiff(train_cols, test_cols)
  
  if(length(missing_cols) > 0) {
    stop("Test data is missing columns that exist in training data: ",
         paste(missing_cols, collapse = ", "))
  }
  
  # Create copies
  train_dt <- train_data
  test_dt <- test_data
  
  # Simple imputation for both train and test
  for(col in names(train_dt)) {
    if(is.numeric(train_dt[[col]])) {
      # For numeric columns, replace NA with mean from training data
      col_mean <- mean(train_dt[[col]], na.rm = TRUE)
      train_dt[[col]][is.na(train_dt[[col]])] <- col_mean
      test_dt[[col]][is.na(test_dt[[col]])] <- col_mean
    } else if(is.factor(train_dt[[col]])) {
      # For factor columns, replace NA with mode from training data
      # Remove NA values before calculating mode
      mode_val <- names(sort(table(train_dt[[col]][!is.na(train_dt[[col]])]), decreasing = TRUE))[1]
      train_dt[[col]][is.na(train_dt[[col]])] <- mode_val
      test_dt[[col]][is.na(test_dt[[col]])] <- mode_val
    }
  }
  
  # Create simple formula
  predictors <- setdiff(names(train_dt), "class")
  formula_string <- paste("class ~", paste(predictors, collapse = " + "))
  
  return(list(
    train = train_dt,
    test = test_dt,
    formula = as.formula(formula_string)
  ))
}

# Simplify train_decision_tree function
train_decision_tree <- function(prepared_data, k_folds = 5, seed_value = 123) {
  message("\n=== Training Decision Tree Model ===")
  set.seed(seed_value)
  
  # Create simple rpart model
  model <- rpart::rpart(
    formula = prepared_data$formula,
    data = prepared_data$train,
    method = "class"
  )
  
  return(model)
}

# Function to visualize the decision tree
plot_decision_tree <- function(model, output_dir = "results/models/decision_tree") {
  message("\n=== Visualizing Decision Tree ===")
  
  # Check if output directory exists
  if(!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    message("Created directory: ", output_dir)
  }
  
  # Check if required packages are available
  if(!requireNamespace("rpart.plot", quietly = TRUE)) {
    message("Installing rpart.plot package...")
    install.packages("rpart.plot", repos = "https://cloud.r-project.org")
  }
  
  # Extract the final model (for caret models)
  if(inherits(model, "train")) {
    final_model <- model$finalModel
  } else {
    final_model <- model
  }
  
  # Create plot
  png(file.path(output_dir, "decision_tree.png"), width = 1200, height = 800, res = 120)
  rpart.plot::rpart.plot(
    final_model,
    box.palette = "RdBu",      # Red-Blue palette
    shadow.col = "gray",       # Shadow color
    nn = TRUE,                 # Display the node numbers
    fallen.leaves = TRUE,      # Put all leaves at the bottom
    main = "Decision Tree for German Credit Data"
  )
  dev.off()
  message("Decision tree plot saved to: ", file.path(output_dir, "decision_tree.png"))
  
  # Create a simpler plot for better visibility in reports
  png(file.path(output_dir, "decision_tree_simple.png"), width = 1000, height = 600, res = 100)
  rpart.plot::rpart.plot(
    final_model,
    type = 2,                  # Use 'type = 2' for simpler node display
    extra = 104,               # Show class probabilities and percentages
    box.palette = "auto",      # Automatic color based on node prediction
    branch = 0.5,              # Make branches narrower
    main = "Simplified Decision Tree for German Credit Data"
  )
  dev.off()
  message("Simplified decision tree plot saved to: ", file.path(output_dir, "decision_tree_simple.png"))
  
  return(TRUE)
}

# Function to generate predictions using the trained model
generate_predictions <- function(model, test_data) {
  message("\n=== Generating Predictions ===")
  
  # Create a copy of test data
  test_df <- data.frame(test_data, stringsAsFactors = TRUE)
  
  # Ensure all required columns exist
  required_cols <- all.vars(model$terms)[-1]  # Exclude response variable
  missing_cols <- setdiff(required_cols, names(test_df))
  
  if(length(missing_cols) > 0) {
    stop("Missing required columns in test data: ", paste(missing_cols, collapse = ", "))
  }
  
  # Generate predictions
  tryCatch({
    pred_class <- predict(model, newdata = test_df, type = "class")
    pred_prob <- predict(model, newdata = test_df, type = "prob")
    
    message("Generated predictions for ", length(pred_class), " test samples")
    
    return(list(
      class = pred_class,
      prob = pred_prob[, "Good"],
      all_probs = pred_prob
    ))
  }, error = function(e) {
    message("Error details:")
    message("Available columns: ", paste(names(test_df), collapse = ", "))
    message("Required columns: ", paste(required_cols, collapse = ", "))
    stop("ERROR generating predictions: ", e$message)
  })
}

# Function to evaluate model performance
evaluate_decision_tree <- function(predictions, actual, model, output_dir = "results/models/decision_tree") {
  message("\n=== Evaluating Model Performance ===")
  
  # Add input validation and debugging
  message("Checking input data dimensions:")
  message("Length of predictions: ", length(predictions$class))
  message("Length of actual values: ", length(actual))
  
  # Ensure predictions and actual values are factors with the same levels
  pred_class <- factor(predictions$class, levels = c("Bad", "Good"))
  actual_class <- factor(actual, levels = c("Bad", "Good"))
  
  # Verify probability predictions
  if(is.null(predictions$prob) || length(predictions$prob) != length(actual)) {
    message("WARNING: Issue with probability predictions. Length of probabilities: ", 
            length(predictions$prob))
  }
  
  # Create output directory if it doesn't exist
  if(!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    message("Created directory: ", output_dir)
  }
  
  # Evaluate model using the evaluation.R utility with error handling
  tryCatch({
    performance <- evaluate_model(
      pred = pred_class,
      actual = actual_class,
      pred_prob = predictions$prob,
      positive_class = "Good"
    )
  }, error = function(e) {
    message("Error in evaluate_model: ", e$message)
    # Return a minimal performance object if evaluation fails
    return(list(
      accuracy = NA,
      precision = NA,
      recall = NA,
      f1 = NA,
      auc = NA,
      confusion_matrix = table(Predicted = pred_class, Actual = actual_class)
    ))
  })
  
  # Print performance metrics
  message("\nPerformance Metrics:")
  metrics <- c("accuracy", "precision", "recall", "f1", "auc")
  for(metric in all_of(metrics)) {
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
    model_name = "Decision Tree"
  )
  dev.off()
  message("ROC curve saved to: ", file.path(output_dir, "roc_curve.png"))
  
  # Plot confusion matrix
  if(requireNamespace("ggplot2", quietly = TRUE)) {
    confusion_plot <- plot_confusion_matrix(
      performance$confusion_matrix,
      title = "Decision Tree Confusion Matrix"
    )
    ggplot2::ggsave(
      file.path(output_dir, "confusion_matrix.png"),
      confusion_plot,
      width = 8,
      height = 6
    )
    message("Confusion matrix plot saved to: ", file.path(output_dir, "confusion_matrix.png"))
  }
  
  # Plot variable importance
  if(requireNamespace("ggplot2", quietly = TRUE)) {
    # Extract variable importance
    if(inherits(model, "train")) {
      # For caret models
      if(!is.null(model$finalModel$variable.importance)) {
        var_imp <- model$finalModel$variable.importance
        var_imp_df <- data.frame(
          Variable = names(var_imp),
          Importance = as.numeric(var_imp)
        )
        var_imp_df <- var_imp_df[order(var_imp_df$Importance, decreasing = TRUE), ]
        
        # Rename columns to match expected format
        names(var_imp_df) <- c("Feature", "Gain")
        
        # Create and save the plot
        importance_plot <- plot_variable_importance(
          var_imp_df,
          title = "Decision Tree - Variable Importance",
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
    } else if(inherits(model, "rpart")) {
      # For direct rpart models
      if(!is.null(model$variable.importance)) {
        var_imp <- model$variable.importance
        var_imp_df <- data.frame(
          Variable = names(var_imp),
          Importance = as.numeric(var_imp)
        )
        var_imp_df <- var_imp_df[order(var_imp_df$Importance, decreasing = TRUE), ]
        
        # Rename columns to match expected format
        names(var_imp_df) <- c("Feature", "Gain")
        
        # Create and save the plot
        importance_plot <- plot_variable_importance(
          var_imp_df,
          title = "Decision Tree - Variable Importance",
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
    }
  }
  
  # Save performance results
  performance_file <- file.path(output_dir, "performance_metrics.RData")
  save(performance, file = performance_file)
  message("Performance metrics saved to: ", performance_file)
  
  # Save a summary text file
  summary_file <- file.path(output_dir, "model_summary.txt")
  sink(summary_file)
  cat("=== DECISION TREE MODEL SUMMARY ===\n\n")
  cat("Date: ", as.character(Sys.Date()), "\n\n")
  
  cat("PERFORMANCE METRICS:\n")
  for(metric in all_of(metrics)) {
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
  if(inherits(model, "train")) {
    cat("Best Complexity Parameter (cp): ", model$bestTune$cp, "\n")
    cat("Number of Terminal Nodes: ", length(model$finalModel$where), "\n")
  } else if(inherits(model, "rpart")) {
    cat("Complexity Parameter (cp): ", model$cptable[1, "CP"], "\n")
    cat("Number of Terminal Nodes: ", length(unique(model$where)), "\n")
  }
  
  sink()
  message("Model summary saved to: ", summary_file)
  
  return(performance)
}

# Simplify main workflow
run_decision_tree <- function(train_data, test_data) {
  message("\n====== Running Decision Tree Workflow ======\n")
  
  # Prepare data
  prepared_data <- prepare_for_decision_tree(train_data, test_data)
  
  # Train model
  tree_model <- train_decision_tree(prepared_data)
  
  # Generate predictions using prepared test data
  predictions <- generate_predictions(tree_model, test_data)
  
  # Evaluate model
  performance <- evaluate_decision_tree(predictions, test_data$class, tree_model)
  
  return(list(
    model = tree_model,
    predictions = predictions,
    performance = performance
  ))
}

# Run the model if this script is being run directly
if(!exists("DECISION_TREE_SOURCED") || !DECISION_TREE_SOURCED) {
  # Check if required data is available
  if(!exists("train_data") || !exists("test_data")) {
    source("scripts/02_data_preprocessing.R")
  }
  
  # Run decision tree
  results <- run_decision_tree(train_data, test_data)
  
  # Save model for later use
  saveRDS(results$model, "results/models/decision_tree/decision_tree_model.rds")
  
  DECISION_TREE_SOURCED <- TRUE
} else {
  message("decision_tree.R has been sourced. Use run_decision_tree() to train and evaluate the model.")
}