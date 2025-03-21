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

# Load preprocessed data if not already in environment
if(!exists("train_data") || !exists("test_data")) {
  source("scripts/02_data_preprocessing.R")
}

# Function to prepare data specifically for decision trees
prepare_for_decision_tree <- function(train_data, test_data) {
  message("\n=== Preparing Data for Decision Tree ===")
  
  # Create copies to avoid modifying the originals
  train_dt <- train_data
  test_dt <- test_data
  
  # For decision trees:
  # 1. No need to standardize numeric variables
  # 2. No need for one-hot encoding of categorical variables
  # 3. No special handling for outliers (trees are robust to them)
  # 4. Might want to create interaction features (optional)
  
  # Handle factor levels to ensure consistency
  factor_cols <- names(train_dt)[sapply(train_dt, is.factor)]
  for(col in factor_cols) {
    # Get all levels from both datasets
    all_levels <- unique(c(levels(train_dt[[col]]), levels(test_dt[[col]])))
    
    # Set the levels for both datasets
    levels(train_dt[[col]]) <- all_levels
    levels(test_dt[[col]]) <- all_levels
  }
  
  # Decision trees can benefit from creating interaction features, especially
  # between key variables. For example:
  if("duration" %in% names(train_dt) && "credit_amount" %in% names(train_dt)) {
    message("Creating duration-to-amount ratio feature...")
    train_dt$duration_to_amount <- train_dt$duration / (train_dt$credit_amount + 1)  # Add 1 to avoid division by zero
    test_dt$duration_to_amount <- test_dt$duration / (test_dt$credit_amount + 1)
  }
  
  # Age categories might be more informative than continuous age
  if("age" %in% names(train_dt)) {
    message("Creating age category feature...")
    train_dt$age_cat <- cut(train_dt$age,
                           breaks = c(0, 25, 40, 60, 100),
                           labels = c("Young", "Adult", "Middle", "Senior"))
    test_dt$age_cat <- cut(test_dt$age,
                          breaks = c(0, 25, 40, 60, 100),
                          labels = c("Young", "Adult", "Middle", "Senior"))
  }
  
  # Create a formula that includes all predictors except the class
  predictors <- setdiff(names(train_dt), "class")
  formula_string <- paste("class ~", paste(predictors, collapse = " + "))
  model_formula <- as.formula(formula_string)
  
  message("Created model formula with ", length(predictors), " predictors")
  
  return(list(
    train = train_dt,
    test = test_dt,
    formula = model_formula
  ))
}

# Function to train decision tree model with cross-validation
train_decision_tree <- function(prepared_data, k_folds = 5, seed_value = 123) {
  message("\n=== Training Decision Tree Model with Cross-Validation ===")
  
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
  
  # Set tuning grid for cp (complexity parameter)
  tuning_grid <- expand.grid(
    cp = seq(0.001, 0.05, by = 0.005)  # Try 10 different cp values
  )
  
  # Train the model with error handling
  tryCatch({
    message("Training decision tree model...")
    
    # Start timing
    start_time <- proc.time()
    
    # Check if required packages are available
    if(!requireNamespace("rpart", quietly = TRUE)) {
      message("Installing rpart package...")
      install.packages("rpart", repos = "https://cloud.r-project.org")
    }
    
    # Train the model with cross-validation and parameter tuning
    tree_model <- train(
      model_formula, 
      data = train_data,
      method = "rpart",
      trControl = ctrl,
      tuneGrid = tuning_grid,
      metric = "ROC"
    )
    
    # End timing
    end_time <- proc.time()
    train_time <- end_time - start_time
    
    message("Model training completed in ", round(train_time[3], 2), " seconds")
    
    # Print model summary
    message("\nModel Summary:")
    print(tree_model)
    
    # Print cross-validation results
    message("\nCross-Validation Results:")
    print(tree_model$results)
    
    # Print best tuning parameters
    message("\nBest Tuning Parameters:")
    print(tree_model$bestTune)
    
    # Get variable importance
    if(requireNamespace("rpart", quietly = TRUE)) {
      message("\nVariable Importance:")
      var_imp <- rpart::rpart.object.size(tree_model$finalModel)
      print(sort(tree_model$finalModel$variable.importance, decreasing = TRUE))
    }
    
    return(tree_model)
    
  }, error = function(e) {
    message("ERROR training decision tree: ", e$message)
    
    # Try with a simpler approach if the original fails
    message("Attempting with a simplified model...")
    
    # Create a simpler model with minimal settings
    simple_model <- rpart::rpart(
      formula = model_formula,
      data = train_data,
      method = "class",
      control = rpart::rpart.control(cp = 0.01)
    )
    
    message("Simplified model training completed")
    return(simple_model)
  })
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
  
  # Handle different model types
  if(inherits(model, "train")) {
    # Caret's train object
    # Generate class predictions
    pred_class <- predict(model, newdata = test_data)
    
    # Generate probability predictions
    pred_prob <- predict(model, newdata = test_data, type = "prob")
  } else if(inherits(model, "rpart")) {
    # Direct rpart object
    # Generate class predictions
    pred_class <- predict(model, newdata = test_data, type = "class")
    
    # Generate probability predictions
    pred_prob_matrix <- predict(model, newdata = test_data, type = "prob")
    # Create a data frame with appropriate column names
    pred_prob <- data.frame(pred_prob_matrix)
    if(ncol(pred_prob) == 2) {
      colnames(pred_prob) <- c("Bad", "Good")
    }
  } else {
    stop("Unknown model type. Cannot generate predictions.")
  }
  
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
evaluate_decision_tree <- function(predictions, actual, model, output_dir = "results/models/decision_tree") {
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

# Main function to run the entire decision tree workflow
run_decision_tree <- function(train_data, test_data, k_folds = 5, seed_value = 123) {
  message("\n====== Running Decision Tree Workflow ======\n")
  
  # Step 1: Prepare data for decision tree
  prepared_data <- prepare_for_decision_tree(train_data, test_data)
  
  # Step 2: Train decision tree model
  tree_model <- train_decision_tree(prepared_data, k_folds, seed_value)
  
  # Step 3: Visualize the decision tree
  plot_decision_tree(tree_model)
  
  # Step 4: Generate predictions
  predictions <- generate_predictions(tree_model, test_data)
  
  # Step 5: Evaluate model performance
  performance <- evaluate_decision_tree(predictions, test_data$class, tree_model)
  
  message("\n====== Decision Tree Workflow Complete ======\n")
  
  # Return model and performance metrics
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
  tree_results <- run_decision_tree(train_data, test_data)
  
  # Save model for later use
  saveRDS(tree_results$model, "results/models/decision_tree/decision_tree_model.rds")
  
  DECISION_TREE_SOURCED <- TRUE
} else {
  message("decision_tree.R has been sourced. Use run_decision_tree() to train and evaluate the model.")
}