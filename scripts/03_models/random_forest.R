# ======= random_forest.R =======
# This script implements a random forest model for the German Credit dataset.
# The script handles:
# - Data preparation specific to random forests
# - Model training with cross-validation
# - Prediction generation
# - Performance evaluation

# Source utility scripts
source("scripts/utils/setup.R")
source("scripts/utils/evaluation.R")

# Load preprocessed data if not already in environment
if(!exists("train_data") || !exists("test_data")) {
  message("Loading preprocessed data...")
  # Check if CSV files exist
  train_file <- "data/processed/train_data.csv"
  test_file <- "data/processed/test_data.csv"
  
  if(file.exists(train_file) && file.exists(test_file)) {
    train_data <- read.csv(train_file, stringsAsFactors = TRUE)
    test_data <- read.csv(test_file, stringsAsFactors = TRUE)
    message("Data loaded from CSV files")
  } else {
    message("Processing raw data...")
    source("scripts/02_data_preprocessing.R")
  }
}

# Function to prepare data specifically for random forests
prepare_for_random_forest <- function(train_data, test_data) {
  message("\n=== Preparing Data for Random Forest ===")
  
  # Create copies to avoid modifying the originals
  train_rf <- data.frame(train_data, stringsAsFactors = TRUE)
  test_rf <- data.frame(test_data, stringsAsFactors = TRUE)
  
  # Handle factor levels to ensure consistency
  factor_cols <- names(train_rf)[sapply(train_rf, is.factor)]
  for(col in factor_cols) {
    # Get all levels from both datasets
    all_levels <- unique(c(levels(train_rf[[col]]), levels(test_rf[[col]])))
    
    # Set the levels for both datasets
    train_rf[[col]] <- factor(train_rf[[col]], levels = all_levels)
    test_rf[[col]] <- factor(test_rf[[col]], levels = all_levels)
  }
  
  # Optional: Create additional features that might be useful
  # Move feature engineering into a separate helper function
  engineer_features <- function(data) {
    # Create a copy of the data to avoid modifying the original
    data <- data.frame(data, stringsAsFactors = TRUE)
    
    # Calculate credit amount per month
    if("duration" %in% names(data) && "credit_amount" %in% names(data)) {
      data$monthly_payment <- data$credit_amount / data$duration
      
      # Handle potential Inf/NA values
      data$monthly_payment[!is.finite(data$monthly_payment)] <- 
        max(data$monthly_payment[is.finite(data$monthly_payment)], na.rm = TRUE)
    }
    
    # Age to employment ratio (stability indicator)
    if("age" %in% names(data) && "employment" %in% names(data)) {
      if(is.factor(data$employment)) {
        employment_map <- c(
          "A71" = 0.5,   # unemployed/unskilled - non-resident
          "A72" = 1,     # unskilled - resident
          "A73" = 2,     # skilled employee / official
          "A74" = 5,     # management / self-employed / highly qualified employee / officer
          "A75" = 7      # highest level
        )
        
        # Convert employment to numeric, with error handling
        data$employment_years <- employment_map[as.character(data$employment)]
        
        # Handle any NA values that might occur from unmatched levels
        if(any(is.na(data$employment_years))) {
          warning("Some employment categories couldn't be mapped. Using median value for missing cases.")
          data$employment_years[is.na(data$employment_years)] <- median(employment_map)
        }
        
        data$age_employment_ratio <- data$age / (data$employment_years + 0.1)
      }
    }
    return(data)
  }
  
  # Apply feature engineering to both datasets
  train_rf <- engineer_features(train_rf)
  test_rf <- engineer_features(test_rf)
  
  # Create a formula that includes all predictors except the class
  predictors <- setdiff(names(train_rf), "class")
  formula_string <- paste("class ~", paste(predictors, collapse = " + "))
  model_formula <- as.formula(formula_string)
  
  message("Created model formula with ", length(predictors), " predictors")
  
  return(list(
    train = train_rf,
    test = test_rf,
    formula = model_formula,
    engineer_features = engineer_features  # Return the function for later use
  ))
}

# Function to train random forest model with cross-validation
train_random_forest <- function(prepared_data, k_folds = 5, seed_value = 123) {
  message("\n=== Training Random Forest Model with Cross-Validation ===")
  
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
  
  # Create a tuning grid for mtry (number of variables randomly sampled at each split)
  # sqrt(p) is a common rule of thumb for classification, where p is the number of predictors
  p <- length(setdiff(names(train_data), "class"))
  tuning_grid <- expand.grid(
    mtry = c(floor(sqrt(p)), floor(p/3), floor(p/2))  # Try different mtry values
  )
  
  # Train the model with error handling
  tryCatch({
    message("Training random forest model...")
    
    # Start timing
    start_time <- proc.time()
    
    # Check if required packages are available
    if(!requireNamespace("randomForest", quietly = TRUE)) {
      message("Installing randomForest package...")
      install.packages("randomForest", repos = "https://cloud.r-project.org")
    }
    
    # Train the model with cross-validation and parameter tuning
    rf_model <- train(
      model_formula, 
      data = train_data,
      method = "rf",
      trControl = ctrl,
      tuneGrid = tuning_grid,
      metric = "ROC",
      importance = TRUE,      # Calculate variable importance
      ntree = 500             # Number of trees in the forest
    )
    
    # End timing
    end_time <- proc.time()
    train_time <- end_time - start_time
    
    message("Model training completed in ", round(train_time[3], 2), " seconds")
    
    # Print model summary
    message("\nModel Summary:")
    print(rf_model)
    
    # Print cross-validation results
    message("\nCross-Validation Results:")
    print(rf_model$results)
    
    # Print best tuning parameters
    message("\nBest Tuning Parameters:")
    print(rf_model$bestTune)
    
    # Print top variable importance
    message("\nTop 10 Variables by Importance:")
    if(!is.null(rf_model$finalModel)) {
      var_imp <- randomForest::importance(rf_model$finalModel)
      var_imp_df <- data.frame(
        Variable = rownames(var_imp),
        Importance = var_imp[, "MeanDecreaseGini"]
      )
      
      # Rename columns to match expected names for plotting function
      names(var_imp_df) <- c("Feature", "Gain")
      
      var_imp_df <- var_imp_df[order(var_imp_df$Gain, decreasing = TRUE), ]
      print(head(var_imp_df, 10))
    }
    
    return(rf_model)
    
  }, error = function(e) {
    message("ERROR training random forest: ", e$message)
    
    # Try with a simpler approach if the original fails
    message("Attempting with a simplified model...")
    
    # Create a simpler model with fewer trees and default mtry
    simple_model <- randomForest::randomForest(
      formula = model_formula,
      data = train_data,
      ntree = 200,
      importance = TRUE
    )
    
    message("Simplified model training completed")
    return(simple_model)
  })
}

# Function to generate predictions using the trained model
generate_predictions <- function(model, test_data, prepared_data = NULL) {
  message("\n=== Generating Predictions ===")
  
  # Validate test data
  if(is.null(test_data)) {
    stop("Test data is NULL")
  }
  
  if(!is.data.frame(test_data)) {
    test_data <- as.data.frame(test_data)
  }
  
  if(nrow(test_data) == 0) {
    stop("Test data is empty")
  }
  
  # Create a copy of test data
  test_df <- data.frame(test_data, stringsAsFactors = TRUE)
  
  # Apply feature engineering if available
  if (!is.null(prepared_data) && !is.null(prepared_data$engineer_features)) {
    message("Applying feature engineering to test data...")
    test_df <- prepared_data$engineer_features(test_df)
  }
  
  # Get required columns from model
  if (inherits(model, "train")) {
    required_cols <- setdiff(names(model$trainingData), ".outcome")
  } else {
    required_cols <- all.vars(model$terms)[-1]  # Exclude response variable
  }
  
  # Add missing columns with appropriate default values
  missing_cols <- setdiff(required_cols, names(test_df))
  if (length(missing_cols) > 0) {
    message("Adding missing columns with default values...")
    for(col in missing_cols) {
      if (inherits(model, "train")) {
        # Get column type from training data
        col_class <- class(model$trainingData[[col]])
      } else {
        # Get column type from model terms
        col_class <- class(model$model[[col]])
      }
      
      # Add column with appropriate default value
      if (col_class == "numeric") {
        test_df[[col]] <- 0
      } else if (col_class == "factor") {
        test_df[[col]] <- factor(NA, levels = levels(model$model[[col]]))
      }
    }
  }
  
  # Generate predictions
  tryCatch({
    if(inherits(model, "train")) {
      pred_class <- predict(model, newdata = test_df)
      pred_prob <- predict(model, newdata = test_df, type = "prob")
    } else {
      pred_class <- predict(model, newdata = test_df, type = "class")
      pred_prob <- predict(model, newdata = test_df, type = "prob")
    }
    
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
evaluate_random_forest <- function(predictions, actual, model, prepared_data = NULL, output_dir = "results/models/random_forest") {
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
    model_name = "Random Forest"
  )
  dev.off()
  message("ROC curve saved to: ", file.path(output_dir, "roc_curve.png"))
  
  # Plot confusion matrix
  if(requireNamespace("ggplot2", quietly = TRUE)) {
    confusion_plot <- plot_confusion_matrix(
      performance$confusion_matrix,
      title = "Random Forest Confusion Matrix"
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
    if(inherits(model, "train") && inherits(model$finalModel, "randomForest")) {
      # For caret models with randomForest
      var_imp <- randomForest::importance(model$finalModel)
      
      # MeanDecreaseGini is typically used for classification
      if("MeanDecreaseGini" %in% colnames(var_imp)) {
        var_imp_df <- data.frame(
          Variable = rownames(var_imp),
          Importance = var_imp[, "MeanDecreaseGini"]
        )
      } else {
        # Fallback to the first importance column
        var_imp_df <- data.frame(
          Variable = rownames(var_imp),
          Importance = var_imp[, 1]
        )
      }
      
      # Rename columns to match expected names for plotting function
      names(var_imp_df) <- c("Feature", "Gain")
      
      var_imp_df <- var_imp_df[order(var_imp_df$Gain, decreasing = TRUE), ]
      
      # Create and save the plot
      importance_plot <- plot_variable_importance(
        var_imp_df,
        title = "Random Forest - Variable Importance",
        max_vars = 20
      )
      ggplot2::ggsave(
        file.path(output_dir, "variable_importance.png"),
        importance_plot,
        width = 10,
        height = 8
      )
      message("Variable importance plot saved to: ", file.path(output_dir, "variable_importance.png"))
    } else if(inherits(model, "randomForest")) {
      # For direct randomForest models
      var_imp <- randomForest::importance(model)
      
      # MeanDecreaseGini is typically used for classification
      if("MeanDecreaseGini" %in% colnames(var_imp)) {
        var_imp_df <- data.frame(
          Variable = rownames(var_imp),
          Importance = var_imp[, "MeanDecreaseGini"]
        )
      } else {
        # Fallback to the first importance column
        var_imp_df <- data.frame(
          Variable = rownames(var_imp),
          Importance = var_imp[, 1]
        )
      }
      
      # Rename columns to match expected names for plotting function
      names(var_imp_df) <- c("Feature", "Gain")
      
      var_imp_df <- var_imp_df[order(var_imp_df$Gain, decreasing = TRUE), ]
      
      # Create and save the plot
      importance_plot <- plot_variable_importance(
        var_imp_df,
        title = "Random Forest - Variable Importance",
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
  
  # Plot partial dependence plots for top predictors
  if(requireNamespace("pdp", quietly = TRUE) && requireNamespace("ggplot2", quietly = TRUE)) {
    message("Generating partial dependence plots for top predictors...")
    
    # Get top 6 important variables, strip factor level indicators
    if(inherits(model, "train") && inherits(model$finalModel, "randomForest")) {
      var_imp <- randomForest::importance(model$finalModel)
      if("MeanDecreaseGini" %in% colnames(var_imp)) {
        # Strip factor level indicators (e.g., "checking_statusA14" -> "checking_status")
        clean_names <- gsub("([A-Za-z_]+).*", "\\1", rownames(var_imp))
        var_imp_clean <- aggregate(var_imp[, "MeanDecreaseGini"], 
                                 by = list(Variable = clean_names), 
                                 FUN = sum)
        top_vars <- var_imp_clean$Variable[order(var_imp_clean$x, decreasing = TRUE)][1:6]
      } else {
        clean_names <- gsub("([A-Za-z_]+).*", "\\1", rownames(var_imp))
        var_imp_clean <- aggregate(var_imp[, 1], 
                                 by = list(Variable = clean_names), 
                                 FUN = sum)
        top_vars <- var_imp_clean$Variable[order(var_imp_clean$x, decreasing = TRUE)][1:6]
      }
    } else if(inherits(model, "randomForest")) {
      var_imp <- randomForest::importance(model)
      if("MeanDecreaseGini" %in% colnames(var_imp)) {
        top_vars <- names(sort(var_imp[, "MeanDecreaseGini"], decreasing = TRUE)[1:6])
      } else {
        top_vars <- names(sort(var_imp[, 1], decreasing = TRUE)[1:6])
      }
    } else {
      # Fallback to some common important variables if we can't get them from the model
      top_vars <- c("checking_status", "duration", "credit_amount", "age")
    }
    
    # Create PDP plot directory
    pdp_dir <- file.path(output_dir, "pdp_plots")
    if(!dir.exists(pdp_dir)) {
      dir.create(pdp_dir, recursive = TRUE)
    }
    
    # Use the prepared test data if available, otherwise engineer features for test data
    if (!is.null(prepared_data) && !is.null(prepared_data$test)) {
      pdp_data <- prepared_data$test
    } else if (!is.null(prepared_data) && !is.null(prepared_data$engineer_features)) {
      pdp_data <- prepared_data$engineer_features(test_data)
    } else {
      pdp_data <- test_data  # Fallback to original test data if no preparation available
    }
    
    # Generate PDP plots for each top variable
    for(var in top_vars) {
      tryCatch({
        # Skip if variable doesn't exist in the prepared data
        if(!(var %in% names(pdp_data))) {
          message("Skipping PDP plot for variable ", var, ": variable not found in data")
          next
        }
        
        # Create partial dependence plot
        pdp_result <- pdp::partial(
          model, 
          pred.var = var, 
          train = pdp_data,
          prob = TRUE,
          plot = FALSE,
          which.class = 2
        )
        
        # Convert to data frame if not already
        pdp_df <- as.data.frame(pdp_result)
        
        # Plot with ggplot2 using proper data reference
        pdp_plot <- ggplot2::ggplot(pdp_df, ggplot2::aes(x = .data[[var]], y = .data[["yhat"]])) +
          ggplot2::geom_line(color = "blue", size = 1) +
          ggplot2::geom_point(color = "blue", size = 2) +
          ggplot2::labs(
            title = paste("Partial Dependence Plot for", var),
            x = var,
            y = "Predicted Probability of Good Credit"
          ) +
          ggplot2::theme_minimal()
        
        # Save the plot
        ggplot2::ggsave(
          file.path(pdp_dir, paste0("pdp_", var, ".png")),
          pdp_plot,
          width = 8,
          height = 6
        )
        message("PDP plot saved for variable: ", var)
      }, error = function(e) {
        message("Error creating PDP plot for variable ", var, ": ", e$message)
      })
    }
  }
  
  # Save performance results
  performance_file <- file.path(output_dir, "performance_metrics.RData")
  save(performance, file = performance_file)
  message("Performance metrics saved to: ", performance_file)
  
  # Save a summary text file
  summary_file <- file.path(output_dir, "model_summary.txt")
  sink(summary_file)
  cat("=== RANDOM FOREST MODEL SUMMARY ===\n\n")
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
    cat("Best mtry: ", model$bestTune$mtry, "\n")
    cat("Number of Trees: ", model$finalModel$ntree, "\n")
    cat("OOB Error Rate: ", model$finalModel$err.rate[model$finalModel$ntree, "OOB"], "\n")
  } else if(inherits(model, "randomForest")) {
    cat("mtry: ", model$mtry, "\n")
    cat("Number of Trees: ", model$ntree, "\n")
    cat("OOB Error Rate: ", model$err.rate[model$ntree, "OOB"], "\n")
  }
  
  cat("\nTOP 10 VARIABLES BY IMPORTANCE:\n")
  if(inherits(model, "train") && inherits(model$finalModel, "randomForest")) {
    var_imp <- randomForest::importance(model$finalModel)
    var_imp_sorted <- sort(var_imp[, "MeanDecreaseGini"], decreasing = TRUE)
    print(head(var_imp_sorted, 10))
  } else if(inherits(model, "randomForest")) {
    var_imp <- randomForest::importance(model)
    var_imp_sorted <- sort(var_imp[, "MeanDecreaseGini"], decreasing = TRUE)
    print(head(var_imp_sorted, 10))
  }
  
  sink()
  message("Model summary saved to: ", summary_file)
  
  return(performance)
}

# Main function to run the entire random forest workflow
run_random_forest <- function(train_data, test_data, k_folds = 5, seed_value = 123) {
  message("\n====== Running Random Forest Workflow ======\n")
  
  # Add data validation
  if(nrow(test_data) == 0 || ncol(test_data) == 0) {
    stop("Test data is empty. Please check data loading.")
  }
  
  message("Initial test data dimensions: ", nrow(test_data), " x ", ncol(test_data))
  
  # Step 1: Prepare data for random forest
  prepared_data <- prepare_for_random_forest(train_data, test_data)
  
  # Validate prepared data
  if(is.null(prepared_data$test) || nrow(prepared_data$test) == 0) {
    stop("Prepared test data is empty. Check prepare_for_random_forest function.")
  }
  
  message("Prepared test data dimensions: ", nrow(prepared_data$test), " x ", ncol(prepared_data$test))
  
  # Step 2: Train random forest model
  rf_model <- train_random_forest(prepared_data, k_folds, seed_value)
  
  # Step 3: Generate predictions using the prepared test data
  predictions <- generate_predictions(rf_model, prepared_data$test, prepared_data)
  
  # Step 4: Evaluate model performance
  performance <- evaluate_random_forest(predictions, test_data$class, rf_model, prepared_data)
  
  message("\n====== Random Forest Workflow Complete ======\n")
  
  return(list(
    model = rf_model,
    predictions = predictions,
    performance = performance,
    prepared_data = prepared_data
  ))
}

# Run the model if this script is being run directly
if(!exists("RANDOM_FOREST_SOURCED") || !RANDOM_FOREST_SOURCED) {
  # Run random forest
  rf_results <- run_random_forest(train_data, test_data)
  
  # Save model for later use
  saveRDS(rf_results$model, "results/models/random_forest/random_forest_model.rds")
  
  RANDOM_FOREST_SOURCED <- TRUE
} else {
  message("random_forest.R has been sourced. Use run_random_forest() to train and evaluate the model.")
}