# ======= evaluation.R =======
# This script contains functions for model evaluation, including:
# - Model performance metrics calculation (accuracy, precision, recall, F1, etc.)
# - ROC curve plotting and AUC calculation
# - Model comparison utilities

# Load required libraries if not already loaded
required_packages <- c("caret", "ROCR", "pROC", "ggplot2")
for(pkg in required_packages) {
  if(!requireNamespace(pkg, quietly = TRUE)) {
    message(paste("Loading package:", pkg))
    library(pkg, character.only = TRUE)
  }
}

# Function to evaluate model performance with comprehensive metrics
evaluate_model <- function(pred, actual, pred_prob = NULL, positive_class = "Good") {
  tryCatch({
    # Ensure both pred and actual are factors with the same levels
    if(!is.factor(pred)) pred <- as.factor(pred)
    if(!is.factor(actual)) actual <- as.factor(actual)
    
    # Check and align levels
    if(!identical(levels(pred), levels(actual))) {
      message("WARNING: Prediction levels don't match actual levels.")
      message("Pred levels: ", paste(levels(pred), collapse=", "))
      message("Actual levels: ", paste(levels(actual), collapse=", "))
      
      # Force the same levels
      all_levels <- unique(c(levels(pred), levels(actual)))
      levels(pred) <- all_levels
      levels(actual) <- all_levels
    }
    
    # Make sure positive class is in the levels
    if(!(positive_class %in% levels(actual))) {
      stop(paste("Positive class", positive_class, "not found in the levels of actual data"))
    }
    
    # Confusion Matrix
    conf_matrix <- caret::confusionMatrix(pred, actual, positive = positive_class)
    
    # Extract basic metrics
    accuracy <- conf_matrix$overall["Accuracy"]
    
    # Handle cases where some metrics might be NA
    precision <- if("Pos Pred Value" %in% names(conf_matrix$byClass)) 
                   conf_matrix$byClass["Pos Pred Value"] else NA
    recall <- if("Sensitivity" %in% names(conf_matrix$byClass)) 
                conf_matrix$byClass["Sensitivity"] else NA
    specificity <- if("Specificity" %in% names(conf_matrix$byClass)) 
                    conf_matrix$byClass["Specificity"] else NA
    f1 <- if("F1" %in% names(conf_matrix$byClass)) 
            conf_matrix$byClass["F1"] else NA
    
    # Initialize advanced metrics
    auc_value <- NA
    gini <- NA
    ks_stat <- NA
    
    # Calculate AUC-ROC if probabilities are available
    if(!is.null(pred_prob)) {
      # Convert factor to numeric for prediction object (positive_class = 1, others = 0)
      actual_numeric <- ifelse(actual == positive_class, 1, 0)
      
      # Summarize probability distribution
      message("Probability summary: ", 
             paste(names(summary(pred_prob)), "=", summary(pred_prob), collapse=", "))
      
      # Create prediction object for ROCR
      pred_obj <- tryCatch({
        ROCR::prediction(pred_prob, actual_numeric)
      }, error = function(e) {
        message("ERROR creating prediction object: ", e$message)
        return(NULL)
      })
      
      if(!is.null(pred_obj)) {
        # Calculate AUC
        auc_perf <- ROCR::performance(pred_obj, "auc")
        auc_value <- auc_perf@y.values[[1]]
        
        # Gini coefficient
        gini <- 2 * auc_value - 1
        
        # KS statistic
        ks_perf <- ROCR::performance(pred_obj, "tpr", "fpr")
        ks_stat <- max(abs(ks_perf@y.values[[1]] - ks_perf@x.values[[1]]))
      }
    }
    
    # Calculate additional metrics
    true_pos <- sum(pred == positive_class & actual == positive_class)
    false_pos <- sum(pred == positive_class & actual != positive_class)
    true_neg <- sum(pred != positive_class & actual != positive_class)
    false_neg <- sum(pred != positive_class & actual == positive_class)
    
    # Calculate balanced accuracy
    balanced_accuracy <- (recall + specificity) / 2
    
    # Create results list
    results <- list(
      accuracy = accuracy,
      balanced_accuracy = balanced_accuracy,
      precision = precision,
      recall = recall,
      specificity = specificity,
      f1 = f1,
      auc = auc_value,
      gini = gini,
      ks = ks_stat,
      true_positives = true_pos,
      false_positives = false_pos,
      true_negatives = true_neg,
      false_negatives = false_neg,
      confusion_matrix = conf_matrix
    )
    
    return(results)
    
  }, error = function(e) {
    message("ERROR in evaluate_model: ", e$message)
    # Return NA for all metrics on error
    return(list(
      accuracy = NA,
      balanced_accuracy = NA,
      precision = NA,
      recall = NA,
      specificity = NA,
      f1 = NA,
      auc = NA,
      gini = NA,
      ks = NA,
      true_positives = NA,
      false_positives = NA,
      true_negatives = NA,
      false_negatives = NA,
      error = e$message
    ))
  })
}

# Function to plot ROC curve for a single model
plot_roc_curve <- function(pred_prob, actual, positive_class = "Good", 
                          model_name = "Model", color = "blue", add = FALSE) {
  tryCatch({
    # Convert actual to binary (1 for positive class, 0 for others)
    if(is.factor(actual)) {
      actual_binary <- ifelse(actual == positive_class, 1, 0)
    } else {
      actual_binary <- actual  # Assume already binary
    }
    
    # Create prediction object
    pred_obj <- ROCR::prediction(pred_prob, actual_binary)
    
    # Calculate performance metrics
    roc_perf <- ROCR::performance(pred_obj, "tpr", "fpr")
    auc_perf <- ROCR::performance(pred_obj, "auc")
    auc_value <- auc_perf@y.values[[1]]
    
    # Calculate Gini coefficient
    gini <- 2 * auc_value - 1
    
    # Calculate KS statistic
    ks_stat <- max(abs(roc_perf@y.values[[1]] - roc_perf@x.values[[1]]))
    
    # Create title with metrics
    plot_title <- paste0("ROC Curve", if(!is.null(model_name)) paste(" -", model_name) else "")
    
    # Create plot
    if(!add) {
      plot(roc_perf, 
           main = plot_title, 
           col = color, 
           lwd = 2,
           xlab = "False Positive Rate (1 - Specificity)",
           ylab = "True Positive Rate (Sensitivity)")
      abline(0, 1, lty = 2, col = "gray")
      legend_x <- 0.7
      legend_y <- 0.3
    } else {
      plot(roc_perf, add = TRUE, col = color, lwd = 2)
      legend_x <- par("usr")[2] * 0.7  # 70% of the way to the right
      legend_y <- par("usr")[3] + 0.3 * (par("usr")[4] - par("usr")[3])  # 30% up from bottom
    }
    
    # Add legend text with metrics
    legend_text <- c(
      paste(model_name, "(AUC =", round(auc_value, 3), ")"),
      paste("Gini =", round(gini, 3)),
      paste("KS =", round(ks_stat, 3))
    )
    
    # Add legend if this is a new plot (not adding to existing)
    if(!add) {
      legend("bottomright", legend = legend_text, col = color, lwd = 2, cex = 0.8)
    }
    
    # Return metrics for further use
    return(list(
      auc = auc_value,
      gini = gini,
      ks = ks_stat,
      roc_perf = roc_perf
    ))
    
  }, error = function(e) {
    message("ERROR plotting ROC curve: ", e$message)
    return(NULL)
  })
}

# Function to plot multiple ROC curves for comparison
plot_multiple_roc_curves <- function(pred_probs, actual, positive_class = "Good", 
                                    model_names = NULL, colors = NULL) {
  # Check inputs
  if(!is.list(pred_probs)) {
    stop("pred_probs must be a list of prediction probability vectors")
  }
  
  num_models <- length(pred_probs)
  
  # Set default model names if not provided
  if(is.null(model_names)) {
    model_names <- paste0("Model ", 1:num_models)
  } else if(length(model_names) != num_models) {
    warning("Length of model_names doesn't match pred_probs. Using default names.")
    model_names <- paste0("Model ", 1:num_models)
  }
  
  # Set default colors if not provided
  if(is.null(colors)) {
    colors <- rainbow(num_models)
  } else if(length(colors) != num_models) {
    warning("Length of colors doesn't match pred_probs. Using default colors.")
    colors <- rainbow(num_models)
  }
  
  # Initialize empty plot
  plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), 
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)",
       main = "ROC Curves Comparison")
  abline(0, 1, lty = 2, col = "gray")
  
  # Store results for each model
  results <- list()
  
  # Plot each model's ROC curve
  for(i in 1:num_models) {
    model_result <- plot_roc_curve(
      pred_probs[[i]], 
      actual, 
      positive_class = positive_class,
      model_name = model_names[i],
      color = colors[i],
      add = TRUE
    )
    
    results[[model_names[i]]] <- model_result
  }
  
  # Add legend
  legend_text <- sapply(1:num_models, function(i) {
    auc <- round(results[[model_names[i]]]$auc, 3)
    paste(model_names[i], "(AUC =", auc, ")")
  })
  
  legend("bottomright", legend = legend_text, col = colors, lwd = 2, cex = 0.8)
  
  # Return all results
  return(results)
}

# Function to create a performance comparison table
create_performance_table <- function(model_results, metrics = c("accuracy", "precision", "recall", "f1", "auc")) {
  # Check inputs
  if(!is.list(model_results)) {
    stop("model_results must be a list of model evaluation results")
  }
  
  # Initialize an empty data frame for the comparison
  comparison <- data.frame(
    Model = names(model_results),
    stringsAsFactors = FALSE
  )
  
  # Add each requested metric as a column
  for(metric in metrics) {
    comparison[[metric]] <- sapply(model_results, function(result) {
      if(is.null(result) || is.null(result[[metric]])) NA else result[[metric]]
    })
  }
  
  # Only keep models with at least one valid metric
  valid_rows <- apply(comparison[, metrics, drop = FALSE], 1, function(row) any(!is.na(row)))
  if(sum(valid_rows) > 0) {
    comparison <- comparison[valid_rows, ]
    
    # Find the best metric to sort by (prioritize AUC, then accuracy, etc.)
    for(sort_metric in c("auc", "f1", "accuracy", "precision", "recall")) {
      if(sort_metric %in% metrics && any(!is.na(comparison[[sort_metric]]))) {
        # Sort by this metric (descending)
        comparison <- comparison[order(comparison[[sort_metric]], decreasing = TRUE, na.last = TRUE), ]
        break
      }
    }
  } else {
    message("No valid metrics found for any models.")
  }
  
  return(comparison)
}

# Function to visualize model comparison with ggplot2
visualize_model_comparison <- function(comparison_df, 
                                     metrics = c("Accuracy", "Precision", "Recall", "F1_Score", "AUC"),
                                     title = "Model Performance Comparison") {
  
  # Check if ggplot2 is available
  if(!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package is required for this function")
  }
  
  # Check if all requested metrics are in the dataframe
  available_metrics <- intersect(metrics, colnames(comparison_df))
  
  if(length(available_metrics) == 0) {
    stop("None of the requested metrics are in the comparison dataframe")
  }
  
  if(length(available_metrics) < length(metrics)) {
    warning("Some requested metrics are not available: ", 
            paste(setdiff(metrics, available_metrics), collapse = ", "))
    metrics <- available_metrics
  }
  
  # Convert to long format for plotting
  comparison_long <- tidyr::pivot_longer(
    comparison_df, 
    cols = metrics,
    names_to = "Metric",
    values_to = "Value"
  )
  
  # Create the plot
  p <- ggplot2::ggplot(comparison_long, ggplot2::aes(x = Model, y = Value, fill = Metric)) +
    ggplot2::geom_bar(stat = "identity", position = "dodge") +
    ggplot2::labs(title = title, x = "Model", y = "Score") +
    ggplot2::theme_minimal() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
  
  return(p)
}

# Function to visualize the confusion matrix
plot_confusion_matrix <- function(conf_matrix, positive_class = "Good", 
                                title = "Confusion Matrix", 
                                color_palette = c("white", "steelblue")) {
  # Check if ggplot2 is available
  if(!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package is required for this function")
  }
  
  # Extract the confusion matrix table
  if(class(conf_matrix) == "confusionMatrix") {
    # Using caret's confusionMatrix object
    cm_table <- as.table(conf_matrix$table)
  } else {
    # Assume it's already a table
    cm_table <- as.table(conf_matrix)
  }
  
  # Convert to data frame for ggplot
  cm_df <- as.data.frame(cm_table)
  names(cm_df) <- c("Predicted", "Actual", "Frequency")
  
  # Get the max frequency for scaling
  max_freq <- max(cm_df$Frequency)
  
  # Calculate percentages
  cm_df$Percentage <- cm_df$Frequency / sum(cm_df$Frequency) * 100
  
  # Format labels for display
  cm_df$Label <- paste0(cm_df$Frequency, "\n(", round(cm_df$Percentage, 1), "%)")
  
  # Create the plot
  p <- ggplot2::ggplot(cm_df, ggplot2::aes(x = Actual, y = Predicted, fill = Frequency)) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(ggplot2::aes(label = Label), color = "black", size = 4) +
    ggplot2::scale_fill_gradient(low = color_palette[1], high = color_palette[2]) +
    ggplot2::labs(title = title, x = "Actual", y = "Predicted") +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      axis.text = ggplot2::element_text(size = 12),
      axis.title = ggplot2::element_text(size = 14),
      plot.title = ggplot2::element_text(size = 16, hjust = 0.5)
    )
  
  return(p)
}

# Function to extract variable importance from different model types
extract_variable_importance <- function(model, top_n = 10, model_type = NULL) {
  if(is.null(model_type)) {
    # Try to infer model type
    if(inherits(model, "train")) {
      # caret model
      model_type <- "caret"
    } else if(inherits(model, "randomForest")) {
      model_type <- "randomForest"
    } else if(inherits(model, "xgb.Booster")) {
      model_type <- "xgboost"
    } else if(inherits(model, "glm")) {
      model_type <- "glm"
    } else if(inherits(model, "rpart")) {
      model_type <- "rpart"
    } else {
      stop("Unable to determine model type. Please specify model_type.")
    }
  }
  
  # Extract variable importance based on model type
  tryCatch({
    if(model_type == "caret") {
      # For caret models, use varImp
      imp <- caret::varImp(model)
      if(inherits(imp, "varImp.train")) {
        imp_df <- as.data.frame(imp$importance)
        # Some caret models have columns for each class, select the appropriate one
        if(ncol(imp_df) > 1) {
          # Take the maximum across classes or Overall if it exists
          if("Overall" %in% colnames(imp_df)) {
            imp_df <- data.frame(Importance = imp_df$Overall)
          } else {
            imp_df <- data.frame(Importance = apply(imp_df, 1, max))
          }
        } else {
          imp_df <- data.frame(Importance = imp_df[,1])
        }
        imp_df$Variable <- rownames(imp_df)
      }
    } else if(model_type == "randomForest") {
      # For randomForest models
      imp <- randomForest::importance(model)
      imp_df <- data.frame(
        Variable = rownames(imp),
        Importance = imp[, 1]  # Usually MeanDecreaseGini or MeanDecreaseAccuracy
      )
    } else if(model_type == "xgboost") {
      # For XGBoost models
      imp <- xgboost::xgb.importance(model = model)
      imp_df <- data.frame(
        Variable = imp$Feature,
        Importance = imp$Gain
      )
    } else if(model_type == "glm") {
      # For glm models, use absolute coefficient values
      coefs <- coef(model)[-1]  # Remove intercept
      imp_df <- data.frame(
        Variable = names(coefs),
        Importance = abs(coefs)
      )
    } else if(model_type == "rpart") {
      # For rpart decision trees
      imp <- model$variable.importance
      imp_df <- data.frame(
        Variable = names(imp),
        Importance = imp
      )
    } else {
      stop("Unsupported model type: ", model_type)
    }
    
    # Sort by importance
    imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]
    
    # Take top N variables if requested
    if(!is.null(top_n) && top_n > 0 && nrow(imp_df) > top_n) {
      imp_df <- imp_df[1:top_n, ]
    }
    
    # Reset row names
    rownames(imp_df) <- NULL
    
    return(imp_df)
    
  }, error = function(e) {
    message("ERROR extracting variable importance: ", e$message)
    return(NULL)
  })
}

# Function to plot variable importance
plot_variable_importance <- function(var_imp_df, title = "Variable Importance", max_vars = NULL) {
  # Ensure the required columns exist
  if (!all(c("Feature", "Gain") %in% colnames(var_imp_df))) {
    stop("importance_df must have 'Feature' and 'Gain' columns")
  }
  
  # Sort by importance
  var_imp_df <- var_imp_df[order(var_imp_df$Gain, decreasing = TRUE), ]
  
  # Limit number of variables if specified
  if (!is.null(max_vars)) {
    var_imp_df <- head(var_imp_df, max_vars)
  }
  
  # Create the plot
  plot <- ggplot2::ggplot(var_imp_df, ggplot2::aes(x = reorder(Feature, Gain), y = Gain)) +
    ggplot2::geom_bar(stat = "identity", fill = "steelblue") +
    ggplot2::coord_flip() +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = title,
      x = "Features",
      y = "Importance"
    ) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5),
      axis.text.y = ggplot2::element_text(size = 8)
    )
  
  return(plot)
}

# Function to calculate lift and gain
calculate_lift_gain <- function(pred_prob, actual, positive_class = "Good", 
                              num_bins = 10) {
  # Ensure actual is in the right format
  if(is.factor(actual)) {
    actual_binary <- ifelse(actual == positive_class, 1, 0)
  } else {
    actual_binary <- actual  # Assume already binary
  }
  
  # Combine predictions and actual values
  data <- data.frame(
    pred_prob = pred_prob,
    actual = actual_binary
  )
  
  # Sort by predicted probability (descending)
  data <- data[order(data$pred_prob, decreasing = TRUE), ]
  
  # Calculate bin edges
  bin_size <- ceiling(nrow(data) / num_bins)
  
  # Initialize results
  lift_table <- data.frame(
    Bin = 1:num_bins,
    CumulativeObservations = numeric(num_bins),
    CumulativeEvents = numeric(num_bins),
    CumulativePct = numeric(num_bins),
    EventRate = numeric(num_bins),
    Lift = numeric(num_bins),
    Gain = numeric(num_bins)
  )
  
  # Base event rate
  base_rate <- mean(actual_binary)
  
  # Calculate metrics for each bin
  for(i in 1:num_bins) {
    start_idx <- (i-1) * bin_size + 1
    end_idx <- min(i * bin_size, nrow(data))
    
    # Current bin data
    bin_data <- data[start_idx:end_idx, ]
    
    # Cumulative metrics
    cum_data <- data[1:end_idx, ]
    
    # Calculate metrics
    cum_obs <- nrow(cum_data)
    cum_events <- sum(cum_data$actual)
    cum_pct <- cum_obs / nrow(data)
    event_rate <- mean(bin_data$actual)
    lift <- mean(cum_data$actual) / base_rate
    gain <- cum_events / sum(actual_binary)
    
    # Store in table
    lift_table[i, ] <- c(i, cum_obs, cum_events, cum_pct, event_rate, lift, gain)
  }
  
  return(lift_table)
}

# Function to plot lift chart
plot_lift_chart <- function(lift_table, title = "Lift Chart") {
  # Check if ggplot2 is available
  if(!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package is required for this function")
  }
  
  # Create the plot
  p <- ggplot2::ggplot(lift_table, ggplot2::aes(x = Bin)) +
    ggplot2::geom_line(ggplot2::aes(y = Lift), color = "blue", size = 1) +
    ggplot2::geom_point(ggplot2::aes(y = Lift), color = "blue", size = 3) +
    ggplot2::geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    ggplot2::labs(title = title, x = "Bin (Sorted by Predicted Probability)", y = "Lift") +
    ggplot2::theme_minimal() +
    ggplot2::scale_x_continuous(breaks = lift_table$Bin)
  
  return(p)
}

# Function to plot gain chart
plot_gain_chart <- function(lift_table, title = "Cumulative Gain Chart") {
  # Check if ggplot2 is available
  if(!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package is required for this function")
  }
  
  # Create the plot
  p <- ggplot2::ggplot(lift_table, ggplot2::aes(x = CumulativePct)) +
    ggplot2::geom_line(ggplot2::aes(y = Gain), color = "green", size = 1) +
    ggplot2::geom_point(ggplot2::aes(y = Gain), color = "green", size = 3) +
    ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
    ggplot2::labs(title = title, x = "Cumulative Population (%)", y = "Cumulative Gain (%)") +
    ggplot2::theme_minimal() +
    ggplot2::scale_x_continuous(labels = scales::percent) +
    ggplot2::scale_y_continuous(labels = scales::percent)
  
  return(p)
}

# If this script is run directly, provide information about its purpose
if(!exists("EVALUATION_SOURCED") || !EVALUATION_SOURCED) {
  message("This script contains evaluation functions for the German Credit Analysis project.")
  message("Source this script in your main analysis code to use these functions.")
  EVALUATION_SOURCED <- TRUE
}