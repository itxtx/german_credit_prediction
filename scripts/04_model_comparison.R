# ======= 04_model_comparison.R =======
# This script compares the performance of all trained models.
# The script handles:
# - Collecting and comparing performance metrics
# - Generating comparative visualizations
# - ROC curve comparisons
# - Determining the best model

# Source utility scripts
source("scripts/utils/setup.R")
source("scripts/utils/evaluation.R")

# Define the list of models to compare
model_list <- c(
  "logistic_regression",
  "naive_bayes",
  "decision_tree",
  "random_forest", 
  "xgboost",
  "svm"
)

# Function to load model results
load_model_results <- function(model_names = model_list) {
  message("\n=== Loading Model Results ===")
  
  # Initialize list to store results
  all_results <- list()
  
  # Load results for each model
  for(model_name in model_names) {
    model_dir <- file.path("results/models", model_name)
    
    # Adjust model filename based on model type
    model_file <- if(model_name == "logistic_regression") {
      file.path(model_dir, "logistic_model.rds")
    } else {
      file.path(model_dir, paste0(model_name, "_model.rds"))
    }
    perf_file <- file.path(model_dir, "performance_metrics.RData")
    
    # Add debug messages
    message("Looking for model file: ", model_file)
    message("Looking for performance file: ", perf_file)
    
    # Check if files exist
    if(file.exists(model_file) && file.exists(perf_file)) {
      # Load model
      model <- tryCatch({
        readRDS(model_file)
      }, error = function(e) {
        message("WARNING: Could not load model for ", model_name, ": ", e$message)
        return(NULL)
      })
      
      # Load performance metrics
      perf <- tryCatch({
        perf_env <- new.env()
        load(perf_file, envir = perf_env)
        if("performance" %in% ls(perf_env)) {
          perf_env$performance
        } else {
          # Try to get the first object in the environment
          perf_env[[ls(perf_env)[1]]]
        }
      }, error = function(e) {
        message("WARNING: Could not load performance metrics for ", model_name, ": ", e$message)
        return(NULL)
      })
      
      # Store results
      all_results[[model_name]] <- list(
        model = model,
        performance = perf
      )
      
      message("Loaded results for ", model_name)
    } else {
      message("WARNING: Results not found for ", model_name)
    }
  }
  
  return(all_results)
}

# Function to create a comparison table of performance metrics
create_comparison_table <- function(all_results, metrics = c("accuracy", "precision", "recall", "f1", "auc")) {
  message("\n=== Creating Performance Comparison Table ===")
  
  # Initialize data frame
  comparison <- data.frame(
    Model = character(),
    stringsAsFactors = FALSE
  )
  
  # Add metrics columns
  for(metric in metrics) {
    comparison[[metric]] <- numeric()
  }
  
  # Fill in metrics for each model
  for(model_name in names(all_results)) {
    # Get performance metrics
    perf <- all_results[[model_name]]$performance
    
    # Skip if performance metrics not available
    if(is.null(perf)) {
      message("WARNING: No performance metrics for ", model_name, ". Skipping...")
      next
    }
    
    # Create row for this model
    row <- data.frame(
      Model = model_name,
      stringsAsFactors = FALSE
    )
    
    # Add metrics
    for(metric in metrics) {
      if(!is.null(perf[[metric]])) {
        row[[metric]] <- perf[[metric]]
      } else {
        row[[metric]] <- NA
      }
    }
    
    # Add to comparison table
    comparison <- rbind(comparison, row)
  }
  
  # Identify best model for each metric
  for(metric in metrics) {
    # Skip if all values are NA
    if(all(is.na(comparison[[metric]]))) {
      next
    }
    
    # Find the best model (highest value)
    best_idx <- which.max(comparison[[metric]])
    
    # Mark the best model with an asterisk
    comparison[[paste0(metric, "_best")]] <- ""
    comparison[[paste0(metric, "_best")]][best_idx] <- "*"
  }
  
  # Sort by AUC or accuracy if AUC not available
  if("auc" %in% metrics && !all(is.na(comparison$auc))) {
    comparison <- comparison[order(comparison$auc, decreasing = TRUE), ]
  } else if("accuracy" %in% metrics && !all(is.na(comparison$accuracy))) {
    comparison <- comparison[order(comparison$accuracy, decreasing = TRUE), ]
  }
  
  # Format metrics to 4 decimal places
  for(metric in metrics) {
    if(metric %in% names(comparison)) {
      comparison[[metric]] <- round(comparison[[metric]], 4)
    }
  }
  
  return(comparison)
}

# Function to visualize model comparison
visualize_model_comparison <- function(comparison_table, output_dir = "results/model_comparison") {
  message("\n=== Creating Comparison Visualizations ===")
  
  # Create output directory if it doesn't exist
  if(!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    message("Created directory: ", output_dir)
  }
  
  # Install and load required packages if not available
  required_packages <- c("ggplot2", "tidyr", "fmsb")
  for(pkg in required_packages) {
    if(!requireNamespace(pkg, quietly = TRUE)) {
      message("Installing package: ", pkg)
      install.packages(pkg, repos = "https://cloud.r-project.org")
    }
    library(pkg, character.only = TRUE)
  }
  
  # Get metrics columns (exclude _best columns)
  metrics <- setdiff(names(comparison_table), c("Model", grep("_best$", names(comparison_table), value = TRUE)))
  
  # 1. Create bar chart for each metric
  for(metric in metrics) {
    # Skip if all values are NA
    if(all(is.na(comparison_table[[metric]]))) {
      message("WARNING: All values are NA for metric ", metric, ". Skipping...")
      next
    }
    
    # Create data for this metric
    metric_data <- comparison_table[, c("Model", metric)]
    metric_data <- metric_data[!is.na(metric_data[[metric]]), ]
    
    # Sort by metric value
    metric_data <- metric_data[order(metric_data[[metric]], decreasing = TRUE), ]
    
    # Convert Model to factor with levels in order
    metric_data$Model <- factor(metric_data$Model, levels = metric_data$Model)
    
    # Create plot
    plot <- ggplot2::ggplot(metric_data, ggplot2::aes(x = Model, y = .data[[metric]])) +
      ggplot2::geom_bar(stat = "identity", fill = "steelblue") +
      ggplot2::geom_text(ggplot2::aes(label = .data[[metric]]), vjust = -0.5, size = 3) +
      ggplot2::labs(
        title = paste0("Model Comparison by ", toupper(substr(metric, 1, 1)), substr(metric, 2, nchar(metric))),
        x = "Model",
        y = toupper(substr(metric, 1, 1)), substr(metric, 2, nchar(metric))
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
        plot.title = ggplot2::element_text(hjust = 0.5)
      )
    
    # Save plot
    plot_file <- file.path(output_dir, paste0(metric, "_comparison.png"))
    ggplot2::ggsave(plot_file, plot, width = 10, height = 6)
    message("Saved ", metric, " comparison plot to ", plot_file)
  }
  
  # 2. Create summary bar chart with all metrics
  # Reshape data for plotting
  if(requireNamespace("tidyr", quietly = TRUE)) {
    # Use tidyr to reshape data with updated syntax
    long_data <- tidyr::pivot_longer(
      comparison_table,
      cols = all_of(metrics),  # Use all_of() for external vectors
      names_to = "Metric",
      values_to = "Value"
    )
    
    # Create plot with explicit column names
    plot <- ggplot2::ggplot(long_data, ggplot2::aes(x = .data[["Model"]], 
                                                    y = .data[["Value"]], 
                                                    fill = .data[["Metric"]])) +
      ggplot2::geom_bar(stat = "identity", position = "dodge") +
      ggplot2::labs(
        title = "Model Comparison across All Metrics",
        x = "Model",
        y = "Value"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
        plot.title = ggplot2::element_text(hjust = 0.5)
      )
    
    # Save plot
    plot_file <- file.path(output_dir, "all_metrics_comparison.png")
    ggplot2::ggsave(plot_file, plot, width = 12, height = 8)
    message("Saved all metrics comparison plot to ", plot_file)
  } else {
    message("tidyr package not available. Skipping all metrics comparison plot...")
  }
  
  # 3. Create radar chart if fmsb package is available
  if(requireNamespace("fmsb", quietly = TRUE)) {
    # Get metrics columns (exclude Model and _best columns)
    metrics <- setdiff(names(comparison_table), c("Model", grep("_best$", names(comparison_table), value = TRUE)))
    
    # Prepare data for radar chart - transpose so metrics are axes
    radar_data <- as.data.frame(t(comparison_table[, metrics]))
    colnames(radar_data) <- comparison_table$Model
    
    # Ensure row names are set to metrics
    rownames(radar_data) <- metrics
    
    # Add max and min rows required by fmsb
    radar_data <- rbind(
      rep(1, ncol(radar_data)),  # Max value for all metrics is 1
      rep(0, ncol(radar_data)),  # Min value for all metrics is 0
      radar_data
    )
    
    # Replace NA with min value
    radar_data[is.na(radar_data)] <- 0
    
    # Create radar chart
    png(file.path(output_dir, "radar_chart.png"), width = 800, height = 800)
    par(mar = c(1, 1, 2, 1))  # Adjust margins
    
    fmsb::radarchart(
      radar_data,
      pcol = rainbow(ncol(radar_data)),
      pfcol = rainbow(ncol(radar_data), alpha = 0.3),
      plwd = 2,
      cglcol = "grey",
      cglty = 1,
      axislabcol = "grey",
      caxislabels = seq(0, 1, 0.2),
      cglwd = 0.8,
      title = "Model Performance Comparison",
      axistype = 1,  # Show axis labels
      seg = 5  # Number of segments
    )
    
    # Add legend
    legend(
      "topright",
      legend = colnames(radar_data),
      col = rainbow(ncol(radar_data)),
      lty = 1,
      lwd = 2,
      pch = 20,
      bty = "n"
    )
    dev.off()
    message("Saved radar chart to ", file.path(output_dir, "radar_chart.png"))
  } else {
    message("fmsb package not available. Skipping radar chart...")
  }
  
  return(TRUE)
}

# Function to compare ROC curves
compare_roc_curves <- function(all_results, test_data, output_dir = "results/model_comparison") {
  message("\n=== Comparing ROC Curves ===")
  
  # Create output directory if it doesn't exist
  if(!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    message("Created directory: ", output_dir)
  }
  
  # Verify pROC package is available
  if(!requireNamespace("pROC", quietly = TRUE)) {
    message("Installing pROC package...")
    install.packages("pROC", repos = "https://cloud.r-project.org")
  }
  
  # Ensure test data has been loaded
  if(!exists("test_data")) {
    message("WARNING: test_data not found. Loading from processed data...")
    test_data <- tryCatch({
      read.csv("data/processed/test_data.csv", stringsAsFactors = TRUE)
    }, error = function(e) {
      message("ERROR: Could not load test_data: ", e$message)
      return(NULL)
    })
    
    if(is.null(test_data)) {
      message("ERROR: Could not load test data. Cannot compare ROC curves.")
      return(NULL)
    }
  }
  
  # Extract actual class values
  actual <- test_data$class
  
  # Initialize list to store prediction probabilities
  pred_probs <- list()
  
  # Load models and generate predictions
  for(model_name in names(all_results)) {
    # Get model
    model_result <- all_results[[model_name]]
    
    # Skip if model not available
    if(is.null(model_result$model)) {
      message("WARNING: Model not available for ", model_name, ". Skipping...")
      next
    }
    
    # Check if we already have performance data with predictions
    if(!is.null(model_result$performance) && !is.null(model_result$performance$pred_prob)) {
      pred_probs[[model_name]] <- model_result$performance$pred_prob
      message("Using existing predictions for ", model_name)
      next
    }
    
    # Otherwise, load model and generate predictions
    model <- model_result$model
    
    # Load the corresponding script to get the prediction function
    script_path <- file.path("scripts/03_models", paste0(model_name, ".R"))
    
    if(file.exists(script_path)) {
      # Source the script to get the generate_predictions function
      source(script_path)
      
      # Try to generate predictions
      tryCatch({
        # Different models might need different inputs for prediction
        if(model_name == "xgboost") {
          # For XGBoost, we need to prepare the test data differently
          source("scripts/03_models/xgboost.R")
          prepared_data <- prepare_for_xgboost(train_data, test_data)
          predictions <- generate_predictions(model, prepared_data$test_matrix)
        } else if(model_name == "random_forest") {
          # Add specific handling for random forest
          # First check if all required columns are present
          required_cols <- names(model$forest$xlevels)  # Get required feature names
          missing_cols <- setdiff(required_cols, names(test_data))
          
          if(length(missing_cols) > 0) {
            message("WARNING: Missing columns for random_forest: ", paste(missing_cols, collapse=", "))
            return(NULL)
          }
          predictions <- generate_predictions(model, test_data)
        } else {
          # For other models, we can use the test data directly
          predictions <- generate_predictions(model, test_data)
        }
        
        # Store prediction probabilities
        pred_probs[[model_name]] <- predictions$prob
        message("Generated predictions for ", model_name)
      }, error = function(e) {
        message("ERROR generating predictions for ", model_name, ": ", e$message)
        # Print additional debug information
        message("Available columns in test_data: ", paste(names(test_data), collapse=", "))
      })
    } else {
      message("WARNING: Script not found for ", model_name, ". Skipping...")
    }
  }
  
  # Compare ROC curves if we have at least two models with predictions
  if(length(pred_probs) >= 2) {
    message("Comparing ROC curves for ", length(pred_probs), " models")
    
    # Define the output file path
    output_file <- file.path(output_dir, "combined_roc_curves.png")
    message("Will save ROC curves to: ", output_file)
    
    tryCatch({
      # Create combined ROC plot with enhanced styling
      png(output_file, width = 1200, height = 1000, res = 120)
      on.exit(dev.off(), add = TRUE)  # Ensure device is closed even if error occurs
      
      # Set up the plotting area with margins
      par(mar = c(5, 5, 4, 8), xaxs = "i", yaxs = "i")
      
      # Initialize plot with first model
      model_name <- names(pred_probs)[1]
      roc_obj <- pROC::roc(actual, pred_probs[[model_name]])
      
      # Create base plot
      plot(roc_obj, 
           main = "Comparison of ROC Curves",
           col = rainbow(length(pred_probs))[1], 
           lwd = 3,
           cex.main = 1.5,
           cex.lab = 1.2,
           cex.axis = 1.1)
      
      # Add grid
      grid(nx = 10, ny = 10, col = "lightgray", lty = "dotted")
      
      # Add other ROC curves
      results <- list()
      results[[model_name]] <- list(roc = roc_obj, auc = pROC::auc(roc_obj))
      
      for(i in 2:length(pred_probs)) {
        model_name <- names(pred_probs)[i]
        roc_obj <- pROC::roc(actual, pred_probs[[model_name]])
        lines(roc_obj, col = rainbow(length(pred_probs))[i], lwd = 3)
        results[[model_name]] <- list(roc = roc_obj, auc = pROC::auc(roc_obj))
      }
      
      # Add diagonal reference line
      abline(a = 0, b = 1, lty = 2, col = "gray50", lwd = 2)
      
      # Add legend with AUC values (sorted by AUC)
      auc_values <- sapply(results, function(x) x$auc)
      sorted_idx <- order(auc_values, decreasing = TRUE)
      legend_text <- paste0(
        gsub("_", " ", names(pred_probs)[sorted_idx]), 
        " (AUC = ", 
        sprintf("%.3f", auc_values[sorted_idx]), 
        ")"
      )
      
      # Position legend outside the plot
      legend("bottomright",
             legend = legend_text,
             col = rainbow(length(pred_probs))[sorted_idx],
             lwd = 3,
             bty = "n",
             cex = 0.9)
      
      # Add box around plot
      box(lwd = 2)
      
      message("Successfully created ROC plot")
      
      # Return results
      return(list(
        roc_results = results,
        auc_values = auc_values,
        output_file = output_file
      ))
      
    }, error = function(e) {
      message("ERROR creating ROC plot: ", e$message)
      return(NULL)
    })
  }
  
  # Return NULL if we don't have enough data
  return(NULL)
}

# Function to determine the best model
determine_best_model <- function(comparison_table, primary_metric = "auc", secondary_metric = "f1") {
  message("\n=== Determining Best Model ===")
  
  # Check if primary metric is available
  if(!(primary_metric %in% names(comparison_table))) {
    message("WARNING: Primary metric ", primary_metric, " not found in comparison table")
    
    # Try to use secondary metric
    if(secondary_metric %in% names(comparison_table)) {
      message("Using secondary metric ", secondary_metric, " instead")
      primary_metric <- secondary_metric
    } else {
      # Use first available metric
      available_metrics <- setdiff(names(comparison_table), c("Model", grep("_best$", names(comparison_table), value = TRUE)))
      if(length(available_metrics) > 0) {
        primary_metric <- available_metrics[1]
        message("Using first available metric ", primary_metric)
      } else {
        message("ERROR: No metrics available to determine best model")
        return(NULL)
      }
    }
  }
  
  # Sort by primary metric
  sorted_table <- comparison_table[order(comparison_table[[primary_metric]], decreasing = TRUE), ]
  
  # Get best model
  best_model <- sorted_table$Model[1]
  best_value <- sorted_table[[primary_metric]][1]
  
  message("Best model based on ", primary_metric, " (", round(best_value, 4), "): ", best_model)
  
  # Create ranking for all models
  ranking <- data.frame(
    Rank = 1:nrow(sorted_table),
    Model = sorted_table$Model,
    Value = sorted_table[[primary_metric]],
    Metric = primary_metric,
    stringsAsFactors = FALSE
  )
  
  return(list(
    best_model = best_model,
    best_value = best_value,
    metric = primary_metric,
    ranking = ranking
  ))
}

# Function to save the best model
save_best_model <- function(best_model_name, all_results, output_dir = "results/model_comparison") {
  message("\n=== Saving Best Model ===")
  
  # Check if best model is available
  if(!(best_model_name %in% names(all_results))) {
    message("ERROR: Best model ", best_model_name, " not found in results")
    return(FALSE)
  }
  
  # Get best model
  best_model <- all_results[[best_model_name]]$model
  
  # Check if model is available
  if(is.null(best_model)) {
    message("ERROR: Model object not available for ", best_model_name)
    return(FALSE)
  }
  
  # Create output directory if it doesn't exist
  if(!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    message("Created directory: ", output_dir)
  }
  
  # Save model
  model_file <- file.path(output_dir, "best_model.rds")
  saveRDS(best_model, model_file)
  message("Saved best model to ", model_file)
  
  # Save model name
  name_file <- file.path(output_dir, "best_model_name.txt")
  writeLines(best_model_name, name_file)
  message("Saved best model name to ", name_file)
  
  return(TRUE)
}

# Function to create a detailed report
create_comparison_report <- function(comparison_table, best_model_info, roc_results = NULL, output_dir = "results/model_comparison") {
  message("\n=== Creating Comparison Report ===")
  
  # Create output directory if it doesn't exist
  if(!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    message("Created directory: ", output_dir)
  }
  
  # Create report file
  report_file <- file.path(output_dir, "model_comparison_report.md")
  
  # Open file for writing
  sink(report_file)
  
  # Write report header
  cat("# German Credit Risk Model Comparison Report\n\n")
  cat("Date: ", format(Sys.Date(), "%B %d, %Y"), "\n\n")
  
  # Overview section
  cat("## 1. Overview\n\n")
  cat("This report compares the performance of different machine learning models trained on the German Credit Risk dataset. ")
  cat("The models are evaluated on various metrics to determine which one performs best for predicting credit risk.\n\n")
  
  # Models section
  cat("## 2. Models Compared\n\n")
  cat("The following models were trained and evaluated:\n\n")
  for(model_name in unique(comparison_table$Model)) {
    cat("- **", gsub("_", " ", model_name), "**\n")
  }
  cat("\n")
  
  # Performance metrics section
  cat("## 3. Performance Metrics\n\n")
  cat("### 3.1 Comparison Table\n\n")
  
  # Create markdown table
  cat("| Model | ", paste(setdiff(names(comparison_table), c("Model", grep("_best$", names(comparison_table), value = TRUE))), collapse = " | "), " |\n")
  cat("|", paste(rep("---", 1 + length(setdiff(names(comparison_table), c("Model", grep("_best$", names(comparison_table), value = TRUE))))), collapse = "|"), "|\n")
  
  for(i in 1:nrow(comparison_table)) {
    row <- comparison_table[i, ]
    metrics <- setdiff(names(row), c("Model", grep("_best$", names(row), value = TRUE)))
    metric_values <- sapply(metrics, function(m) {
      val <- row[[m]]
      best <- row[[paste0(m, "_best")]]
      if(!is.na(val)) {
        if(best == "*") {
          paste0("**", round(val, 4), "**")
        } else {
          as.character(round(val, 4))
        }
      } else {
        "N/A"
      }
    })
    cat("| ", row$Model, " | ", paste(metric_values, collapse = " | "), " |\n")
  }
  
  cat("\n")
  
  # Best model section
  cat("## 4. Best Model\n\n")
  cat("Based on the ", best_model_info$metric, " metric, the best performing model is **", 
      gsub("_", " ", best_model_info$best_model), "** with a value of **", 
      round(best_model_info$best_value, 4), "**.\n\n")
  
  # Add model ranking
  cat("### 4.1 Model Ranking\n\n")
  cat("| Rank | Model | ", best_model_info$metric, " |\n")
  cat("|---|---|---|\n")
  for(i in 1:nrow(best_model_info$ranking)) {
    rank <- best_model_info$ranking[i, ]
    cat("| ", rank$Rank, " | ", gsub("_", " ", rank$Model), " | ", round(rank$Value, 4), " |\n")
  }
  cat("\n")
  
  # ROC curve comparison section
  if(!is.null(roc_results) && !is.null(roc_results$auc_values)) {
    cat("## 5. ROC Curve Comparison\n\n")
    cat("### 5.1 AUC Values\n\n")
    cat("| Model | AUC |\n")
    cat("|---|---|\n")
    
    # Sort AUC values
    sorted_auc <- sort(roc_results$auc_values, decreasing = TRUE)
    for(i in 1:length(sorted_auc)) {
      model_name <- names(sorted_auc)[i]
      auc_value <- sorted_auc[i]
      cat("| ", gsub("_", " ", model_name), " | ", round(auc_value, 4), " |\n")
    }
    cat("\n")
    
    cat("### 5.2 ROC Curves\n\n")
    cat("![ROC Curve Comparison](", roc_results$output_file, ")\n\n")
  }
  
  # Visualization section
  cat("## 6. Performance Visualizations\n\n")
  
  # Get metrics
  metrics <- setdiff(names(comparison_table), c("Model", grep("_best$", names(comparison_table), value = TRUE)))
  
  for(metric in metrics) {
    cat("### 6.", which(metrics == metric), " ", toupper(substr(metric, 1, 1)), substr(metric, 2, nchar(metric)), " Comparison\n\n")
    cat("![", toupper(substr(metric, 1, 1)), substr(metric, 2, nchar(metric)), " Comparison](", metric, "_comparison.png)\n\n")
  }
  
  cat("### 6.", length(metrics) + 1, " All Metrics Comparison\n\n")
  cat("![All Metrics Comparison](all_metrics_comparison.png)\n\n")
  
  if(file.exists(file.path(output_dir, "radar_chart.png"))) {
    cat("### 6.", length(metrics) + 2, " Radar Chart\n\n")
    cat("![Radar Chart](radar_chart.png)\n\n")
  }
  
  # Conclusion section
  cat("## 7. Conclusion\n\n")
  cat("After comparing various machine learning models on the German Credit Risk dataset, ")
  cat("the **", gsub("_", " ", best_model_info$best_model), "** model demonstrated the best performance ")
  cat("with a ", best_model_info$metric, " of ", round(best_model_info$best_value, 4), ". ")
  
  # Add some insights based on the comparison
  if(best_model_info$best_model %in% c("random_forest", "xgboost")) {
    cat("This suggests that ensemble methods perform well on this dataset, ")
    cat("likely due to their ability to handle complex relationships and interactions between features.\n\n")
  } else if(best_model_info$best_model == "logistic_regression") {
    cat("This suggests that a simpler linear model works well for this dataset, ")
    cat("indicating that the relationship between features and credit risk might be relatively linear.\n\n")
  } else if(best_model_info$best_model == "svm") {
    cat("This suggests that the margin-based approach of SVM works well for this dataset, ")
    cat("possibly due to its ability to find a good decision boundary in the feature space.\n\n")
  } else {
    cat("Further analysis would be beneficial to understand why this model performs best on this specific dataset.\n\n")
  }
  
  # Final recommendation
  cat("For credit risk prediction tasks on similar data, the ", gsub("_", " ", best_model_info$best_model), " ")
  cat("model is recommended based on its superior performance.\n\n")
  
  # Close file
  sink()
  
  message("Saved comparison report to ", report_file)
  
  return(report_file)
}

# Add this function at the beginning of the script
validate_test_data <- function(test_data, model_list) {
  message("\n=== Validating Test Data ===")
  
  # Check if test data exists
  if(is.null(test_data)) {
    message("ERROR: Test data is NULL")
    return(FALSE)
  }
  
  # Check for required columns
  required_cols <- c("class")  # Add base required columns
  missing_cols <- setdiff(required_cols, names(test_data))
  
  if(length(missing_cols) > 0) {
    message("ERROR: Missing required columns: ", paste(missing_cols, collapse=", "))
    return(FALSE)
  }
  
  # Print available columns for debugging
  message("Available columns in test data: ", paste(names(test_data), collapse=", "))
  
  return(TRUE)
}

# Main function to run the entire model comparison workflow
run_model_comparison <- function(model_names = model_list, primary_metric = "auc", secondary_metric = "f1") {
  message("\n====== Running Model Comparison Workflow ======\n")
  
  # Validate test data first
  if(!validate_test_data(test_data, model_names)) {
    message("ERROR: Test data validation failed")
    return(NULL)
  }
  
  # Step 1: Load model results
  all_results <- load_model_results(model_names)
  
  # Step 2: Create comparison table
  comparison_table <- create_comparison_table(all_results)
  
  # Print comparison table
  message("\nModel Performance Comparison:")
  print(comparison_table)
  
  # Step 3: Visualize model comparison
  visualize_model_comparison(comparison_table)
  
  # Step 4: Compare ROC curves
  roc_results <- compare_roc_curves(all_results, test_data)
  
  # Step 5: Determine best model
  best_model_info <- determine_best_model(comparison_table, primary_metric, secondary_metric)
  
  # Step 6: Save best model
  if(!is.null(best_model_info)) {
    save_best_model(best_model_info$best_model, all_results)
  }
  
  # Step 7: Create comparison report
  report_file <- create_comparison_report(comparison_table, best_model_info, roc_results)
  
  message("\n====== Model Comparison Workflow Complete ======\n")
  
  # Return comparison results
  return(list(
    comparison_table = comparison_table,
    best_model_info = best_model_info,
    roc_results = roc_results,
    report_file = report_file
  ))
}

# Run the comparison if this script is being run directly
if(!exists("MODEL_COMPARISON_SOURCED") || !MODEL_COMPARISON_SOURCED) {
  # Load test data if not already in environment
  if(!exists("test_data")) {
    message("Loading test data...")
    test_data <- read.csv("data/processed/test_data.csv", stringsAsFactors = TRUE)
  }
  
  # Run model comparison
  comparison_results <- run_model_comparison()
  
  MODEL_COMPARISON_SOURCED <- TRUE
} else {
  message("04_model_comparison.R has been sourced. Use run_model_comparison() to compare models.")
}

# Update the comparison_metrics function
comparison_metrics <- function(data, metrics) {
  if(requireNamespace("dplyr", quietly = TRUE)) {
    data %>% 
      dplyr::select(dplyr::all_of(metrics))
  } else {
    # Fallback to base R if dplyr is not available
    data[, metrics, drop = FALSE]
  }
}

# Update the compare_models function
compare_models <- function(results_list, metrics) {
  # Use base R operations instead of tidyselect
  comparison_table <- do.call(rbind, results_list)
  comparison_table$Model <- rownames(comparison_table)
  rownames(comparison_table) <- NULL
  
  # Select columns and sort
  result <- comparison_table[, c("Model", metrics)]
  result <- result[order(result$accuracy, decreasing = TRUE), ]
  
  return(result)
}

# Update evaluate_performance function
evaluate_performance <- function(predictions, actual_values, metrics) {
  results <- data.frame(
    actual = actual_values,
    predicted = predictions
  )
  
  # Add metrics columns if needed
  for(metric in metrics) {
    if(!(metric %in% names(results))) {
      results[[metric]] <- NA
    }
  }
  
  return(results)
}