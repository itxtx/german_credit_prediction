# ======= 05_analysis.R =======
# This script provides a modular workflow for the German Credit Risk Analysis

# Set working directory to project root
if (!require(here)) {
  install.packages("here")
  library(here)
}
project_root <- here()
setwd(project_root)

# Record start time
start_time <- Sys.time()

# Create log file with timestamp
log_file <- "/app/logs/analysis_log_20250410_154452.txt"  # Absolute path within container
#dir.create("../logs", showWarnings = FALSE, recursive = TRUE)

# Start logging
sink(log_file, append = TRUE, split = TRUE)  # split=TRUE keeps output in console as well
cat("Logging started at:", format(start_time, "%Y-%m-%d %H:%M:%S"), "\n\n")

# Create a function to run a script and measure its execution time
run_script <- function(script_path, script_name) {
  cat("\n")
  cat("------------------------------------------------------------------\n")
  cat("EXECUTING:", script_name, "\n")
  cat("------------------------------------------------------------------\n")
  
  script_start_time <- Sys.time()
  
  result <- tryCatch({
    source(script_path)
    "SUCCESS"
  }, error = function(e) {
    cat("ERROR in", script_name, ":", e$message, "\n")
    "FAILED"
  }, warning = function(w) {
    cat("WARNING in", script_name, ":", w$message, "\n")
    "COMPLETED WITH WARNINGS"
  })
  
  script_duration <- Sys.time() - script_start_time
  
  cat("\n")
  cat(script_name, "execution:", result, "\n")
  cat("Duration:", format(script_duration, digits = 2), "\n")
  
  return(list(
    script = script_name,
    result = result,
    duration = script_duration
  ))
}

# Function to run a specific phase of the analysis
run_phase <- function(phase_name, scripts) {
  cat("\n")
  cat("==================================================================\n")
  cat("                 PHASE:", phase_name, "\n")
  cat("==================================================================\n")
  
  results <- list()
  for (script_name in names(scripts)) {
    results[[script_name]] <- run_script(scripts[[script_name]], basename(scripts[[script_name]]))
    
    # If a script fails, stop the phase
    if (results[[script_name]]$result == "FAILED") {
      cat("\nPhase", phase_name, "failed at", script_name, "\n")
      return(results)
    }
  }
  
  return(results)
}

# Define phases and their scripts
phases <- list(
  setup = list(
    setup = "scripts/utils/setup.R",
    preprocessing = "scripts/utils/preprocessing.R",
    evaluation = "scripts/utils/evaluation.R"
  ),
  data_preparation = list(
    import = "scripts/01_data_import.R",
    preprocessing = "scripts/02_data_preprocessing.R"
  ),
  modeling = list(
    logistic_regression = "scripts/03_models/logistic_regression.R",
    naive_bayes = "scripts/03_models/naive_bayes.R",
    decision_tree = "scripts/03_models/decision_tree.R",
    random_forest = "scripts/03_models/random_forest.R",
    xgboost = "scripts/03_models/xgboost.R",
    svm = "scripts/03_models/svm.R"
  ),
  evaluation = list(
    comparison = "scripts/04_model_comparison.R"
  )
)

# Function to run the entire workflow or specific phases
run_workflow <- function(selected_phases = names(phases)) {
  execution_log <- list()
  
  for (phase_name in selected_phases) {
    if (phase_name %in% names(phases)) {
      cat("\nRunning phase:", phase_name, "\n")
      execution_log[[phase_name]] <- run_phase(phase_name, phases[[phase_name]])
      
      # Check if phase completed successfully
      phase_results <- sapply(execution_log[[phase_name]], function(x) x$result)
      if (any(phase_results == "FAILED")) {
        cat("\nWorkflow stopped due to failure in phase:", phase_name, "\n")
        break
      }
    } else {
      cat("\nWarning: Phase", phase_name, "not found\n")
    }
  }
  
  return(execution_log)
}

# Function to generate summary report
generate_summary <- function(execution_log) {
  # Implementation of summary generation
  # (Keep existing summary generation code)
}

# Main execution
if (!exists("SKIP_EXECUTION")) {
  # Run all phases by default
  cat("\nStarting German Credit Risk Analysis\n")
  execution_log <- run_workflow()
  
  # Generate summary
  generate_summary(execution_log)
  
  # Print completion message
  end_time <- Sys.time()
  total_duration <- end_time - start_time
  
  cat("\n")
  cat("==================================================================\n")
  cat("                   ANALYSIS COMPLETED                             \n")
  cat("==================================================================\n")
  cat("End time:", format(end_time, "%Y-%m-%d %H:%M:%S"), "\n")
  cat("Total duration:", format(total_duration, digits = 2), "\n\n")
  
  # Stop logging
  sink()
  cat("Analysis log saved to:", log_file, "\n")
} else {
  cat("\nScript loaded without execution. Use run_workflow() to run analysis.\n")
  cat("Example usage:\n")
  cat("- run_workflow()  # Run all phases\n")
  cat("- run_workflow(c('setup', 'data_preparation'))  # Run specific phases\n")
  
  # Stop logging
  sink()
}