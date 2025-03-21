# ======= setup.R =======
# This script contains functions for setting up the environment, 
# package installation and loading, debugging functions, and 
# XGBoost installation troubleshooting.

# Turn on debugging mode to help identify issues
DEBUG_MODE <- TRUE

# Debug printing function
debug_print <- function(...) {
  if(DEBUG_MODE) {
    cat("[DEBUG] ", ..., "\n")
  }
}

# Print script information and R version
print_environment_info <- function() {
  debug_print("Starting German Credit Analysis script")
  debug_print("R version:", R.version.string)
  debug_print("Working directory:", getwd())
  
  # Print all loaded packages to help with debugging
  if(DEBUG_MODE) {
    cat("Loaded packages:\n")
    print((.packages()))
  }
}

# Function to install and load required packages
setup_packages <- function() {
  # Define required packages
  required_packages <- c(
    "readr", "dplyr", "tidyr", "ggplot2", "caret", "rpart", "rpart.plot", 
    "randomForest", "neuralnet", "ROCR", "pROC", "e1071", 
    "kernlab", "MASS", "recipes"
  )
  
  # Check which packages need to be installed
  new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
  
  # Install missing packages with a specified mirror
  if(length(new_packages) > 0) {
    cat("Installing missing packages:", paste(new_packages, collapse=", "), "\n")
    tryCatch({
      install.packages(new_packages, repos="https://cloud.r-project.org")
    }, error = function(e) {
      cat("ERROR: Failed to install packages automatically.\n")
      cat("Please install the following packages manually:\n")
      cat(paste(new_packages, collapse=", "), "\n")
      cat("You can install them using this command in R console:\n")
      cat('install.packages(c("', paste(new_packages, collapse='", "'), '"), repos="https://cloud.r-project.org")', "\n")
    })
  }
  
  # Load required libraries with error handling
  for(lib in required_packages) {
    tryCatch({
      library(lib, character.only = TRUE)
      debug_print("Loaded package:", lib)
    }, error = function(e) {
      cat("WARNING: Could not load package", lib, ":", e$message, "\n")
      cat("Some functionality may be limited.\n")
    })
  }
  
  return(required_packages)
}

# Add a safety function to check for all old boost models and related objects
check_for_old_models <- function() {
  debug_print("Checking for old model references...")
  # List of patterns to check for
  patterns <- c("adaboost", "gbm", "^ada", "boost")
  found_vars <- c()
  
  # Get all variables in the global environment
  all_vars <- ls(envir = .GlobalEnv)
  
  # Check for each pattern
  for(pattern in patterns) {
    matches <- grep(pattern, all_vars, value = TRUE)
    if(length(matches) > 0) {
      found_vars <- c(found_vars, matches)
    }
  }
  
  # If we found any matches, warn and remove them
  if(length(found_vars) > 0) {
    cat("WARNING: Found", length(found_vars), "variables that might conflict with XGBoost:\n")
    print(found_vars)
    cat("Removing these variables to prevent conflicts...\n")
    rm(list = found_vars, envir = .GlobalEnv)
    
    # Double check that they're gone
    remaining <- ls(envir = .GlobalEnv)[ls(envir = .GlobalEnv) %in% found_vars]
    if(length(remaining) > 0) {
      cat("ERROR: Failed to remove some variables:", paste(remaining, collapse=", "), "\n")
      return(FALSE)
    } else {
      cat("Successfully removed all conflicting variables.\n")
      return(TRUE)
    }
  } else {
    debug_print("No old model references found.")
    return(TRUE)
  }
}

# Check if xgboost is installed, and install it if not
check_xgboost <- function() {
  debug_print("Checking for XGBoost package")
  if(!requireNamespace("xgboost", quietly = TRUE)) {
    cat("XGBoost package not found. Attempting to install xgboost...\n")
    tryCatch({
      install.packages("xgboost", repos="https://cloud.r-project.org")
      library(xgboost)
      cat("XGBoost successfully installed and loaded.\n")
      return(TRUE)
    }, error = function(e) {
      cat("ERROR: Failed to install xgboost package:", e$message, "\n")
      cat("Will use fallback methods instead.\n")
      return(FALSE)
    })
  } else {
    # Try to load xgboost
    return(tryCatch({
      library(xgboost)
      cat("XGBoost package loaded successfully.\n")
      TRUE
    }, error = function(e) {
      cat("NOTE: Could not load XGBoost package:", e$message, "\n")
      FALSE
    }))
  }
}

# Function to check system dependencies for XGBoost
check_system_dependencies <- function() {
  os_info <- Sys.info()
  cat("Operating System:", os_info["sysname"], os_info["release"], "\n")
  
  # Check for necessary system libraries on Linux
  if(os_info["sysname"] == "Linux") {
    cat("Checking Linux dependencies...\n")
    has_gcc <- system("which gcc", ignore.stdout = TRUE) == 0
    has_cmake <- system("which cmake", ignore.stdout = TRUE) == 0
    has_make <- system("which make", ignore.stdout = TRUE) == 0
    
    if(!has_gcc || !has_cmake || !has_make) {
      cat("WARNING: Missing system dependencies required for XGBoost compilation:\n")
      if(!has_gcc) cat("- gcc (C++ compiler)\n")
      if(!has_cmake) cat("- cmake\n")
      if(!has_make) cat("- make\n")
      cat("These may need to be installed via your system package manager.\n")
      return(FALSE)
    }
  }
  
  # Check for necessary R packages
  required_for_xgboost <- c("Matrix", "data.table", "jsonlite", "Rcpp")
  missing_pkgs <- required_for_xgboost[!(required_for_xgboost %in% installed.packages()[,"Package"])]
  
  if(length(missing_pkgs) > 0) {
    cat("Installing dependencies required for XGBoost compilation:\n")
    cat(paste(missing_pkgs, collapse=", "), "\n")
    install.packages(missing_pkgs, repos="https://cloud.r-project.org")
  }
  
  return(TRUE)
}

# Multiple installation methods for XGBoost
install_xgboost <- function() {
  cat("Attempting to install XGBoost with multiple methods...\n")
  
  # Method 1: Standard CRAN installation
  cat("Method 1: Standard CRAN installation...\n")
  success <- tryCatch({
    install.packages("xgboost", repos="https://cloud.r-project.org")
    if("xgboost" %in% installed.packages()[,"Package"]) {
      library(xgboost)
      cat("CRAN installation successful!\n")
      return(TRUE)
    }
    FALSE
  }, error = function(e) {
    cat("CRAN installation failed:", e$message, "\n")
    FALSE
  })
  
  if(success) return(TRUE)
  
  # Method 2: Try via BiocManager (sometimes works when CRAN fails)
  cat("Method 2: Installation via BiocManager...\n")
  success <- tryCatch({
    if(!requireNamespace("BiocManager", quietly = TRUE))
      install.packages("BiocManager", repos="https://cloud.r-project.org")
    
    BiocManager::install("xgboost")
    if("xgboost" %in% installed.packages()[,"Package"]) {
      library(xgboost)
      cat("BiocManager installation successful!\n")
      return(TRUE)
    }
    FALSE
  }, error = function(e) {
    cat("BiocManager installation failed:", e$message, "\n")
    FALSE
  })
  
  if(success) return(TRUE)
  
  # Method 3: Try from GitHub
  cat("Method 3: Installation from GitHub...\n")
  success <- tryCatch({
    if(!requireNamespace("devtools", quietly = TRUE))
      install.packages("devtools", repos="https://cloud.r-project.org")
    
    devtools::install_github("dmlc/xgboost", subdir="R-package")
    if("xgboost" %in% installed.packages()[,"Package"]) {
      library(xgboost)
      cat("GitHub installation successful!\n")
      return(TRUE)
    }
    FALSE
  }, error = function(e) {
    cat("GitHub installation failed:", e$message, "\n")
    FALSE
  })
  
  if(success) return(TRUE)
  
  # Method 4: Try with specific version constraints
  cat("Method 4: Installation with specific version constraints...\n")
  success <- tryCatch({
    install.packages("xgboost", repos="https://cloud.r-project.org", 
                     dependencies=TRUE, INSTALL_opts = "--no-multiarch")
    if("xgboost" %in% installed.packages()[,"Package"]) {
      library(xgboost)
      cat("Installation with specific constraints successful!\n")
      return(TRUE)
    }
    FALSE
  }, error = function(e) {
    cat("Constrained installation failed:", e$message, "\n")
    FALSE
  })
  
  if(success) return(TRUE)
  
  # All methods failed
  cat("All installation methods failed.\n")
  return(FALSE)
}

# Alternative XGBoost implementation using gbm package
setup_alternative_xgboost <- function() {
  cat("Setting up alternative gradient boosting implementation using gbm package...\n")
  
  if(!requireNamespace("gbm", quietly = TRUE)) {
    install.packages("gbm", repos="https://cloud.r-project.org")
  }
  
  tryCatch({
    library(gbm)
    cat("Successfully loaded gbm package as XGBoost alternative.\n")
    return(TRUE)
  }, error = function(e) {
    cat("Failed to set up gbm package:", e$message, "\n")
    return(FALSE)
  })
}

# Main XGBoost resolution function
resolve_xgboost_issues <- function() {
  cat("Beginning XGBoost resolution process...\n")
  
  # Step 1: Check if XGBoost is already installed and loadable
  if(requireNamespace("xgboost", quietly = TRUE)) {
    tryCatch({
      library(xgboost)
      cat("XGBoost is already installed and loadable!\n")
      return(TRUE)
    }, error = function(e) {
      cat("XGBoost is installed but not loadable:", e$message, "\n")
    })
  }
  
  # Step 2: Check system dependencies
  if(!check_system_dependencies()) {
    cat("System dependency issues detected. Proceeding with caution...\n")
  }
  
  # Step 3: Try to install XGBoost
  if(install_xgboost()) {
    cat("XGBoost successfully installed!\n")
    return(TRUE)
  }
  
  # Step 4: If XGBoost installation fails, try alternative
  if(setup_alternative_xgboost()) {
    cat("Alternative gradient boosting setup successfully.\n")
    cat("Note: This uses gbm package instead of xgboost. Some features may differ.\n")
    # Define a wrapper function to make gbm work like xgboost
    assign("xgb.train", function(params, data, nrounds, ...) {
      formula <- as.formula(paste(params$objective, "~ ."))
      gbm(formula, data=data, n.trees=nrounds, distribution="bernoulli", ...)
    }, envir = .GlobalEnv)
    
    assign("xgb.importance", function(model, ...) {
      imp <- summary(model, plotit=FALSE)
      return(imp)
    }, envir = .GlobalEnv)
    
    return(TRUE)
  }
  
  # If all else fails
  cat("Could not resolve XGBoost issues. Will proceed with other models only.\n")
  return(FALSE)
}

# Main setup function that runs all setup procedures
setup_environment <- function(debug_mode = TRUE, seed_value = 123) {
  # Set debugging mode
  DEBUG_MODE <<- debug_mode
  
  # Print environment information
  print_environment_info()
  
  # Install and load packages
  setup_packages()
  
  # Set seed for reproducibility
  set.seed(seed_value)
  
  # Run the check for old models
  check_for_old_models()
  
  # Check for XGBoost
  xgboost_available <- check_xgboost()
  if(!xgboost_available) {
    cat("XGBoost not available. Will try to resolve...\n")
    xgboost_available <- resolve_xgboost_issues()
  }
  
  return(list(
    debug_mode = debug_mode,
    packages_loaded = TRUE,
    xgboost_available = xgboost_available
  ))
}

# If this script is run directly, perform setup
if(!exists("SETUP_SOURCED") || !SETUP_SOURCED) {
  cat("Running setup.R directly...\n")
  setup_environment()
  SETUP_SOURCED <- TRUE
}