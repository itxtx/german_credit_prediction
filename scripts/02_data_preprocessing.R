# ======= 02_data_preprocessing.R =======
# This script performs data preprocessing on the German Credit dataset
# - Converting categorical variables to factors
# - Handling class imbalance
# - Imputing missing values
# - Train/test split
# - Near-zero variance handling

# Source utility scripts
source("scripts/utils/setup.R")
source("scripts/utils/preprocessing.R")

# Import data if not already in environment
if(!exists("german_credit")) {
  source("scripts/01_data_import.R")
}

# Main preprocessing function
preprocess_data <- function(data, 
                          class_to_binary = TRUE,
                          handle_missing = TRUE,
                          remove_nzv = TRUE,
                          balance_method = "both", 
                          train_ratio = 0.7,
                          seed_value = 123) {
  
  message("\n====== Starting Data Preprocessing ======\n")
  
  # Set seed for reproducibility
  set.seed(seed_value)
  
  # Create a copy of the data to avoid modifying the original
  processed_data <- data
  
  # Step 1: Convert categorical variables to factors
  message("\n=== Step 1: Converting Categorical Variables to Factors ===")
  categorical_cols <- c(
    "checking_status", "credit_history", "purpose", "savings_status", 
    "employment", "personal_status", "other_parties", "property_magnitude", 
    "other_payment_plans", "housing", "job", "own_telephone", 
    "foreign_worker", "class"
  )
  
  # Original data types
  original_types <- sapply(processed_data[categorical_cols], class)
  message("Original variable types:")
  print(original_types)
  
  # Convert to factors
  processed_data <- convert_to_factors(processed_data, categorical_cols)
  
  # Verify conversion
  new_types <- sapply(processed_data[categorical_cols], class)
  message("New variable types after conversion:")
  print(new_types)
  
  # Step 2: Convert class to binary Good/Bad if requested
  if(class_to_binary) {
    message("\n=== Step 2: Converting Class to Binary (Good/Bad) ===")
    # Check current class values
    current_values <- levels(processed_data$class)
    message("Current class levels: ", paste(current_values, collapse = ", "))
    
    # Store original class distribution
    orig_class_dist <- table(processed_data$class)
    message("Original class distribution:")
    print(orig_class_dist)
    
    # Handle different possible formats
    if(length(current_values) == 2) {
      if(all(current_values %in% c("1", "2"))) {
        processed_data$class <- factor(ifelse(processed_data$class == "1", "Good", "Bad"))
        message("Converted numeric classes (1/2) to Good/Bad")
      } else if(all(current_values %in% c("A201", "A202"))) {
        processed_data$class <- factor(ifelse(processed_data$class == "A201", "Good", "Bad"))
        message("Converted text classes (A201/A202) to Good/Bad")
      } else {
        # Try to infer which is which based on proportions (assuming good credit is more common)
        major_class <- names(which.max(table(processed_data$class)))
        processed_data$class <- factor(ifelse(processed_data$class == major_class, "Good", "Bad"))
        message("Inferred class conversion based on frequencies")
      }
    } else {
      warning("Unexpected number of class levels. Check the class variable.")
    }
    
    # Verify the conversion worked
    message("Class distribution after conversion:")
    print(table(processed_data$class))
  } else {
    message("\n=== Step 2: Skipping class conversion (keeping original) ===")
  }
  
  # Step 3: Check for missing values and impute if necessary
  if(handle_missing) {
    message("\n=== Step 3: Checking and Handling Missing Values ===")
    # Check for missing values
    missing_values <- colSums(is.na(processed_data))
    total_missing <- sum(missing_values)
    
    if(total_missing > 0) {
      message("Found ", total_missing, " missing values across ", 
             sum(missing_values > 0), " columns.")
      # Impute missing values
      processed_data <- handle_missing_values(processed_data)
    } else {
      message("No missing values found in the dataset.")
    }
  } else {
    message("\n=== Step 3: Skipping missing value handling ===")
  }
  
  # Step 4: Create train/test split
  message("\n=== Step 4: Creating Train/Test Split ===")
  split_result <- create_train_test_split(processed_data, "class", p = train_ratio, seed = seed_value)
  train_data <- split_result$train
  test_data <- split_result$test
  
  if(nrow(test_data) == 0) {
    stop("Test data split resulted in 0 rows. Check the train_ratio parameter and data size.")
  }
  
  # Step 5: Check for near-zero variance predictors
  if(remove_nzv) {
    message("\n=== Step 5: Checking for Near-Zero Variance Predictors ===")
    nzv_result <- check_near_zero_variance(train_data, saveMetrics = TRUE)
    
    # Extract names of NZV predictors
    nzv_cols <- rownames(nzv_result)[nzv_result$nzv]
    
    if(length(nzv_cols) > 0) {
      message("Removing ", length(nzv_cols), " near-zero variance predictors: ", 
             paste(nzv_cols, collapse = ", "))
      
      # Remove NZV predictors from train and test sets
      train_data <- train_data[, !names(train_data) %in% nzv_cols]
      test_data <- test_data[, !names(test_data) %in% nzv_cols]
    } else {
      message("No near-zero variance predictors found.")
    }
  } else {
    message("\n=== Step 5: Skipping near-zero variance handling ===")
    nzv_result <- NULL
    nzv_cols <- NULL
  }
  
  # Step 6: Balance classes in training data only
  if(!is.null(balance_method)) {
    message("\n=== Step 6: Handling Class Imbalance ===")
    message("Original training data class distribution:")
    print(table(train_data$class))
    
    # Balance the training data
    train_data <- balance_classes(train_data, "class", method = balance_method, seed = seed_value)
    
    message("Class distribution after balancing:")
    print(table(train_data$class))
  } else {
    message("\n=== Step 6: Skipping class balancing ===")
  }
  
  # Step 7: Check for highly correlated numeric predictors
  message("\n=== Step 7: Checking for Highly Correlated Predictors ===")
  numeric_cols <- names(train_data)[sapply(train_data, is.numeric)]
  
  if(length(numeric_cols) > 1) {
    corr_result <- check_correlations(train_data[, numeric_cols], cutoff = 0.75)
    
    if(!is.null(corr_result) && nrow(corr_result) > 0) {
      message("Found ", nrow(corr_result), " highly correlated pairs.")
      # Note: We're not removing correlated predictors by default, 
      # just flagging them for awareness
    } else {
      message("No highly correlated numeric predictors found.")
    }
  } else {
    message("Insufficient numeric variables to check correlations.")
    corr_result <- NULL
  }
  
  # Step 8: Scale numeric features
  message("\n=== Step 8: Scaling Numeric Features ===")
  numeric_cols <- names(train_data)[sapply(train_data, is.numeric)]
  
  if(length(numeric_cols) > 0) {
    message("Scaling ", length(numeric_cols), " numeric features.")
    
    # Scale both training and test data
    scaled_data <- scale_features(train_data, test_data)
    train_data <- scaled_data$train
    test_data <- scaled_data$test
    scaling_params <- scaled_data$params
  } else {
    message("No numeric features to scale.")
    scaling_params <- NULL
  }
  
  # Step 9: Ensure test set has same factor levels as train set
  message("\n=== Step 9: Aligning Factor Levels Between Train and Test Sets ===")
  factor_cols <- names(train_data)[sapply(train_data, is.factor)]
  level_mismatches <- 0
  
  for(col in factor_cols) {
    train_levels <- levels(train_data[[col]])
    test_levels <- levels(test_data[[col]])
    
    # Check if levels match
    if(!identical(train_levels, test_levels)) {
      level_mismatches <- level_mismatches + 1
      
      # Get all levels from both datasets
      all_levels <- unique(c(train_levels, test_levels))
      
      # Set the levels for both datasets
      levels(train_data[[col]]) <- all_levels
      levels(test_data[[col]]) <- all_levels
    }
  }
  
  if(level_mismatches > 0) {
    message("Fixed level mismatches in ", level_mismatches, " factor columns.")
  } else {
    message("All factor levels already aligned between train and test sets.")
  }
  
  # Step 10: Save processed datasets
  message("\n=== Step 10: Saving Processed Datasets ===")
  
  # Create directory if it doesn't exist
  processed_dir <- "data/processed"
  if(!dir.exists(processed_dir)) {
    dir.create(processed_dir, recursive = TRUE)
    message("Created directory: ", processed_dir)
  }
  
  # Save train and test datasets
  train_path <- file.path(processed_dir, "train_data.csv")
  write.csv(train_data, train_path, row.names = FALSE)
  message("Training data saved to: ", train_path)
  
  test_path <- file.path(processed_dir, "test_data.csv")
  write.csv(test_data, test_path, row.names = FALSE)
  message("Test data saved to: ", test_path)
  
  # Save preprocessing information
  preprocess_info <- list(
    categorical_columns = categorical_cols,
    nzv_columns = nzv_cols,
    numeric_columns = numeric_cols,
    scaling_parameters = scaling_params,
    correlation_results = corr_result,
    train_samples = nrow(train_data),
    test_samples = nrow(test_data),
    train_class_distribution = table(train_data$class),
    test_class_distribution = table(test_data$class)
  )
  
  # Create results directory if it doesn't exist
  results_dir <- "results"
  if(!dir.exists(results_dir)) {
    dir.create(results_dir, recursive = TRUE)
  }
  
  # Save preprocessing info
  preprocess_path <- file.path(results_dir, "preprocessing_info.RData")
  save(preprocess_info, file = preprocess_path)
  message("Preprocessing information saved to: ", preprocess_path)
  
  message("\n====== Data Preprocessing Complete ======\n")
  
  # Add feature names to preprocessing info
  preprocess_info$feature_names <- names(train_data)
  
  # Return processed data and preprocessing steps
  return(list(
    train = train_data,
    test = test_data,
    preprocessing_steps = preprocess_info
  ))
}

# Add these validation functions after the preprocess_data function but before main()

#' Validate input data structure and contents
#' @param data Data frame to validate
#' @param required_features Vector of required feature names
#' @param model_type Type of model being used
#' @return List with validation status and any error messages
validate_model_data <- function(data, required_features, model_type = NULL) {
  validation_results <- list(
    is_valid = TRUE,
    messages = character()
  )
  
  # Check if data is a data frame
  if (!is.data.frame(data)) {
    validation_results$is_valid <- FALSE
    validation_results$messages <- c(validation_results$messages,
                                   "Input must be a data frame")
    return(validation_results)
  }
  
  # Check for required features
  missing_features <- setdiff(required_features, names(data))
  if (length(missing_features) > 0) {
    validation_results$is_valid <- FALSE
    validation_results$messages <- c(validation_results$messages,
                                   paste("Missing required features:",
                                         paste(missing_features, collapse = ", ")))
  }
  
  # Check for appropriate data types
  for (col in names(data)) {
    if (col %in% required_features) {
      expected_type <- get_expected_type(col, model_type)
      actual_type <- class(data[[col]])[1]
      
      if (actual_type != expected_type) {
        validation_results$is_valid <- FALSE
        validation_results$messages <- c(validation_results$messages,
                                       paste("Invalid type for", col, 
                                             "- Expected:", expected_type,
                                             "Got:", actual_type))
      }
    }
  }
  
  # Check for missing values
  na_cols <- colnames(data)[colSums(is.na(data)) > 0]
  if (length(na_cols) > 0) {
    validation_results$is_valid <- FALSE
    validation_results$messages <- c(validation_results$messages,
                                   paste("Missing values found in columns:",
                                         paste(na_cols, collapse = ", ")))
  }
  
  return(validation_results)
}

#' Get expected data type for a feature
#' @param feature_name Name of the feature
#' @param model_type Type of model being used
#' @return Expected R data type as string
get_expected_type <- function(feature_name, model_type = NULL) {
  # Define expected types for known features
  categorical_features <- c(
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone",
    "foreign_worker", "class"
  )
  
  numeric_features <- c(
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents"
  )
  
  if (feature_name %in% categorical_features) {
    return("factor")
  } else if (feature_name %in% numeric_features) {
    return("numeric")
  } else {
    return(NA_character_)
  }
}

#' Prepare new data for model prediction
#' @param data New data to prepare
#' @param preprocessing_steps List of preprocessing steps from training
#' @param model_type Type of model being used
#' @return Processed data frame ready for prediction
prepare_model_data <- function(data, preprocessing_steps, model_type = NULL) {
  # Validate input data
  validation <- validate_model_data(data, 
                                  preprocessing_steps$feature_names,
                                  model_type)
  
  if (!validation$is_valid) {
    stop("Data validation failed:\n",
         paste(validation$messages, collapse = "\n"))
  }
  
  # Apply preprocessing steps
  processed_data <- data
  
  # Convert categorical variables to factors
  categorical_cols <- preprocessing_steps$categorical_columns
  processed_data <- convert_to_factors(processed_data, categorical_cols)
  
  # Scale numeric features using saved parameters
  if (!is.null(preprocessing_steps$scaling_parameters)) {
    numeric_cols <- names(preprocessing_steps$scaling_parameters$center)
    for (col in numeric_cols) {
      if (col %in% names(processed_data)) {
        processed_data[[col]] <- scale(processed_data[[col]],
                                     center = preprocessing_steps$scaling_parameters$center[col],
                                     scale = preprocessing_steps$scaling_parameters$scale[col])
      }
    }
  }
  
  return(processed_data)
}

# Main execution section
main <- function() {
  # Ensure german_credit data is loaded
  if(!exists("german_credit")) {
    source("scripts/01_data_import.R")
  }
  
  # Run preprocessing
  processed_data <- preprocess_data(
    german_credit,
    class_to_binary = TRUE,
    handle_missing = TRUE,
    remove_nzv = TRUE,
    balance_method = "both",
    train_ratio = 0.7,
    seed_value = 123
  )
  
  # Return processed data
  return(processed_data)
}

# Run the main function if this script is being run directly
if(!exists("DATA_PREPROCESSING_SOURCED") || !DATA_PREPROCESSING_SOURCED) {
  processed_data <- main()
  train_data <- processed_data$train
  test_data <- processed_data$test
  DATA_PREPROCESSING_SOURCED <- TRUE
} else {
  message("02_data_preprocessing.R has been sourced. Use main() to run the preprocessing process.")
}

# Update the binning function
create_bins <- function(data) {
  # Create duration bins with better ranges
  data$duration_bin <- cut(data$duration,
                          breaks = c(-Inf, 12, 24, 36, 48, Inf),
                          labels = c("0-12", "13-24", "25-36", "37-48", "48+"),
                          include.lowest = TRUE)
  
  # Create credit amount bins with better ranges
  data$credit_amount_bin <- cut(data$credit_amount,
                               breaks = c(-Inf, 5000, 10000, 15000, Inf),
                               labels = c("0-5k", "5k-10k", "10k-15k", "15k+"),
                               include.lowest = TRUE)
  
  # Check for NAs and log warnings
  na_counts <- list(
    duration_bin = sum(is.na(data$duration_bin)),
    credit_amount_bin = sum(is.na(data$credit_amount_bin))
  )
  
  if(any(unlist(na_counts) > 0)) {
    warning("NAs found in binned features:\n",
            "duration_bin: ", na_counts$duration_bin, "\n",
            "credit_amount_bin: ", na_counts$credit_amount_bin)
    
    # Fill NAs with a new category
    data$duration_bin[is.na(data$duration_bin)] <- "unknown"
    data$credit_amount_bin[is.na(data$credit_amount_bin)] <- "unknown"
    
    # Convert back to factors with new level
    data$duration_bin <- factor(data$duration_bin)
    data$credit_amount_bin <- factor(data$credit_amount_bin)
  }
  
  return(data)
}