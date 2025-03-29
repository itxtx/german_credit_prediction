# ======= preprocessing.R =======
# This script contains functions for data preprocessing, including:
# - Class balancing
# - Handling missing values
# - Feature scaling
# - One-hot encoding

# Load required libraries if not already loaded
required_packages <- c("caret", "dplyr", "recipes")
for(pkg in required_packages) {
  if(!requireNamespace(pkg, quietly = TRUE)) {
    message(paste("Loading package:", pkg))
    library(pkg, character.only = TRUE)
  }
}

# Function to identify categorical and numeric columns
identify_column_types <- function(data) {
  # Identify categorical variables (factors or character)
  categorical_cols <- names(data)[sapply(data, function(x) is.factor(x) || is.character(x))]
  
  # Identify numeric variables
  numeric_cols <- names(data)[sapply(data, is.numeric)]
  
  return(list(
    categorical = categorical_cols,
    numeric = numeric_cols
  ))
}

# Function to convert categorical variables to factors
convert_to_factors <- function(data, categorical_cols = NULL) {
  # If categorical columns not specified, detect them
  if(is.null(categorical_cols)) {
    categorical_cols <- names(data)[sapply(data, function(x) is.character(x) || is.logical(x))]
  }
  
  # Convert specified columns to factors
  if(length(categorical_cols) > 0) {
    data[categorical_cols] <- lapply(data[categorical_cols], as.factor)
  }
  
  return(data)
}

# Function to handle missing values
handle_missing_values <- function(data, numeric_strategy = "median", categorical_strategy = "mode") {
  # Check for missing values
  missing_values <- colSums(is.na(data))
  total_missing <- sum(missing_values)
  
  if(total_missing == 0) {
    message("No missing values found.")
    return(data)
  }
  
  message(paste("Found", total_missing, "missing values across", sum(missing_values > 0), "columns."))
  
  # Identify column types
  col_types <- identify_column_types(data)
  
  # Handle missing values in numeric columns
  numeric_cols_with_na <- col_types$numeric[col_types$numeric %in% names(missing_values[missing_values > 0])]
  if(length(numeric_cols_with_na) > 0) {
    for(col in numeric_cols_with_na) {
      if(numeric_strategy == "median") {
        replacement_value <- median(data[[col]], na.rm = TRUE)
      } else if(numeric_strategy == "mean") {
        replacement_value <- mean(data[[col]], na.rm = TRUE)
      } else if(numeric_strategy == "zero") {
        replacement_value <- 0
      } else {
        stop("Invalid numeric_strategy. Choose 'median', 'mean', or 'zero'.")
      }
      
      data[is.na(data[[col]]), col] <- replacement_value
      message(paste("Replaced", sum(is.na(data[[col]])), "missing values in", col, "with", replacement_value))
    }
  }
  
  # Handle missing values in categorical columns
  categorical_cols_with_na <- col_types$categorical[col_types$categorical %in% names(missing_values[missing_values > 0])]
  if(length(categorical_cols_with_na) > 0) {
    for(col in categorical_cols_with_na) {
      if(categorical_strategy == "mode") {
        # Find mode (most frequent value)
        freq_table <- table(data[[col]], useNA = "no")
        replacement_value <- names(freq_table)[which.max(freq_table)]
      } else if(categorical_strategy == "new_category") {
        # Create a new category for missing values
        levels(data[[col]]) <- c(levels(data[[col]]), "Missing")
        replacement_value <- "Missing"
      } else {
        stop("Invalid categorical_strategy. Choose 'mode' or 'new_category'.")
      }
      
      data[is.na(data[[col]]), col] <- replacement_value
      message(paste("Replaced", sum(is.na(data[[col]])), "missing values in", col, "with", replacement_value))
    }
  }
  
  # Verify all missing values have been handled
  remaining_missing <- sum(is.na(data))
  if(remaining_missing > 0) {
    warning(paste("There are still", remaining_missing, "missing values in the dataset."))
  } else {
    message("All missing values have been handled successfully.")
  }
  
  return(data)
}

# Function to perform feature scaling
scale_features <- function(train_data, test_data = NULL, columns = NULL, method = "z-score") {
  # If columns not specified, use all numeric columns
  if(is.null(columns)) {
    columns <- names(train_data)[sapply(train_data, is.numeric)]
  }
  
  # Remove target/class column if present in columns
  class_cols <- c("class", "target", "label", "y")
  columns <- setdiff(columns, class_cols)
  
  if(length(columns) == 0) {
    message("No numeric columns to scale.")
    if(is.null(test_data)) {
      return(train_data)
    } else {
      return(list(train = train_data, test = test_data))
    }
  }
  
  # Create preprocessing object based on training data
  if(method == "z-score") {
    # Z-score standardization (mean=0, sd=1)
    preprocess_params <- list()
    for(col in columns) {
      preprocess_params[[col]] <- list(
        mean = mean(train_data[[col]], na.rm = TRUE),
        sd = sd(train_data[[col]], na.rm = TRUE)
      )
    }
    
    # Apply to training data
    for(col in columns) {
      mean_val <- preprocess_params[[col]]$mean
      sd_val <- preprocess_params[[col]]$sd
      
      # Avoid division by zero
      if(sd_val > 0) {
        train_data[[col]] <- (train_data[[col]] - mean_val) / sd_val
      } else {
        warning(paste("Column", col, "has zero standard deviation. Scaling skipped."))
      }
    }
    
    # Apply to test data if provided
    if(!is.null(test_data)) {
      for(col in columns) {
        mean_val <- preprocess_params[[col]]$mean
        sd_val <- preprocess_params[[col]]$sd
        
        # Avoid division by zero
        if(sd_val > 0) {
          test_data[[col]] <- (test_data[[col]] - mean_val) / sd_val
        }
      }
    }
  } else if(method == "min-max") {
    # Min-max normalization (range 0-1)
    preprocess_params <- list()
    for(col in columns) {
      preprocess_params[[col]] <- list(
        min = min(train_data[[col]], na.rm = TRUE),
        max = max(train_data[[col]], na.rm = TRUE)
      )
    }
    
    # Apply to training data
    for(col in columns) {
      min_val <- preprocess_params[[col]]$min
      max_val <- preprocess_params[[col]]$max
      
      # Avoid division by zero
      if(max_val > min_val) {
        train_data[[col]] <- (train_data[[col]] - min_val) / (max_val - min_val)
      } else {
        warning(paste("Column", col, "has no range. Scaling skipped."))
      }
    }
    
    # Apply to test data if provided
    if(!is.null(test_data)) {
      for(col in columns) {
        min_val <- preprocess_params[[col]]$min
        max_val <- preprocess_params[[col]]$max
        
        # Avoid division by zero
        if(max_val > min_val) {
          test_data[[col]] <- (test_data[[col]] - min_val) / (max_val - min_val)
        }
      }
    }
  } else {
    stop("Invalid scaling method. Choose 'z-score' or 'min-max'.")
  }
  
  # Return the scaled data
  if(is.null(test_data)) {
    return(train_data)
  } else {
    return(list(train = train_data, test = test_data, params = preprocess_params))
  }
}

# Function for one-hot encoding
one_hot_encode <- function(train_data, test_data = NULL, columns = NULL, drop_original = TRUE) {
  # If columns not specified, use all factor columns
  if(is.null(columns)) {
    columns <- names(train_data)[sapply(train_data, is.factor)]
  }
  
  # Remove class/target column if it's a factor
  class_cols <- c("class", "target", "label", "y")
  columns <- setdiff(columns, intersect(class_cols, columns))
  
  if(length(columns) == 0) {
    message("No categorical columns to encode.")
    if(is.null(test_data)) {
      return(train_data)
    } else {
      return(list(train = train_data, test = test_data))
    }
  }
  
  # Create dummy variables using caret
  for(col in columns) {
    # Convert to factor if not already
    if(!is.factor(train_data[[col]])) {
      train_data[[col]] <- as.factor(train_data[[col]])
      if(!is.null(test_data)) {
        test_data[[col]] <- as.factor(test_data[[col]])
      }
    }
    
    # Create dummies
    dummies <- caret::dummyVars(paste0("~", col), data = train_data)
    
    # Apply to training data
    train_dummies <- predict(dummies, newdata = train_data)
    train_data <- cbind(train_data, train_dummies)
    
    # Apply to test data if provided
    if(!is.null(test_data)) {
      test_dummies <- predict(dummies, newdata = test_data)
      test_data <- cbind(test_data, test_dummies)
    }
    
    # Remove original column if specified
    if(drop_original) {
      train_data[[col]] <- NULL
      if(!is.null(test_data)) {
        test_data[[col]] <- NULL
      }
    }
  }
  
  # Return the encoded data
  if(is.null(test_data)) {
    return(train_data)
  } else {
    return(list(train = train_data, test = test_data))
  }
}

# Function to handle class imbalance using over/under sampling
balance_classes <- function(data, class_column = "class", method = "both", ratio = 1.0, seed = NULL) {
  # Set seed if provided
  if(!is.null(seed)) {
    set.seed(seed)
  }
  
  # Verify the class column exists
  if(!(class_column %in% colnames(data))) {
    stop(paste("Class column", class_column, "not found in the dataset."))
  }
  
  # Ensure class column is a factor
  data[[class_column]] <- as.factor(data[[class_column]])
  
  # Get class counts
  class_counts <- table(data[[class_column]])
  message("Original class distribution:")
  print(class_counts)
  
  # Identify majority and minority classes
  majority_class <- names(class_counts)[which.max(class_counts)]
  minority_class <- names(class_counts)[which.min(class_counts)]
  
  # Calculate target count based on method and ratio
  if(method == "both") {
    # Balance both classes to be approximately equal
    target_count <- min(max(class_counts) / ratio, min(class_counts) * ratio)
  } else if(method == "oversample") {
    # Only oversample minority class to match majority
    target_count <- max(class_counts)
  } else if(method == "undersample") {
    # Only undersample majority class to match minority
    target_count <- min(class_counts)
  } else {
    stop("Invalid method. Choose 'both', 'oversample', or 'undersample'.")
  }
  
  message(paste("Target count per class:", target_count))
  
  # Separate classes
  majority_data <- data[data[[class_column]] == majority_class, ]
  minority_data <- data[data[[class_column]] == minority_class, ]
  
  # Sample based on method
  if(method %in% c("both", "undersample")) {
    # Undersample majority class
    majority_sample <- majority_data[sample(1:nrow(majority_data), min(nrow(majority_data), target_count)), ]
  } else {
    # Keep all majority samples
    majority_sample <- majority_data
  }
  
  if(method %in% c("both", "oversample")) {
    # Oversample minority class
    if(target_count > nrow(minority_data)) {
      # Need to oversample with replacement
      minority_sample <- minority_data[sample(1:nrow(minority_data), target_count, replace = TRUE), ]
    } else {
      # Can just take a sample without replacement
      minority_sample <- minority_data[sample(1:nrow(minority_data), target_count), ]
    }
  } else {
    # Keep all minority samples
    minority_sample <- minority_data
  }
  
  # Combine the balanced samples
  balanced_data <- rbind(majority_sample, minority_sample)
  
  # Shuffle the rows to mix the classes
  balanced_data <- balanced_data[sample(1:nrow(balanced_data)), ]
  
  # Check final class distribution
  final_counts <- table(balanced_data[[class_column]])
  message("Final class distribution after balancing:")
  print(final_counts)
  
  return(balanced_data)
}

# Function to check for near-zero variance predictors
check_near_zero_variance <- function(data, freqCut = 95/5, uniqueCut = 10, saveMetrics = FALSE) {
  # Use caret's nearZeroVar function
  nzv <- caret::nearZeroVar(data, freqCut = freqCut, uniqueCut = uniqueCut, saveMetrics = TRUE)
  
  # Report results
  if(any(nzv$nzv)) {
    message(paste("Found", sum(nzv$nzv), "near-zero variance predictors:"))
    print(rownames(nzv)[nzv$nzv])
  } else {
    message("No near-zero variance predictors found.")
  }
  
  # Return full metrics or just the names of NZV predictors
  if(saveMetrics) {
    return(nzv)
  } else {
    return(rownames(nzv)[nzv$nzv])
  }
}

# Function to check for highly correlated predictors
check_correlations <- function(data, cutoff = 0.75, use = "pairwise.complete.obs") {
  # Identify numeric columns
  numeric_cols <- names(data)[sapply(data, is.numeric)]
  
  if(length(numeric_cols) <= 1) {
    message("Insufficient numeric columns to calculate correlations.")
    return(NULL)
  }
  
  # Calculate correlation matrix
  correlation_matrix <- cor(data[, numeric_cols], use = use)
  
  # Find highly correlated pairs
  high_cor <- which(abs(correlation_matrix) > cutoff & abs(correlation_matrix) < 1, arr.ind = TRUE)
  
  # Convert to pairs
  if(length(high_cor) > 0) {
    # Get unique pairs (avoid duplicates due to symmetry)
    high_cor <- high_cor[high_cor[,1] < high_cor[,2], ]
    
    if(nrow(high_cor) > 0) {
      # Create a data frame of correlated pairs
      cor_pairs <- data.frame(
        Var1 = numeric_cols[high_cor[,1]],
        Var2 = numeric_cols[high_cor[,2]],
        Correlation = diag(correlation_matrix[high_cor[,1], high_cor[,2]])
      )
      
      # Sort by absolute correlation value
      cor_pairs <- cor_pairs[order(abs(cor_pairs$Correlation), decreasing = TRUE), ]
      
      message(paste("Found", nrow(cor_pairs), "highly correlated pairs (|r| >", cutoff, "):"))
      print(cor_pairs)
      
      return(cor_pairs)
    }
  }
  
  message("No highly correlated pairs found.")
  return(NULL)
}

# Function to create train/test split with stratification
create_train_test_split <- function(data, class_column = "class", p = 0.7, seed = NULL) {
  # Set seed if provided
  if(!is.null(seed)) {
    set.seed(seed)
  }
  
  # Create stratified split using caret
  train_index <- caret::createDataPartition(data[[class_column]], p = p, list = FALSE)
  
  # Split the data
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  # Report on the split
  message(paste("Created train/test split:", 
                nrow(train_data), "training samples,", 
                nrow(test_data), "test samples"))
  
  # Check class distributions
  train_dist <- prop.table(table(train_data[[class_column]])) * 100
  test_dist <- prop.table(table(test_data[[class_column]])) * 100
  
  message("Class distribution in training set (%):")
  print(train_dist)
  message("Class distribution in testing set (%):")
  print(test_dist)
  
  return(list(train = train_data, test = test_data))
}

# Function to process the German Credit dataset for modeling
preprocess_german_credit <- function(data, class_to_binary = TRUE, 
                                   balance_classes_method = "both",
                                   train_test_split = 0.7, seed = 123, 
                                   model_type = NULL) {
  # Set seed for reproducibility
  set.seed(seed)
  
  # Make a copy of the data to avoid modifying the original
  processed_data <- data
  
  # Store original column names and order
  original_columns <- names(processed_data)
  
  # 1. Convert categorical variables to factors
  categorical_cols <- c(
    "checking_status", "credit_history", "purpose", "savings_status", 
    "employment", "personal_status", "other_parties", "property_magnitude", 
    "other_payment_plans", "housing", "job", "own_telephone", 
    "foreign_worker", "class"
  )
  
  processed_data <- convert_to_factors(processed_data, categorical_cols)
  
  # 2. Convert class to binary Good/Bad if requested
  if(class_to_binary) {
    processed_data$class <- standardize_class_labels(processed_data$class)
  }
  
  # 3. Handle missing values before split to ensure consistent treatment
  processed_data <- handle_missing_values(processed_data)
  
  # 4. Create train/test split
  split <- create_train_test_split(processed_data, "class", p = train_test_split, seed = seed)
  train_data <- split$train
  test_data <- split$test
  
  # Ensure factor consistency between train and test
  test_data <- ensure_factor_consistency(train_data, test_data)
  
  # Validate data consistency
  validation_result <- validate_data_consistency(train_data, test_data)
  if(!validation_result$is_valid) {
    warning("Data consistency issues found:\n",
            paste(validation_result$messages, collapse="\n"))
  }
  
  # 5. Balance classes in training data only
  if(!is.null(balance_classes_method)) {
    train_data <- balance_classes(train_data, "class", method = balance_classes_method, seed = seed)
  }
  
  # 6. Feature engineering based on model type (if specified)
  if(!is.null(model_type)) {
    train_data <- prepare_model_data(train_data, model_type)
    test_data <- prepare_model_data(test_data, model_type)
    
    # Re-validate after model-specific preprocessing
    validation_result <- validate_data_consistency(train_data, test_data)
    if(!validation_result$is_valid) {
      warning("Post-processing data consistency issues found:\n",
              paste(validation_result$messages, collapse="\n"))
    }
  }
  
  # 7. Check for near-zero variance predictors using only training data
  nzv <- check_near_zero_variance(train_data)
  if(length(nzv) > 0) {
    message("Removing near-zero variance predictors...")
    train_data <- train_data[, !names(train_data) %in% nzv]
    test_data <- test_data[, !names(test_data) %in% nzv]
  }
  
  # 8. Scale numeric features
  scaled_data <- scale_features(train_data, test_data)
  train_data <- scaled_data$train
  test_data <- scaled_data$test
  
  # 9. Ensure column consistency between train and test
  train_data <- ensure_column_consistency(train_data, test_data)
  test_data <- ensure_column_consistency(test_data, train_data)
  
  # 10. Preserve column order
  column_order <- union(original_columns, names(train_data))
  train_data <- train_data[, column_order[column_order %in% names(train_data)]]
  test_data <- test_data[, column_order[column_order %in% names(test_data)]]
  
  return(list(
    train = train_data,
    test = test_data,
    preprocessing_steps = list(
      nzv_removed = nzv,
      scaling_params = scaled_data$params,
      feature_names = names(train_data),
      categorical_cols = categorical_cols,
      column_order = names(train_data)
    )
  ))
}

# New function to standardize class labels
standardize_class_labels <- function(class_vector) {
  # Convert to factor if not already
  class_vector <- as.factor(class_vector)
  
  # Handle different possible formats
  if(all(levels(class_vector) %in% c("1", "2"))) {
    return(factor(ifelse(class_vector == "1", "Good", "Bad")))
  } else if(all(levels(class_vector) %in% c("A201", "A202"))) {
    return(factor(ifelse(class_vector == "A201", "Good", "Bad")))
  } else if(all(levels(class_vector) %in% c("Good", "Bad"))) {
    return(class_vector)
  } else {
    # Try to infer based on frequencies
    major_class <- names(which.max(table(class_vector)))
    return(factor(ifelse(class_vector == major_class, "Good", "Bad")))
  }
}

# New function to ensure column consistency
ensure_column_consistency <- function(data1, data2) {
  # Find columns in data2 that are missing in data1
  missing_cols <- setdiff(names(data2), names(data1))
  
  if(length(missing_cols) > 0) {
    for(col in missing_cols) {
      if(is.factor(data2[[col]])) {
        data1[[col]] <- factor(NA, levels = levels(data2[[col]]))
      } else if(is.numeric(data2[[col]])) {
        data1[[col]] <- NA_real_
      } else {
        data1[[col]] <- NA
      }
    }
  }
  
  return(data1)
}

# Update prepare_model_data function
prepare_model_data <- function(data, model_type) {
  if(is.null(model_type)) return(data)
  
  processed_data <- data.frame(data) # Ensure we're working with a data frame
  
  # Model-specific preprocessing
  if(model_type == "naive_bayes") {
    # Ensure all categorical variables are properly factored
    categorical_cols <- names(processed_data)[sapply(processed_data, function(x) 
      is.character(x) || is.logical(x) || is.factor(x))]
    
    for(col in categorical_cols) {
      if(!is.factor(processed_data[[col]])) {
        processed_data[[col]] <- as.factor(processed_data[[col]])
      }
    }
    
    # Bin numeric variables to handle continuous data
    numeric_cols <- names(processed_data)[sapply(processed_data, is.numeric)]
    for(col in numeric_cols) {
      # Create bins using training data distribution
      breaks <- quantile(processed_data[[col]], probs = seq(0, 1, 0.2), na.rm = TRUE)
      bin_name <- paste0(col, "_bin")
      processed_data[[bin_name]] <- cut(processed_data[[col]], 
                                      breaks = breaks,
                                      labels = paste0("bin", 1:5),
                                      include.lowest = TRUE)
    }
  }
  
  # Standard derived features for tree-based models
  if(model_type %in% c("decision_tree", "random_forest", "xgboost")) {
    if(all(c("age", "employment") %in% names(processed_data))) {
      processed_data$age_employment_ratio <- processed_data$age / 
        as.numeric(processed_data$employment)
    }
    if(all(c("credit_amount", "duration") %in% names(processed_data))) {
      processed_data$monthly_payment <- processed_data$credit_amount / 
        processed_data$duration
    }
    if("employment" %in% names(processed_data)) {
      processed_data$employment_years <- as.numeric(processed_data$employment)
    }
  }
  
  # Preserve column names
  names(processed_data) <- make.names(names(processed_data), unique = TRUE)
  
  return(processed_data)
}

# Update validate_model_data to be more robust
validate_model_data <- function(data, required_features, model_type = NULL) {
  # Check for required features
  missing_features <- setdiff(required_features, names(data))
  if(length(missing_features) > 0) {
    stop("Missing required features: ", paste(missing_features, collapse = ", "))
  }
  
  # Validate data types
  for(col in names(data)) {
    if(is.factor(data[[col]]) && any(is.na(data[[col]]))) {
      warning("NA values found in factor column: ", col)
    }
  }
  
  # Model-specific validations
  if(!is.null(model_type)) {
    if(model_type == "naive_bayes") {
      # Check that categorical variables are factors
      non_factor_cats <- names(data)[sapply(data, function(x) 
        is.character(x) && !is.factor(x))]
      if(length(non_factor_cats) > 0) {
        warning("Character columns should be factors: ", 
                paste(non_factor_cats, collapse = ", "))
      }
    }
  }
  
  return(TRUE)
}

# If this script is run directly, demonstrate functionality
if(!exists("PREPROCESSING_SOURCED") || !PREPROCESSING_SOURCED) {
  message("This script contains preprocessing functions for the German Credit Analysis project.")
  message("Source this script in your main analysis code to use these functions.")
  PREPROCESSING_SOURCED <- TRUE
}

# Function to prepare data for model training/prediction
prepare_data <- function(data, model_type) {
  # This is just an alias for prepare_model_data to maintain compatibility
  return(prepare_model_data(data, model_type))
}

# Function to ensure consistent factor levels between datasets
ensure_factor_consistency <- function(reference_data, target_data) {
  message("Ensuring factor level consistency between datasets...")
  
  # Get all factor columns
  factor_cols <- names(reference_data)[sapply(reference_data, is.factor)]
  modifications <- 0
  
  for(col in factor_cols) {
    if(!is.factor(target_data[[col]])) {
      target_data[[col]] <- as.factor(target_data[[col]])
      modifications <- modifications + 1
    }
    
    ref_levels <- levels(reference_data[[col]])
    target_levels <- levels(target_data[[col]])
    
    # Check for new levels in target data
    new_levels <- setdiff(target_levels, ref_levels)
    if(length(new_levels) > 0) {
      warning(sprintf("Column '%s' has %d new levels in target data: %s", 
                     col, length(new_levels), paste(new_levels, collapse=", ")))
      # Replace new levels with NA or most frequent level
      target_data[[col]] <- factor(as.character(target_data[[col]]), 
                                  levels = ref_levels)
      modifications <- modifications + 1
    }
    
    # Ensure all reference levels exist in target
    if(!identical(levels(target_data[[col]]), ref_levels)) {
      levels(target_data[[col]]) <- ref_levels
      modifications <- modifications + 1
    }
  }
  
  message(sprintf("Made %d modifications to ensure factor consistency", modifications))
  return(target_data)
}

# Function to validate data consistency
validate_data_consistency <- function(train_data, test_data) {
  validation_results <- list(
    is_valid = TRUE,
    messages = character()
  )
  
  # Check column presence
  train_cols <- colnames(train_data)
  test_cols <- colnames(test_data)
  
  missing_cols <- setdiff(train_cols, test_cols)
  if(length(missing_cols) > 0) {
    validation_results$is_valid <- FALSE
    validation_results$messages <- c(validation_results$messages,
      sprintf("Missing columns in test data: %s", paste(missing_cols, collapse=", ")))
  }
  
  # Check factor levels
  factor_cols <- names(train_data)[sapply(train_data, is.factor)]
  for(col in factor_cols) {
    if(col %in% test_cols) {
      train_levels <- levels(train_data[[col]])
      test_levels <- levels(test_data[[col]])
      
      if(!identical(train_levels, test_levels)) {
        validation_results$is_valid <- FALSE
        validation_results$messages <- c(validation_results$messages,
          sprintf("Factor level mismatch in column '%s'", col))
      }
    }
  }
  
  # Check data types
  for(col in intersect(train_cols, test_cols)) {
    if(!identical(class(train_data[[col]]), class(test_data[[col]]))) {
      validation_results$is_valid <- FALSE
      validation_results$messages <- c(validation_results$messages,
        sprintf("Type mismatch in column '%s': train=%s, test=%s",
                col, class(train_data[[col]]), class(test_data[[col]])))
    }
  }
  
  return(validation_results)
}