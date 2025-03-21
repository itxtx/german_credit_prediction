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
preprocess_german_credit <- function(data, class_to_binary = TRUE, balance_classes_method = "both", 
                                     train_test_split = 0.7, seed = 123) {
  # Set seed for reproducibility
  set.seed(seed)
  
  # Make a copy of the data to avoid modifying the original
  processed_data <- data
  
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
    # Check current class values
    current_values <- levels(processed_data$class)
    message("Current class levels: ", paste(current_values, collapse = ", "))
    
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
  }
  
  # 3. Check for missing values
  processed_data <- handle_missing_values(processed_data)
  
  # 4. Create train/test split
  split <- create_train_test_split(processed_data, "class", p = train_test_split, seed = seed)
  train_data <- split$train
  test_data <- split$test
  
  # 5. Balance classes in training data only
  if(!is.null(balance_classes_method)) {
    train_data <- balance_classes(train_data, "class", method = balance_classes_method, seed = seed)
  }
  
  # 6. Check for near-zero variance predictors
  nzv <- check_near_zero_variance(train_data)
  if(length(nzv) > 0) {
    message("Removing near-zero variance predictors...")
    train_data <- train_data[, !names(train_data) %in% nzv]
    test_data <- test_data[, !names(test_data) %in% nzv]
  }
  
  # 7. Scale numeric features
  scaled_data <- scale_features(train_data, test_data)
  train_data <- scaled_data$train
  test_data <- scaled_data$test
  
  # 8. Ensure test set has same levels as train set for all factors
  factor_cols <- names(train_data)[sapply(train_data, is.factor)]
  for(col in factor_cols) {
    # Skip if column doesn't exist in test data
    if(!(col %in% names(test_data))) next
    
    # Get all levels from both datasets
    all_levels <- unique(c(levels(train_data[[col]]), levels(test_data[[col]])))
    
    # Set the levels for both datasets
    levels(train_data[[col]]) <- all_levels
    levels(test_data[[col]]) <- all_levels
  }
  
  message("Preprocessing complete!")
  
  return(list(
    train = train_data,
    test = test_data,
    preprocessing_steps = list(
      nzv_removed = nzv,
      scaling_params = scaled_data$params
    )
  ))
}

# If this script is run directly, demonstrate functionality
if(!exists("PREPROCESSING_SOURCED") || !PREPROCESSING_SOURCED) {
  message("This script contains preprocessing functions for the German Credit Analysis project.")
  message("Source this script in your main analysis code to use these functions.")
  PREPROCESSING_SOURCED <- TRUE
}