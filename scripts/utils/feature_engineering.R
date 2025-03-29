#' Convert employment years to numeric
#' @param x Employment years value
#' @return Numeric value
convert_employment_years <- function(x) {
  if (is.numeric(x)) return(x)
  # Remove any non-numeric characters except decimal points
  numeric_val <- as.numeric(gsub("[^0-9.]", "", as.character(x)))
  # Replace NA or 0 with median value (will be calculated later)
  return(numeric_val)
}

#' Feature Engineering Function
#' @param data Input dataframe
#' @param is_training Logical indicating if this is training data
#' @return Dataframe with engineered features
engineer_features <- function(data, is_training = TRUE) {
  # Input validation
  if (!is.data.frame(data)) {
    stop("Input must be a dataframe")
  }
  
  required_cols <- c("age", "employment", "loan_amount", "duration")
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }
  
  # Create copy to avoid modifying original
  df <- data.frame(data, stringsAsFactors = TRUE)
  
  # Handle employment years conversion
  df$employment_years <- convert_employment_years(df$employment)
  
  # Replace NA in employment_years with median (only in training)
  if (is_training) {
    employment_median <- median(df$employment_years, na.rm = TRUE)
    saveRDS(employment_median, "results/preprocessing/employment_median.rds")
  } else {
    employment_median <- readRDS("results/preprocessing/employment_median.rds")
  }
  df$employment_years[is.na(df$employment_years)] <- employment_median
  
  # Calculate age-employment ratio with validation
  df$age_employment_ratio <- ifelse(
    df$employment_years > 0,
    df$age / df$employment_years,
    df$age
  )
  
  # Calculate monthly payment with validation
  df$monthly_payment <- ifelse(
    df$duration > 0,
    df$loan_amount / df$duration,
    df$loan_amount
  )
  
  # Store feature engineering parameters if training
  if (is_training) {
    feature_params <- list(
      mean_age_employment = mean(df$age_employment_ratio, na.rm = TRUE),
      mean_monthly_payment = mean(df$monthly_payment, na.rm = TRUE),
      employment_median = employment_median
    )
    saveRDS(feature_params, "results/preprocessing/feature_params.rds")
  }
  
  return(df)
}

#' Preprocess Data Function
#' @param data Input dataframe
#' @param is_training Logical indicating if this is training data
#' @return List containing processed data and preprocessing objects
preprocess_data <- function(data, is_training = TRUE) {
  # Engineer features
  df <- engineer_features(data, is_training)
  
  # Identify numeric and categorical columns
  num_features <- sapply(df, function(x) is.numeric(x) && !is.factor(x))
  cat_features <- sapply(df, function(x) is.factor(x) || is.character(x))
  
  if (is_training) {
    # Create preprocessing objects
    # Calculate numeric column statistics
    preprocess_params <- list(
      num_means = colMeans(df[, num_features, drop = FALSE], na.rm = TRUE),
      num_sds = apply(df[, num_features, drop = FALSE], 2, sd, na.rm = TRUE),
      cat_levels = lapply(df[, cat_features, drop = FALSE], unique)
    )
    
    # Save column types and preprocessing parameters
    feature_types <- list(
      numeric = names(num_features)[num_features],
      categorical = names(cat_features)[cat_features]
    )
    
    saveRDS(feature_types, "results/preprocessing/feature_types.rds")
    saveRDS(preprocess_params, "results/preprocessing/preprocess_params.rds")
  } else {
    # Load preprocessing parameters
    preprocess_params <- readRDS("results/preprocessing/preprocess_params.rds")
    feature_types <- readRDS("results/preprocessing/feature_types.rds")
  }
  
  # Standardize numeric features
  for (col in feature_types$numeric) {
    if (col %in% names(df)) {
      df[[col]] <- scale(df[[col]], 
                        center = preprocess_params$num_means[col],
                        scale = preprocess_params$num_sds[col])
    }
  }
  
  # Convert categorical features to factors with consistent levels
  for (col in feature_types$categorical) {
    if (col %in% names(df)) {
      df[[col]] <- factor(df[[col]], 
                         levels = preprocess_params$cat_levels[[col]])
    }
  }
  
  # Handle any remaining NA values
  df[is.na(df)] <- 0
  
  return(list(
    data = df,
    params = preprocess_params,
    feature_types = feature_types
  ))
} 