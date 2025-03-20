# German Credit Dataset Analysis to Classify Loan Applications
# This script provides a comprehensive analysis of the German Credit dataset
# using various classification techniques

# ======= 0. Setup and Debugging =======

# Turn on debugging mode to help identify issues
DEBUG_MODE <- TRUE

# Debug printing function
debug_print <- function(...) {
  if(DEBUG_MODE) {
    cat("[DEBUG] ", ..., "\n")
  }
}

# Print script information and R version
debug_print("Starting German Credit Analysis script")
debug_print("R version:", R.version.string)
debug_print("Working directory:", getwd())

# Install required packages if not already installed
required_packages <- c("readr", "dplyr", "tidyr", "ggplot2", "caret", "rpart", "rpart.plot", 
                       "randomForest", "neuralnet", "ROCR", "pROC", "e1071", 
                       "kernlab", "MASS", "recipes")

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

# Print all loaded packages to help with debugging
if(DEBUG_MODE) {
  cat("Loaded packages:\n")
  print((.packages()))
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
debug_print("Checking for XGBoost package")
if(!requireNamespace("xgboost", quietly = TRUE)) {
  cat("XGBoost package not found. Attempting to install xgboost...\n")
  tryCatch({
    install.packages("xgboost", repos="https://cloud.r-project.org")
    library(xgboost)
    cat("XGBoost successfully installed and loaded.\n")
    xgboost_available <- TRUE
  }, error = function(e) {
    cat("ERROR: Failed to install xgboost package:", e$message, "\n")
    cat("Will use fallback methods instead.\n")
    xgboost_available <- FALSE
  })
} else {
  # Try to load xgboost
  xgboost_available <- tryCatch({
    library(xgboost)
    cat("XGBoost package loaded successfully.\n")
    TRUE
  }, error = function(e) {
    cat("NOTE: Could not load XGBoost package:", e$message, "\n")
    FALSE
  })
}

# Set seed for reproducibility
set.seed(123)

# Run the check for old models
check_for_old_models()

# ======= 1. Importing and Understanding the Dataset =======
debug_print("Starting data import")

# Download German Credit Dataset directly
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
cat("Downloading German Credit Dataset from UCI repository...\n")

# Create a temporary file to download the data
temp_file <- tempfile()
download.file(url, temp_file, quiet = TRUE)

# Define column names according to UCI documentation
column_names <- c("checking_status", "duration", "credit_history", "purpose", "credit_amount", 
                 "savings_status", "employment", "installment_commitment", "personal_status", 
                 "other_parties", "residence_since", "property_magnitude", "age", 
                 "other_payment_plans", "housing", "existing_credits", "job", 
                 "num_dependents", "own_telephone", "foreign_worker", "class")

# Read the data with the correct column names
german_credit <- read.table(temp_file, sep = " ", header = FALSE)
colnames(german_credit) <- column_names
cat("Dataset successfully loaded with", nrow(german_credit), "rows and", ncol(german_credit), "columns.\n")

# Check for any missing columns or extra columns
cat("Column names in the dataset:\n")
print(colnames(german_credit))

# Display basic dataset information
str(german_credit)
summary(german_credit)

# ======= 2. Understanding Unbalanced Data and Converting Class into Factors =======
debug_print("Processing class variable")

# Check class distribution
table(german_credit$class)
class_distribution <- prop.table(table(german_credit$class)) * 100
print(paste("Percentage of class 1 (Good Credit):", round(class_distribution["1"], 2), "%"))
print(paste("Percentage of class 2 (Bad Credit):", round(class_distribution["2"], 2), "%"))

# Convert categorical variables to factors
categorical_cols <- c("checking_status", "credit_history", "purpose", "savings_status", 
                      "employment", "personal_status", "other_parties", "property_magnitude", 
                      "other_payment_plans", "housing", "existing_credits", "job", 
                      "num_dependents", "own_telephone", "foreign_worker", "class")

german_credit[categorical_cols] <- lapply(german_credit[categorical_cols], as.factor)

# Summary of the data before conversion
cat("Class values before conversion:\n")
print(table(german_credit$class))

# Convert class to a binary factor with valid R variable names
# In the UCI German credit dataset: 1 = good, 2 = bad
current_values <- levels(german_credit$class)
cat("Current class levels:", paste(current_values, collapse=", "), "\n")

# Handle different possible formats
if(length(current_values) == 2) {
  if(all(current_values %in% c("1", "2"))) {
    german_credit$class <- factor(ifelse(german_credit$class == "1", "Good", "Bad"))
    cat("Converted numeric classes (1/2) to Good/Bad\n")
  } else if(all(current_values %in% c("A201", "A202"))) {
    german_credit$class <- factor(ifelse(german_credit$class == "A201", "Good", "Bad"))
    cat("Converted text classes (A201/A202) to Good/Bad\n")
  } else {
    # Try to infer which is which based on proportions (assuming good credit is more common)
    major_class <- names(which.max(table(german_credit$class)))
    german_credit$class <- factor(ifelse(german_credit$class == major_class, "Good", "Bad"))
    cat("Inferred class conversion based on frequencies\n")
  }
} else {
  cat("WARNING: Unexpected number of class levels. Check the class variable.\n")
}

# Verify the conversion worked
cat("\nClass distribution after conversion:\n")
print(table(german_credit$class))

# Visualize the original class distribution
original_plot <- ggplot(data = german_credit, aes(x = class, fill = class)) +
  geom_bar() +
  scale_fill_manual(values = c("red", "green")) +
  labs(title = "Original Class Distribution",
       x = "Credit Risk (Bad vs Good)",
       y = "Count") +
  theme_minimal()

print(original_plot)

# ======= 3. Dividing the Dataset into Equal Parts with Equal Distribution =======
debug_print("Splitting data into train/test sets")
cat("\n=== STEP 3: Dividing the Dataset into Train/Test Sets ===\n")

# Stratified sampling to maintain class distribution
set.seed(123)
train_index <- createDataPartition(german_credit$class, p = 0.7, list = FALSE)
train_data <- german_credit[train_index, ]
test_data <- german_credit[-train_index, ]

# Check if class variable is properly preserved in train and test sets
cat("Train data class levels:", paste(levels(train_data$class), collapse=", "), "\n")
cat("Test data class levels:", paste(levels(test_data$class), collapse=", "), "\n")

# Verify class distribution in training and testing sets
train_dist <- prop.table(table(train_data$class)) * 100
test_dist <- prop.table(table(test_data$class)) * 100

cat("Class distribution in training set (%):\n")
print(train_dist)
cat("Class distribution in testing set (%):\n")
print(test_dist)

# ======= 3b. Handling Class Imbalance with Over/Under Sampling =======
debug_print("Handling class imbalance")
cat("\n=== STEP 3b: Handling Class Imbalance ===\n")

# Check class distribution before balancing
cat("Class levels before balancing:", paste(levels(train_data$class), collapse=", "), "\n")
cat("Class counts before balancing:\n")
print(table(train_data$class))

# Apply balancing technique with error handling
set.seed(123)
tryCatch({
  cat("Attempting to balance class distribution...\n")
  
  # Method 1: Simple random over/under sampling (fallback approach)
  # Count the classes
  class_counts <- table(train_data$class)
  majority_class <- names(class_counts)[which.max(class_counts)]
  minority_class <- names(class_counts)[which.min(class_counts)]
  
  # Target count - we'll use a balanced dataset
  target_count <- min(max(class_counts) / 2, min(class_counts) * 3)
  cat("Target count per class:", target_count, "\n")
  
  # Separate majority and minority classes
  majority_data <- train_data[train_data$class == majority_class, ]
  minority_data <- train_data[train_data$class == minority_class, ]
  
  # Undersample majority class
  majority_sample <- majority_data[sample(1:nrow(majority_data), target_count), ]
  
  # Oversample minority class (with replacement)
  minority_sample <- minority_data[sample(1:nrow(minority_data), target_count, replace=TRUE), ]
  
  # Combine the balanced samples
  balanced_train <- rbind(majority_sample, minority_sample)
  
  # Shuffle the rows to mix the classes
  balanced_train <- balanced_train[sample(1:nrow(balanced_train)), ]
  
  # Check class distribution after balancing
  cat("Class distribution after balancing:\n")
  print(table(balanced_train$class))
  balanced_train_dist <- prop.table(table(balanced_train$class)) * 100
  cat("Percentage distribution after balancing (%):\n")
  print(balanced_train_dist)
  
  # Use the balanced training data for model training
  train_data <- balanced_train
  
  # Visualize the balanced class distribution
  balanced_plot <- ggplot(data = train_data, aes(x = class, fill = class)) +
    geom_bar() +
    scale_fill_manual(values = c("red", "green")) +
    labs(title = "Class Distribution After Balancing",
         x = "Credit Risk (Bad vs Good)",
         y = "Count") +
    theme_minimal()
  
  print(balanced_plot)
  
}, error = function(e) {
  cat("ERROR balancing classes:", e$message, "\n")
  cat("Continuing with original imbalanced data...\n")
  # Visualize the original distribution
  imbalanced_plot <- ggplot(data = train_data, aes(x = class, fill = class)) +
    geom_bar() +
    scale_fill_manual(values = c("red", "green")) +
    labs(title = "Original Class Distribution (Balancing Failed)",
         x = "Credit Risk (Bad vs Good)",
         y = "Count") +
    theme_minimal()
  
  print(imbalanced_plot)
})

# ======= 4. Imputing for Null Values =======
debug_print("Checking for missing values")

# Check for missing values
missing_values <- colSums(is.na(german_credit))
print("Missing values per column:")
print(missing_values)

# If there are missing values, impute them
if(sum(missing_values) > 0) {
  # For numeric columns: impute with median
  numeric_cols <- sapply(german_credit, is.numeric)
  for(col in names(german_credit)[numeric_cols & missing_values > 0]) {
    german_credit[is.na(german_credit[, col]), col] <- median(german_credit[, col], na.rm = TRUE)
  }
  
  # For categorical columns: impute with mode
  for(col in names(german_credit)[!numeric_cols & missing_values > 0]) {
    mode_val <- names(sort(table(german_credit[, col]), decreasing = TRUE))[1]
    german_credit[is.na(german_credit[, col]), col] <- mode_val
  }
  
  # Verify imputation
  missing_after <- colSums(is.na(german_credit))
  print("Missing values after imputation:")
  print(missing_after)
}

# ======= 4.5 Clean Dataset ========
debug_print("Cleaning dataset before modeling")
cat("\n=== STEP 4.5: Cleaning Dataset Before Modeling ===\n")

# Check for any extra/problematic columns
cat("Column names before cleaning:\n")
print(colnames(train_data))

# Remove any NA columns or other problematic columns
if("NA" %in% colnames(train_data)) {
  cat("Removing 'NA' column from the dataset\n")
  train_data <- train_data[, !colnames(train_data) %in% "NA"]
  test_data <- test_data[, !colnames(test_data) %in% "NA"]
}

# Check for any columns with too many levels (may cause issues in modeling)
factor_cols <- sapply(train_data, is.factor)
for(col in names(train_data)[factor_cols]) {
  num_levels <- length(levels(train_data[[col]]))
  if(num_levels > 20) {
    cat("WARNING: Column", col, "has", num_levels, "levels which may cause issues\n")
  }
}

# Check if any factor has empty levels
for(col in names(train_data)[factor_cols]) {
  if("" %in% levels(train_data[[col]])) {
    cat("WARNING: Column", col, "has empty level\n")
    # Fix by dropping empty levels
    train_data[[col]] <- droplevels(train_data[[col]])
    test_data[[col]] <- droplevels(test_data[[col]])
  }
}

# Verify all columns have valid names
if(any(is.na(names(train_data)) | names(train_data) == "")) {
  cat("WARNING: Some columns have invalid names!\n")
}

# Drop unused factor levels
train_data <- droplevels(train_data)
test_data <- droplevels(test_data)

# Ensure test set has same levels as train set
for(col in names(test_data)[sapply(test_data, is.factor)]) {
  # Get all levels from both datasets
  all_levels <- unique(c(levels(train_data[[col]]), levels(test_data[[col]])))
  
  # Set the levels for both datasets
  levels(train_data[[col]]) <- all_levels
  levels(test_data[[col]]) <- all_levels
}

# Final check
cat("Column names after cleaning:\n")
print(colnames(train_data))
cat("Class distribution after cleaning:\n")
print(table(train_data$class))

# ======= 5. Defining Cross-Validation, Metrics, and Preprocessing =======
debug_print("Setting up cross-validation and preprocessing")
cat("\n=== STEP 5: Setting up Cross-Validation and Preprocessing ===\n")

# Setup cross-validation method
ctrl <- trainControl(
  method = "cv",           # Cross-validation
  number = 5,              # 5-fold
  classProbs = TRUE,       # Calculate class probabilities
  summaryFunction = twoClassSummary,  # Use ROC summary
  savePredictions = TRUE   # Save predictions for later analysis
)

# Check for near-zero variance predictors
tryCatch({
  nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
  cat("Near-zero variance predictors:\n")
  print(nzv[nzv$nzv, ])
  
  # Remove near-zero variance predictors if any
  if(any(nzv$nzv)) {
    cat("Removing", sum(nzv$nzv), "near-zero variance predictors\n")
    train_data <- train_data[, !nzv$nzv]
    test_data <- test_data[, !nzv$nzv]
  }
}, error = function(e) {
  cat("ERROR checking for near-zero variance predictors:", e$message, "\n")
  cat("Skipping this step...\n")
})

# Check for highly correlated numeric predictors
numeric_cols <- sapply(train_data, is.numeric)
if(sum(numeric_cols) > 1) {
  tryCatch({
    correlation_matrix <- cor(train_data[, numeric_cols], use = "pairwise.complete.obs")
    highly_correlated <- findCorrelation(correlation_matrix, cutoff = 0.75)
    if(length(highly_correlated) > 0) {
      cat("Found", length(highly_correlated), "highly correlated predictors\n")
      # Optionally remove highly correlated predictors
      # train_data <- train_data[, -highly_correlated]
      # test_data <- test_data[, -highly_correlated]
    }
  }, error = function(e) {
    cat("ERROR checking for highly correlated predictors:", e$message, "\n")
    cat("Skipping this step...\n")
  })
}

# Define preprocessing steps - only center and scale numeric predictors
preprocess_steps <- c("center", "scale")

# Function to evaluate model performance
evaluate_model <- function(pred, actual, pred_prob = NULL) {
  tryCatch({
    # Ensure both pred and actual have the same levels
    if(!identical(levels(pred), levels(actual))) {
      cat("WARNING: Prediction levels don't match actual levels.\n")
      cat("Pred levels:", paste(levels(pred), collapse=", "), "\n")
      cat("Actual levels:", paste(levels(actual), collapse=", "), "\n")
      
      # Force the same levels
      all_levels <- unique(c(levels(pred), levels(actual)))
      levels(pred) <- all_levels
      levels(actual) <- all_levels
    }
    
    # Confusion Matrix
    conf_matrix <- confusionMatrix(pred, actual)
    
    # Calculate metrics
    accuracy <- conf_matrix$overall["Accuracy"]
    precision <- conf_matrix$byClass["Pos Pred Value"]
    recall <- conf_matrix$byClass["Sensitivity"]
    f1 <- conf_matrix$byClass["F1"]
    
    # AUC-ROC if probabilities are available
    auc_value <- NA
    if(!is.null(pred_prob)) {
      # Convert factor to numeric for prediction object (Good = 1, Bad = 0)
      actual_numeric <- ifelse(actual == "Good", 1, 0)
      
      # Debug
      cat("Probability summary:", summary(pred_prob), "\n")
      
      # Create prediction object - handle errors
      pred_obj <- tryCatch({
        prediction(pred_prob, actual_numeric)
      }, error = function(e) {
        cat("ERROR creating prediction object:", e$message, "\n")
        return(NULL)
      })
      
      if(!is.null(pred_obj)) {
        auc_value <- performance(pred_obj, "auc")@y.values[[1]]
        
        # KS statistic
        ks_perf <- performance(pred_obj, "tpr", "fpr")
        ks_stat <- max(abs(ks_perf@y.values[[1]] - ks_perf@x.values[[1]]))
        
        # Gini coefficient
        gini <- 2 * auc_value - 1
        
        # Plot ROC curve
        roc_perf <- performance(pred_obj, "tpr", "fpr")
        plot(roc_perf, main = "ROC Curve", col = "blue", lwd = 2)
        abline(0, 1, lty = 2, col = "gray")
        text(0.8, 0.2, paste("AUC =", round(auc_value, 3)))
        text(0.8, 0.1, paste("Gini =", round(gini, 3)))
        text(0.8, 0.3, paste("KS =", round(ks_stat, 3)))
        
        return(list(
          accuracy = accuracy,
          precision = precision,
          recall = recall,
          f1 = f1,
          auc = auc_value,
          gini = gini,
          ks = ks_stat,
          confusion_matrix = conf_matrix
        ))
      }
    }
    
    return(list(
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      f1 = f1,
      confusion_matrix = conf_matrix
    ))
  }, error = function(e) {
    cat("ERROR in evaluate_model:", e$message, "\n")
    return(list(
      accuracy = NA,
      precision = NA,
      recall = NA,
      f1 = NA,
      error = e$message
    ))
  })
}

# ======= 6. LOGISTIC REGRESSION and Feature Selection =======
debug_print("Training logistic regression model")
cat("\n=== STEP 6: Training Logistic Regression Model ===\n")

# Debug: Check structure of train_data before model training
cat("Structure of training data (columns):\n")
print(names(train_data))
cat("Class distribution in training data:\n")
print(table(train_data$class))

# Ensure train_data has at least two classes to avoid model issues
if(length(unique(train_data$class)) < 2) {
  stop("Training data must have at least two classes!")
}

# Train logistic regression model with error handling
set.seed(123)
tryCatch({
  cat("Training logistic regression model...\n")
  
  # Only preprocess numeric columns
  numeric_cols <- sapply(train_data, is.numeric)
  cat("Numeric columns:", sum(numeric_cols), "\n")
  
  # Create a simple formula using only reliable predictors
  # This helps avoid issues with complex factor variables
  form <- as.formula("class ~ checking_status + duration + credit_amount + age")
  
  # Train a simpler logistic regression model 
  logistic_model <- train(
    form, 
    data = train_data,
    method = "glm",
    family = "binomial",
    trControl = ctrl,
    preProcess = c("center", "scale"),
    metric = "ROC"
  )
  
  # Print model summary
  cat("Logistic regression model successfully trained\n")
  print(logistic_model)
  
  # If available, print more detailed summary
  if(!is.null(logistic_model$finalModel)) {
    cat("\nDetailed model summary:\n")
    print(summary(logistic_model$finalModel))
  }
  
}, error = function(e) {
  cat("ERROR training logistic regression:", e$message, "\n")
  cat("Trying alternative approach...\n")
  
  # Alternative approach: use glm directly with fewer predictors
  simple_formula <- as.formula("class ~ duration + credit_amount + age + checking_status")
  glm_model <- glm(simple_formula, data = train_data, family = binomial())
  cat("Alternative model summary:\n")
  print(summary(glm_model))
  
  # Create a caret-like model object
  logistic_model <<- list(
    finalModel = glm_model,
    preProcess = NULL,
    modelInfo = list(label = "Generalized Linear Model"),
    metric = "ROC"
  )
  class(logistic_model) <<- "train"
})

# Make predictions on test set
cat("\nMaking predictions with logistic regression...\n")
tryCatch({
  logistic_pred <- predict(logistic_model, newdata = test_data)
  cat("Prediction classes:", paste(unique(logistic_pred), collapse=", "), "\n")
  
  # Get prediction probabilities with error handling
  pred_probs <- predict(logistic_model, newdata = test_data, type = "prob")
  cat("Probability columns available:", paste(colnames(pred_probs), collapse=", "), "\n")
  
  # Dynamically select the "Good" class probability
  if("Good" %in% colnames(pred_probs)) {
    logistic_prob <- pred_probs[, "Good"]
  } else if("1" %in% colnames(pred_probs)) {
    logistic_prob <- pred_probs[, "1"]
  } else {
    # If neither expected column exists, use the second column (assuming binary classification)
    logistic_prob <- pred_probs[, 2]
    cat("Using column", colnames(pred_probs)[2], "for probability\n")
  }
  
  # Evaluate the model
  cat("\n--- Logistic Regression Performance ---\n")
  logistic_perf <- evaluate_model(logistic_pred, test_data$class, logistic_prob)
  print(logistic_perf)
  
}, error = function(e) {
  cat("ERROR making predictions:", e$message, "\n")
  cat("Creating dummy performance metrics for continuity\n")
  
  # Create dummy performance metrics to allow script to continue
  logistic_perf <<- list(
    accuracy = NA,
    precision = NA,
    recall = NA,
    f1 = NA,
    auc = NA,
    gini = NA,
    ks = NA,
    confusion_matrix = NA
  )
})

# ======= 7. Applying Bayesian Model & Recursive Partitioning =======
debug_print("Training Naive Bayes model")

# Naive Bayes model
set.seed(123)
tryCatch({
  nb_model <- train(
    class ~ .,
    data = train_data,
    method = "naive_bayes",
    trControl = ctrl,
    preProcess = preprocess_steps,
    metric = "ROC"
  )
  
  print(nb_model)
  
  # Make predictions with Naive Bayes
  cat("\nMaking predictions with Naive Bayes...\n")
  nb_pred <- predict(nb_model, newdata = test_data)
  cat("Naive Bayes prediction classes:", paste(unique(nb_pred), collapse=", "), "\n")
  
  # Get prediction probabilities with error handling
  nb_probs <- predict(nb_model, newdata = test_data, type = "prob")
  cat("Probability columns available:", paste(colnames(nb_probs), collapse=", "), "\n")
  
  # Dynamically select the "Good" class probability
  if("Good" %in% colnames(nb_probs)) {
    nb_prob <- nb_probs[, "Good"]
  } else if("1" %in% colnames(nb_probs)) {
    nb_prob <- nb_probs[, "1"]
  } else {
    # If neither expected column exists, use the second column (assuming binary classification)
    nb_prob <- nb_probs[, 2]
    cat("Using column", colnames(nb_probs)[2], "for probability\n")
  }
  
  # Evaluate Naive Bayes
  cat("\n--- Naive Bayes Performance ---\n")
  nb_perf <- evaluate_model(nb_pred, test_data$class, nb_prob)
  print(nb_perf)
}, error = function(e) {
  cat("ERROR in Naive Bayes:", e$message, "\n")
  nb_perf <<- list(
    accuracy = NA,
    precision = NA,
    recall = NA,
    f1 = NA,
    auc = NA,
    gini = NA,
    ks = NA,
    confusion_matrix = NA
  )
})

# Decision Tree (Recursive Partitioning)
debug_print("Training decision tree model")
set.seed(123)
tryCatch({
  tree_model <- train(
    class ~ .,
    data = train_data,
    method = "rpart",
    trControl = ctrl,
    tuneLength = 10,
    metric = "ROC"
  )
  
  print(tree_model)
  print(tree_model$bestTune)
  
  # Plot the decision tree
  rpart.plot(tree_model$finalModel, box.palette = "RdBu", shadow.col = "gray", nn = TRUE)
  
  # Make predictions with Decision Tree
  cat("\nMaking predictions with Decision Tree...\n")
  tree_pred <- predict(tree_model, newdata = test_data)
  cat("Decision Tree prediction classes:", paste(unique(tree_pred), collapse=", "), "\n")
  
  # Get prediction probabilities with error handling
  tree_probs <- predict(tree_model, newdata = test_data, type = "prob")
  cat("Probability columns available:", paste(colnames(tree_probs), collapse=", "), "\n")
  
  # Dynamically select the "Good" class probability
  if("Good" %in% colnames(tree_probs)) {
    tree_prob <- tree_probs[, "Good"]
  } else if("1" %in% colnames(tree_probs)) {
    tree_prob <- tree_probs[, "1"]
  } else {
    # If neither expected column exists, use the second column (assuming binary classification)
    tree_prob <- tree_probs[, 2]
    cat("Using column", colnames(tree_probs)[2], "for probability\n")
  }
  
  # Evaluate Decision Tree
  cat("\n--- Decision Tree Performance ---\n")
  tree_perf <- evaluate_model(tree_pred, test_data$class, tree_prob)
  print(tree_perf)
}, error = function(e) {
  cat("ERROR in Decision Tree:", e$message, "\n")
  tree_perf <<- list(
    accuracy = NA,
    precision = NA,
    recall = NA,
    f1 = NA,
    auc = NA,
    gini = NA,
    ks = NA,
    confusion_matrix = NA
  )
})

# ======= 8. Improve Results using Random Forest =======
debug_print("Training Random Forest model")

# Train Random Forest model
set.seed(123)
tryCatch({
  rf_model <- train(
    class ~ .,
    data = train_data,
    method = "rf",
    trControl = ctrl,
    tuneLength = 5,
    metric = "ROC",
    importance = TRUE
  )
  
  print(rf_model)
  print(rf_model$bestTune)
  
  # Feature importance from Random Forest
  rf_importance <- varImp(rf_model)
  plot(rf_importance, top = 20, main = "Random Forest - Variable Importance")
  
  # Make predictions with Random Forest
  cat("\nMaking predictions with Random Forest...\n")
  rf_pred <- predict(rf_model, newdata = test_data)
  cat("Random Forest prediction classes:", paste(unique(rf_pred), collapse=", "), "\n")
  
  # Get prediction probabilities with error handling
  rf_probs <- predict(rf_model, newdata = test_data, type = "prob")
  cat("Probability columns available:", paste(colnames(rf_probs), collapse=", "), "\n")
  
  # Dynamically select the "Good" class probability
  if("Good" %in% colnames(rf_probs)) {
    rf_prob <- rf_probs[, "Good"]
  } else if("1" %in% colnames(rf_probs)) {
    rf_prob <- rf_probs[, "1"]
  } else {
    # If neither expected column exists, use the second column (assuming binary classification)
    rf_prob <- rf_probs[, 2]
    cat("Using column", colnames(rf_probs)[2], "for probability\n")
  }
  
  # Evaluate Random Forest
  cat("\n--- Random Forest Performance ---\n")
  rf_perf <- evaluate_model(rf_pred, test_data$class, rf_prob)
  print(rf_perf)
}, error = function(e) {
  cat("ERROR in Random Forest:", e$message, "\n")
  rf_perf <<- list(
    accuracy = NA,
    precision = NA,
    recall = NA,
    f1 = NA,
    auc = NA,
    gini = NA,
    ks = NA,
    confusion_matrix = NA
  )
})

# ======= 9. Implementing XGBoost =======
cat("\n=== STEP 9: Training XGBoost Model ===\n")

# Check again for any old models before proceeding
check_for_old_models()

# Only proceed with XGBoost if available
if(exists("xgboost_available") && xgboost_available) {
  debug_print("Starting XGBoost model training")
  set.seed(123)
  tryCatch({
    # Prepare data for XGBoost
    debug_print("Preparing data for XGBoost")
    xgb_train <- train_data
    xgb_test <- test_data
    
    # Convert class to numeric (0/1) for better XGBoost performance
    xgb_train$class_numeric <- ifelse(xgb_train$class == "Good", 1, 0)
    xgb_test$class_numeric <- ifelse(xgb_test$class == "Good", 1, 0)
    
    # Define tuning grid for XGBoost - simplified for faster execution
    xgb_grid <- expand.grid(
      nrounds = c(50, 100),
      max_depth = c(3, 6),
      eta = c(0.1, 0.3),
      gamma = 0,
      colsample_bytree = 1,
      min_child_weight = 1,
      subsample = 1
    )
    
    # Train XGBoost model
    debug_print("Training XGBoost model with caret")
    xgb_model <- train(
      class ~ .,
      data = xgb_train %>% select(-class_numeric),  # Remove the numeric class column
      method = "xgbTree",
      trControl = trainControl(
        method = "cv",
        number = 5,
        classProbs = TRUE,
        savePredictions = TRUE
      ),
      tuneGrid = xgb_grid,
      metric = "ROC",
      verbose = FALSE
    )
    
    debug_print("XGBoost model training complete")
    print(xgb_model)
    print(xgb_model$bestTune)
    
    # Make predictions with XGBoost
    debug_print("Making predictions with XGBoost")
    xgb_pred <- predict(xgb_model, newdata = xgb_test)
    cat("XGBoost prediction classes:", paste(unique(xgb_pred), collapse=", "), "\n")
    
    # Get prediction probabilities
    debug_print("Getting prediction probabilities")
    xgb_probs <- predict(xgb_model, newdata = xgb_test, type = "prob")
    cat("Probability columns available:", paste(colnames(xgb_probs), collapse=", "), "\n")
    
    # Use Good class probability
    if ("Good" %in% colnames(xgb_probs)) {
      debug_print("Using 'Good' column for probabilities")
      xgb_prob <- xgb_probs[, "Good"]
    } else {
      # Fallback to second column
      debug_print("Using second column for probabilities")
      xgb_prob <- xgb_probs[, 2]
      cat("Using column", colnames(xgb_probs)[2], "for probability\n")
    }
    
    # Evaluate XGBoost
    debug_print("Evaluating XGBoost model")
    cat("\n--- XGBoost Performance ---\n")
    xgb_perf <- evaluate_model(xgb_pred, test_data$class, xgb_prob)
    print(xgb_perf)
    
  }, error = function(e) {
    cat("ERROR in XGBoost:", e$message, "\n")
    cat("Setting dummy performance metrics\n")
    xgb_perf <<- list(
      accuracy = NA,
      precision = NA,
      recall = NA,
      f1 = NA,
      auc = NA,
      gini = NA,
      ks = NA,
      confusion_matrix = NA
    )
  })
} else {
  cat("XGBoost not available. Skipping XGBoost model.\n")
  # Create dummy performance metrics for consistency
  xgb_perf <- list(
    accuracy = NA,
    precision = NA,
    recall = NA,
    f1 = NA,
    auc = NA,
    gini = NA,
    ks = NA,
    confusion_matrix = NA
  )
}

# ======= 10. Model Improvement with Gaussian RBF Kernel (SVM) =======
debug_print("Training SVM model")

# Train SVM with Radial Basis Function (RBF) kernel
set.seed(123)
tryCatch({
  svm_model <- train(
    class ~ .,
    data = train_data,
    method = "svmRadial",
    trControl = ctrl,
    preProcess = preprocess_steps,
    tuneLength = 5,
    metric = "ROC"
  )
  
  print(svm_model)
  print(svm_model$bestTune)
  
  # Make predictions with SVM
  cat("\nMaking predictions with SVM...\n")
  svm_pred <- predict(svm_model, newdata = test_data)
  cat("SVM prediction classes:", paste(unique(svm_pred), collapse=", "), "\n")
  
  # Get prediction probabilities with error handling
  svm_probs <- predict(svm_model, newdata = test_data, type = "prob")
  cat("Probability columns available:", paste(colnames(svm_probs), collapse=", "), "\n")
  
  # Dynamically select the "Good" class probability
  if("Good" %in% colnames(svm_probs)) {
    svm_prob <- svm_probs[, "Good"]
  } else if("1" %in% colnames(svm_probs)) {
    svm_prob <- svm_probs[, "1"]
  } else {
    # If neither expected column exists, use the second column (assuming binary classification)
    svm_prob <- svm_probs[, 2]
    cat("Using column", colnames(svm_probs)[2], "for probability\n")
  }
  
  # Evaluate SVM
  cat("\n--- SVM with RBF Kernel Performance ---\n")
  svm_perf <- evaluate_model(svm_pred, test_data$class, svm_prob)
  print(svm_perf)
}, error = function(e) {
  cat("ERROR in SVM:", e$message, "\n")
  svm_perf <<- list(
    accuracy = NA,
    precision = NA,
    recall = NA,
    f1 = NA,
    auc = NA,
    gini = NA,
    ks = NA,
    confusion_matrix = NA
  )
})

# ======= 11. Implementing Neural Network =======
debug_print("Training neural network model")
cat("\n=== STEP 11: Training Neural Network Model ===\n")

# Neural Network - simplified implementation
tryCatch({
  cat("Preparing data for neural network...\n")
  
  # Create a simpler approach for neural network
  # Select only numeric columns to avoid issues with factors
  numeric_cols <- sapply(train_data, is.numeric)
  cat("Number of numeric columns for neural network:", sum(numeric_cols), "\n")
  
  if(sum(numeric_cols) >= 3) {  # Ensure we have at least a few numeric features
    # Create simplified datasets with only numeric features
    train_data_nn <- train_data[, numeric_cols]
    test_data_nn <- test_data[, numeric_cols]
    
    # Add the class column back
    train_data_nn$class <- train_data$class
    test_data_nn$class <- test_data$class
    
    # Convert class to 0/1 for neural network
    train_target <- ifelse(train_data$class == "Good", 1, 0)
    test_target <- ifelse(test_data$class == "Good", 1, 0)
    
    # Scale numeric features
    train_data_nn[, numeric_cols] <- scale(train_data_nn[, numeric_cols])
    
    # Apply same scaling to test data
    for(col in names(train_data_nn)[numeric_cols]) {
      mean_val <- mean(train_data[, col])
      sd_val <- sd(train_data[, col])
      test_data_nn[, col] <- (test_data[, col] - mean_val) / sd_val
    }
    
    # Train a simple neural network
    set.seed(123)
    nn_model <- train(
      x = train_data_nn[, numeric_cols],
      y = factor(train_target, levels = c(0, 1), labels = c("Bad", "Good")),
      method = "nnet",
      trControl = ctrl,
      tuneLength = 3,
      metric = "ROC",
      trace = FALSE,
      maxit = 200,
      MaxNWts = 1000
    )
    
    cat("Neural network model successfully trained\n")
    print(nn_model)
    
    # Make predictions
    nn_pred <- predict(nn_model, newdata = test_data_nn[, numeric_cols])
    
    # Get probabilities
    nn_probs <- predict(nn_model, newdata = test_data_nn[, numeric_cols], type = "prob")
    
    # Use appropriate probability column
    if("Good" %in% colnames(nn_probs)) {
      nn_prob <- nn_probs[, "Good"]
    } else {
      # Use the second column (class 1) if available
      if(ncol(nn_probs) >= 2) {
        nn_prob <- nn_probs[, 2]
      } else {
        nn_prob <- rep(0.5, nrow(test_data_nn))  # Fallback
      }
    }
    
    # Evaluate Neural Network
    cat("\n--- Neural Network Performance ---\n")
    test_class_factor <- factor(ifelse(test_target == 1, "Good", "Bad"), 
                              levels = levels(nn_pred))
    nn_perf <- evaluate_model(nn_pred, test_class_factor, nn_prob)
    print(nn_perf)
  } else {
    cat("Not enough numeric features for neural network. Skipping...\n")
    nn_perf <- list(
      accuracy = NA,
      precision = NA,
      recall = NA,
      f1 = NA,
      auc = NA,
      confusion_matrix = NA
    )
  }
}, error = function(e) {
  cat("ERROR in neural network training:", e$message, "\n")
  # Create dummy metrics
  nn_perf <<- list(
    accuracy = NA,
    precision = NA,
    recall = NA,
    f1 = NA,
    auc = NA,
    confusion_matrix = NA
  )
})

# ======= 12. Performance Comparison and Visualization =======
debug_print("Comparing model performance")
cat("\n=== STEP 12: Model Performance Comparison ===\n")

# Function to safely extract metrics
safe_extract <- function(obj, metric, default = NA) {
  if(!exists(obj, inherits = FALSE) || is.null(get0(obj))) {
    return(default)
  }
  perf_obj <- get0(obj)
  if(!is.list(perf_obj) || is.null(perf_obj[[metric]])) {
    return(default)
  }
  return(perf_obj[[metric]])
}

# Collect all model performances with error handling
models <- c("Logistic Regression", "Naive Bayes", "Decision Tree", 
           "Random Forest", "XGBoost", "SVM (RBF)", "Neural Network")

# Metrics to compare
accuracy <- c(
  safe_extract("logistic_perf", "accuracy"),
  safe_extract("nb_perf", "accuracy"),
  safe_extract("tree_perf", "accuracy"),
  safe_extract("rf_perf", "accuracy"),
  safe_extract("xgb_perf", "accuracy"),
  safe_extract("svm_perf", "accuracy"),
  safe_extract("nn_perf", "accuracy")
)

precision <- c(
  safe_extract("logistic_perf", "precision"),
  safe_extract("nb_perf", "precision"),
  safe_extract("tree_perf", "precision"),
  safe_extract("rf_perf", "precision"),
  safe_extract("xgb_perf", "precision"),
  safe_extract("svm_perf", "precision"),
  safe_extract("nn_perf", "precision")
)

recall <- c(
  safe_extract("logistic_perf", "recall"),
  safe_extract("nb_perf", "recall"),
  safe_extract("tree_perf", "recall"),
  safe_extract("rf_perf", "recall"),
  safe_extract("xgb_perf", "recall"),
  safe_extract("svm_perf", "recall"),
  safe_extract("nn_perf", "recall")
)

f1 <- c(
  safe_extract("logistic_perf", "f1"),
  safe_extract("nb_perf", "f1"),
  safe_extract("tree_perf", "f1"),
  safe_extract("rf_perf", "f1"),
  safe_extract("xgb_perf", "f1"),
  safe_extract("svm_perf", "f1"),
  safe_extract("nn_perf", "f1")
)

auc <- c(
  safe_extract("logistic_perf", "auc"),
  safe_extract("nb_perf", "auc"),
  safe_extract("tree_perf", "auc"),
  safe_extract("rf_perf", "auc"),
  safe_extract("xgb_perf", "auc"),
  safe_extract("svm_perf", "auc"),
  safe_extract("nn_perf", "auc")
)

# Create comparison dataframe
comparison <- data.frame(
  Model = models,
  Accuracy = accuracy,
  Precision = precision,
  Recall = recall,
  F1_Score = f1,
  AUC = auc
)

# Only keep models with valid metrics
valid_rows <- !is.na(comparison$Accuracy) | !is.na(comparison$AUC)
if(sum(valid_rows) > 0) {
  comparison <- comparison[valid_rows, ]
  
  # Sort by AUC (best model on top), handling NAs
  comparison <- comparison[order(comparison$AUC, decreasing = TRUE, na.last = TRUE), ]
  
  # Print comparison table
  cat("Model Performance Comparison:\n")
  print(comparison)
  
  # Visualize model comparison if we have at least one valid model
  if(nrow(comparison) > 0) {
    # Visualize model comparison
    comparison_long <- tidyr::pivot_longer(
      comparison, 
      cols = c("Accuracy", "Precision", "Recall", "F1_Score", "AUC"),
      names_to = "Metric",
      values_to = "Value"
    )
    
    ggplot(comparison_long, aes(x = Model, y = Value, fill = Metric)) +
      geom_bar(stat = "identity", position = "dodge") +
      labs(title = "Model Performance Comparison",
           x = "Model",
           y = "Score") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    # Plot AUC comparison if we have AUC values
    if(any(!is.na(comparison$AUC))) {
      auc_comparison <- comparison[!is.na(comparison$AUC), ]
      ggplot(auc_comparison, aes(x = reorder(Model, AUC), y = AUC)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        geom_text(aes(label = round(AUC, 3)), vjust = -0.3) +
        labs(title = "Model Comparison by AUC",
             x = "Model",
             y = "AUC") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        ylim(0, max(auc_comparison$AUC, na.rm = TRUE) * 1.1)
    }
  } else {
    cat("No valid model metrics available for visualization\n")
  }
} else {
  cat("No valid model metrics available for comparison\n")
}

# ======= 13. ROC Curves for All Models =======
debug_print("Generating ROC curves")
cat("\n=== STEP 13: Generating ROC Curves ===\n")

# Function to safely check if an object exists
exists_and_not_null <- function(obj_name) {
  exists(obj_name) && !is.null(get0(obj_name))
}

# Create a ROC plot for available models
tryCatch({
  # Create an empty list for probability values
  prob_values <- list()
  
  # Add probability values for models that exist
  if(exists_and_not_null("logistic_prob")) prob_values$logistic <- logistic_prob
  if(exists_and_not_null("nb_prob")) prob_values$naive_bayes <- nb_prob
  if(exists_and_not_null("tree_prob")) prob_values$decision_tree <- tree_prob
  if(exists_and_not_null("rf_prob")) prob_values$random_forest <- rf_prob
  if(exists_and_not_null("xgb_prob")) prob_values$xgboost <- xgb_prob
  if(exists_and_not_null("svm_prob")) prob_values$svm <- svm_prob
  if(exists_and_not_null("nn_prob")) prob_values$neural_net <- nn_prob
  
  # Check if we have at least one model to plot
  if(length(prob_values) > 0) {
    # Convert actual class to numeric
    actual_values <- ifelse(test_data$class == "Good", 1, 0)
    
    # Create data frame with actual values and model probabilities
    roc_data <- data.frame(actual = actual_values)
    
    # Add each model's probabilities to the data frame
    for(model_name in names(prob_values)) {
      roc_data[[model_name]] <- prob_values[[model_name]]
    }
    
    # Create empty plot
    plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), 
         xlab = "False Positive Rate", ylab = "True Positive Rate",
         main = "ROC Curves Comparison")
    abline(0, 1, lty = 2, col = "gray")
    
    # Add ROC curves for each model with different colors
    colors <- rainbow(length(prob_values))
    auc_values <- numeric(length(prob_values))
    
    for(i in 1:length(prob_values)) {
      model_name <- names(prob_values)[i]
      
      # Try to create and plot ROC curve
      tryCatch({
        pred_obj <- prediction(roc_data[, model_name+1], roc_data$actual)
        roc_perf <- performance(pred_obj, "tpr", "fpr")
        auc_value <- performance(pred_obj, "auc")@y.values[[1]]
        auc_values[i] <- auc_value
        
        lines(roc_perf@x.values[[1]], roc_perf@y.values[[1]], 
              col = colors[i], lwd = 2)
        
        # Add text for legend
        text_x <- 0.7
        text_y <- 0.3 - (i-1) * 0.05
        text(text_x, text_y, 
             paste(model_name, "(AUC = ", round(auc_value, 3), ")", sep = ""), 
             col = colors[i], cex = 0.8)
      }, error = function(e) {
        cat("Error plotting ROC curve for", model_name, ":", e$message, "\n")
      })
    }
    
    # Store the best model based on AUC
    if(sum(!is.na(auc_values)) > 0) {
      best_idx <- which.max(auc_values)
      best_model_name <- names(prob_values)[best_idx]
      best_model_auc <- auc_values[best_idx]
      
      # Store for conclusion
      cat("\nBest model based on ROC curves: ", best_model_name, 
          " (AUC = ", round(best_model_auc, 3), ")\n", sep="")
    }
  } else {
    cat("No probability values available for ROC curves\n")
  }
}, error = function(e) {
  cat("ERROR plotting ROC curves:", e$message, "\n")
})

# ======= 14. Conclusion =======
debug_print("Generating conclusion")
cat("\n============ CONCLUSION ============\n")

# Gather performance metrics from all available models
perf_objects <- list()
if(exists("logistic_perf") && !is.null(logistic_perf)) perf_objects$`Logistic Regression` <- logistic_perf
if(exists("nb_perf") && !is.null(nb_perf)) perf_objects$`Naive Bayes` <- nb_perf
if(exists("tree_perf") && !is.null(tree_perf)) perf_objects$`Decision Tree` <- tree_perf
if(exists("rf_perf") && !is.null(rf_perf)) perf_objects$`Random Forest` <- rf_perf
if(exists("xgb_perf") && !is.null(xgb_perf)) perf_objects$`XGBoost` <- xgb_perf
if(exists("svm_perf") && !is.null(svm_perf)) perf_objects$`SVM (RBF)` <- svm_perf
if(exists("nn_perf") && !is.null(nn_perf)) perf_objects$`Neural Network` <- nn_perf

# Check if we have any successful models
if(length(perf_objects) > 0) {
  # Find the best model based on AUC
  auc_values <- sapply(perf_objects, function(x) {
    if(!is.null(x$auc)) return(x$auc) else return(NA)
  })
  
  # If we have valid AUC values
  if(sum(!is.na(auc_values)) > 0) {
    best_model_index <- which.max(auc_values)
    best_model_name <- names(perf_objects)[best_model_index]
    best_auc <- auc_values[best_model_index]
    
    cat("Best performing model:", best_model_name, "\n")
    cat("Best AUC:", round(best_auc, 4), "\n")
    cat("Full performance metrics for", best_model_name, ":\n")
    
    # Print metrics for best model
    best_metrics <- perf_objects[[best_model_name]]
    cat("Accuracy:", round(best_metrics$accuracy, 4), "\n")
    cat("Precision:", round(best_metrics$precision, 4), "\n")
    cat("Recall:", round(best_metrics$recall, 4), "\n")
    cat("F1-Score:", round(best_metrics$f1, 4), "\n")
    cat("AUC:", round(best_metrics$auc, 4), "\n")
    
    if(!is.null(best_metrics$gini)) {
      cat("Gini Coefficient:", round(best_metrics$gini, 4), "\n")
    }
    
    if(!is.null(best_metrics$ks)) {
      cat("KS Statistic:", round(best_metrics$ks, 4), "\n")
    }
  } else {
    cat("No valid AUC values available for model comparison.\n")
  }
} else {
  cat("No successful model evaluations available.\n")
}

cat("\nData Balancing Approach:\n")
cat("- The original dataset had an imbalanced class distribution\n")
cat("- We applied a combined over-sampling/under-sampling approach to create a balanced training dataset\n")
cat("- This method under-samples the majority class and over-samples the minority class\n")
cat("- Balancing helps improve model performance by preventing bias toward the majority class\n")

cat("\nRecommendations for Credit Risk Modeling:\n")
cat("1. Feature Importance: Focus on the most predictive features identified by the models\n")
cat("   (especially look at XGBoost feature importance which tends to be very reliable)\n")
cat("2. Model Selection: Use the best performing model for production deployment\n")
cat("3. Threshold Tuning: Adjust decision threshold based on business cost of false positives vs false negatives\n")
cat("4. Monitoring: Regularly validate model performance on new data\n")
cat("5. Explainability: Ensure model decisions can be explained to stakeholders and customers\n")
cat("   (Tree-based models like XGBoost and Random Forest provide feature importance metrics)\n")

cat("\n===== END OF ANALYSIS =====\n")