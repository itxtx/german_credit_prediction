# ======= 1. Importing and Understanding the Dataset =======

# Install required packages if not already installed
required_packages <- c("readr", "dplyr", "tidyr", "ggplot2", "caret", "rpart", "rpart.plot", 
                       "randomForest", "neuralnet", "ROCR", "pROC", "e1071", "DMwR", 
                       "kernlab", "gbm", "MASS")

# Check and install missing packages
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load required libraries
library(readr)        # For reading data
library(dplyr)        # For data manipulation
library(tidyr)        # For data tidying
library(ggplot2)      # For visualization
library(caret)        # For model training and evaluation
library(rpart)        # For decision tree
library(rpart.plot)   # For plotting decision trees
library(randomForest) # For random forest
library(neuralnet)    # For neural networks
library(ROCR)         # For ROC curves
library(pROC)         # For AUC calculation
library(e1071)        # For SVM and Naive Bayes
library(DMwR)         # For SMOTE
library(kernlab)      # For kernel methods
library(gbm)          # For gradient boosting
library(MASS)         # For LDA and other statistical methods

# Set seed for reproducibility
set.seed(123)

# Download German Credit Dataset if needed
# URL: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/
# We'll assume the data is downloaded and saved as 'german_credit.csv'

# Load the dataset
# If you have the original .data file, you may need to specify column names
#german_credit <- read.csv("german_credit.csv", header = TRUE)

# If you don't have the dataset locally, you can use this to download directly:
german_credit <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data", sep = " ", header = FALSE)
column_names <- c("checking_status", "duration", "credit_history", "purpose", "credit_amount", 
                 "savings_status", "employment", "installment_commitment", "personal_status", 
                 "other_parties", "residence_since", "property_magnitude", "age", 
                 "other_payment_plans", "housing", "existing_credits", "job", 
                 "num_dependents", "own_telephone", "foreign_worker", "class")
colnames(german_credit) <- column_names

# Display basic dataset information
str(german_credit)
summary(german_credit)

# ======= 2. Understanding Unbalanced Data and Converting Class into Factors =======

# Check class distribution
table(german_credit$class)
class_distribution <- prop.table(table(german_credit$class)) * 100
print(paste("Percentage of class 1 (Good Credit):", round(class_distribution[1], 2), "%"))
print(paste("Percentage of class 2 (Bad Credit):", round(class_distribution[2], 2), "%"))

# Convert categorical variables to factors
categorical_cols <- c("checking_status", "credit_history", "purpose", "savings_status", 
                      "employment", "personal_status", "other_parties", "property_magnitude", 
                      "other_payment_plans", "housing", "job", "own_telephone", 
                      "foreign_worker", "class")

german_credit[categorical_cols] <- lapply(german_credit[categorical_cols], as.factor)

# Convert class to a binary factor with valid R variable names
# In the original dataset: 1 = good, 2 = bad
# Using "Good" and "Bad" as factor levels instead of 1 and 0
german_credit$class <- factor(ifelse(german_credit$class == 1, "Good", "Bad"))

# Visualize class distribution
ggplot(data = german_credit, aes(x = class, fill = class)) +
  geom_bar() +
  scale_fill_manual(values = c("red", "green")) +
  labs(title = "Class Distribution in German Credit Dataset",
       x = "Credit Risk (0 = Bad, 1 = Good)",
       y = "Count") +
  theme_minimal()

# ======= 3. Dividing the Dataset into Equal Parts with Equal Distribution =======

# Stratified sampling to maintain class distribution
train_index <- createDataPartition(german_credit$class, p = 0.7, list = FALSE)
train_data <- german_credit[train_index, ]
test_data <- german_credit[-train_index, ]

# Verify class distribution in training and testing sets
train_dist <- prop.table(table(train_data$class)) * 100
test_dist <- prop.table(table(test_data$class)) * 100

print("Class distribution in training set (%)")
print(train_dist)
print("Class distribution in testing set (%)")
print(test_dist)

# ======= 4. Imputing for Null Values =======

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

# ======= 5. Defining Cross-Validation, Metrics, and Preprocessing =======

# Setup cross-validation method
ctrl <- trainControl(
  method = "cv",           # Cross-validation
  number = 5,              # 5-fold
  classProbs = TRUE,       # Calculate class probabilities
  summaryFunction = twoClassSummary,  # Use ROC summary
  savePredictions = TRUE   # Save predictions for later analysis
)

# Define preprocessing steps
preprocess_steps <- c("center", "scale")  # Standardize numeric variables

# Function to evaluate model performance
evaluate_model <- function(pred, actual, pred_prob = NULL) {
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
    pred_obj <- prediction(pred_prob, as.numeric(actual) - 1)
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
  } else {
    return(list(
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      f1 = f1,
      confusion_matrix = conf_matrix
    ))
  }
}

# ======= 6. LOGISTIC REGRESSION and Feature Selection =======

# Train logistic regression model
set.seed(123)
logistic_model <- train(
  class ~ ., 
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  preProcess = preprocess_steps,
  metric = "ROC"
)

# Print model summary
print(logistic_model)
summary(logistic_model$finalModel)

# Feature importance based on p-values
logistic_summary <- summary(logistic_model$finalModel)
feature_importance <- data.frame(
  Feature = rownames(logistic_summary$coefficients),
  Coefficient = logistic_summary$coefficients[, "Estimate"],
  PValue = logistic_summary$coefficients[, "Pr(>|z|)"]
)

# Sort by absolute coefficient values
feature_importance <- feature_importance[order(abs(feature_importance$Coefficient), decreasing = TRUE), ]
print("Feature Importance (Logistic Regression):")
print(feature_importance)

# Visualize top 10 features
top_features <- head(feature_importance, 10)
ggplot(top_features, aes(x = reorder(Feature, abs(Coefficient)), y = Coefficient)) +
  geom_col(aes(fill = PValue < 0.05)) +
  coord_flip() +
  labs(title = "Top 10 Features by Coefficient Magnitude",
       x = "Feature",
       y = "Coefficient",
       fill = "Significant (p < 0.05)") +
  theme_minimal()

# Make predictions on test set
logistic_pred <- predict(logistic_model, newdata = test_data)
logistic_prob <- predict(logistic_model, newdata = test_data, type = "prob")[, "1"]

# Evaluate the model
cat("\n--- Logistic Regression Performance ---\n")
logistic_perf <- evaluate_model(logistic_pred, test_data$class, logistic_prob)
print(logistic_perf)

# ======= 7. Applying Bayesian Model & Recursive Partitioning =======

# Naive Bayes model
set.seed(123)
nb_model <- train(
  class ~ .,
  data = train_data,
  method = "naive_bayes",
  trControl = ctrl,
  preProcess = preprocess_steps,
  metric = "ROC"
)

print(nb_model)

1# Make predictions with Naive Bayes
nb_pred <- predict(nb_model, newdata = test_data)
nb_prob <- predict(nb_model, newdata = test_data, type = "prob")[, "1"]

# Evaluate Naive Bayes
cat("\n--- Naive Bayes Performance ---\n")
nb_perf <- evaluate_model(nb_pred, test_data$class, nb_prob)
print(nb_perf)

# Decision Tree (Recursive Partitioning)
set.seed(123)
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
tree_pred <- predict(tree_model, newdata = test_data)
tree_prob <- predict(tree_model, newdata = test_data, type = "prob")[, "1"]

# Evaluate Decision Tree
cat("\n--- Decision Tree Performance ---\n")
tree_perf <- evaluate_model(tree_pred, test_data$class, tree_prob)
print(tree_perf)

# ======= 8. Improve Results using Random Forest =======

# Train Random Forest model
set.seed(123)
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
rf_pred <- predict(rf_model, newdata = test_data)
rf_prob <- predict(rf_model, newdata = test_data, type = "prob")[, "1"]

# Evaluate Random Forest
cat("\n--- Random Forest Performance ---\n")
rf_perf <- evaluate_model(rf_pred, test_data$class, rf_prob)
print(rf_perf)

# ======= 9. Implementing Boosting (AdaBoost & Gradient Boosting) =======

# AdaBoost implementation with gbm
set.seed(123)
adaboost_model <- train(
  class ~ .,
  data = train_data,
  method = "gbm",
  distribution = "adaboost",
  trControl = ctrl,
  tuneLength = 5,
  metric = "ROC",
  verbose = FALSE
)

print(adaboost_model)
print(adaboost_model$bestTune)

# Make predictions with AdaBoost
adaboost_pred <- predict(adaboost_model, newdata = test_data)
adaboost_prob <- predict(adaboost_model, newdata = test_data, type = "prob")[, "1"]

# Evaluate AdaBoost
cat("\n--- AdaBoost Performance ---\n")
adaboost_perf <- evaluate_model(adaboost_pred, test_data$class, adaboost_prob)
print(adaboost_perf)

# Gradient Boosting Model
set.seed(123)
gbm_model <- train(
  class ~ .,
  data = train_data,
  method = "gbm",
  distribution = "bernoulli",
  trControl = ctrl,
  tuneLength = 5,
  metric = "ROC",
  verbose = FALSE
)

print(gbm_model)
print(gbm_model$bestTune)

# Make predictions with Gradient Boosting
gbm_pred <- predict(gbm_model, newdata = test_data)
gbm_prob <- predict(gbm_model, newdata = test_data, type = "prob")[, "1"]

# Evaluate Gradient Boosting
cat("\n--- Gradient Boosting Performance ---\n")
gbm_perf <- evaluate_model(gbm_pred, test_data$class, gbm_prob)
print(gbm_perf)

# ======= 10. Model Improvement with Gaussian RBF Kernel (SVM) =======

# Train SVM with Radial Basis Function (RBF) kernel
set.seed(123)
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
svm_pred <- predict(svm_model, newdata = test_data)
svm_prob <- predict(svm_model, newdata = test_data, type = "prob")[, "1"]

# Evaluate SVM
cat("\n--- SVM with RBF Kernel Performance ---\n")
svm_perf <- evaluate_model(svm_pred, test_data$class, svm_prob)
print(svm_perf)

# ======= 11. Implementing Neural Network =======

# First, prepare the data for neural network
# Normalize numeric variables
numeric_cols <- sapply(train_data, is.numeric)
train_data_nn <- train_data
test_data_nn <- test_data

# Scale numeric features
preproc <- preProcess(train_data[, numeric_cols], method = c("center", "scale"))
train_data_nn[, numeric_cols] <- predict(preproc, train_data[, numeric_cols])
test_data_nn[, numeric_cols] <- predict(preproc, test_data[, numeric_cols])

# Convert factors to dummy variables
dummies <- dummyVars(~ ., data = train_data_nn, fullRank = TRUE)
train_data_nn <- predict(dummies, train_data_nn)
test_data_nn <- predict(dummies, test_data_nn)

# Convert to data frames
train_data_nn <- as.data.frame(train_data_nn)
test_data_nn <- as.data.frame(test_data_nn)

# Ensure target variable is in the right format
train_target <- ifelse(train_data$class == "Good", 1, 0)
test_target <- ifelse(test_data$class == "Good", 1, 0)

# Train neural network model
set.seed(123)
nn_model <- train(
  x = train_data_nn,
  y = factor(train_target),
  method = "nnet",
  trControl = ctrl,
  tuneLength = 5,
  metric = "ROC",
  trace = FALSE,
  maxit = 500
)

print(nn_model)
print(nn_model$bestTune)

# Make predictions with Neural Network
nn_pred <- predict(nn_model, newdata = test_data_nn)
nn_prob <- predict(nn_model, newdata = test_data_nn, type = "prob")[, "1"]

# Evaluate Neural Network
cat("\n--- Neural Network Performance ---\n")
nn_perf <- evaluate_model(nn_pred, factor(test_target), nn_prob)
print(nn_perf)

# ======= 12. Performance Comparison and Visualization =======

# Collect all model performances
models <- c("Logistic Regression", "Naive Bayes", "Decision Tree", 
            "Random Forest", "AdaBoost", "Gradient Boosting", 
            "SVM (RBF)", "Neural Network")

# Metrics to compare
accuracy <- c(logistic_perf$accuracy, nb_perf$accuracy, tree_perf$accuracy, 
              rf_perf$accuracy, adaboost_perf$accuracy, gbm_perf$accuracy, 
              svm_perf$accuracy, nn_perf$accuracy)

precision <- c(logistic_perf$precision, nb_perf$precision, tree_perf$precision, 
               rf_perf$precision, adaboost_perf$precision, gbm_perf$precision, 
               svm_perf$precision, nn_perf$precision)

recall <- c(logistic_perf$recall, nb_perf$recall, tree_perf$recall, 
            rf_perf$recall, adaboost_perf$recall, gbm_perf$recall, 
            svm_perf$recall, nn_perf$recall)

f1 <- c(logistic_perf$f1, nb_perf$f1, tree_perf$f1, 
        rf_perf$f1, adaboost_perf$f1, gbm_perf$f1, 
        svm_perf$f1, nn_perf$f1)

auc <- c(logistic_perf$auc, nb_perf$auc, tree_perf$auc, 
         rf_perf$auc, adaboost_perf$auc, gbm_perf$auc, 
         svm_perf$auc, nn_perf$auc)

# Create comparison dataframe
comparison <- data.frame(
  Model = models,
  Accuracy = accuracy,
  Precision = precision,
  Recall = recall,
  F1_Score = f1,
  AUC = auc
)

# Sort by AUC (best model on top)
comparison <- comparison[order(comparison$AUC, decreasing = TRUE), ]

# Print comparison table
print("Model Performance Comparison:")
print(comparison)

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

# Plot AUC comparison
ggplot(comparison, aes(x = reorder(Model, AUC), y = AUC)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(AUC, 3)), vjust = -0.3) +
  labs(title = "Model Comparison by AUC",
       x = "Model",
       y = "AUC") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ylim(0, max(comparison$AUC) * 1.1)

# ======= 13. ROC Curves for All Models =======

# Create a ROC plot for all models
roc_data <- data.frame(
  actual = as.numeric(test_data$class) - 1,
  logistic = logistic_prob,
  naive_bayes = nb_prob,
  decision_tree = tree_prob,
  random_forest = rf_prob,
  adaboost = adaboost_prob,
  gbm = gbm_prob,
  svm = svm_prob,
  neural_net = nn_prob
)

# Create empty plot
roc_plot <- plot.new()
plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     main = "ROC Curves Comparison")
abline(0, 1, lty = 2, col = "gray")

# Add ROC curves for each model with different colors
colors <- rainbow(8)
for (i in 2:ncol(roc_data)) {
  model_name <- colnames(roc_data)[i]
  pred_obj <- prediction(roc_data[, i], roc_data$actual)
  roc_perf <- performance(pred_obj, "tpr", "fpr")
  auc_value <- performance(pred_obj, "auc")@y.values[[1]]
  
  lines(roc_perf@x.values[[1]], roc_perf@y.values[[1]], 
        col = colors[i-1], lwd = 2)
  
  # Add text for legend
  text_x <- 0.7
  text_y <- 0.3 - (i-2) * 0.05
  text(text_x, text_y, 
       paste(model_name, "(AUC = ", round(auc_value, 3), ")", sep = ""), 
       col = colors[i-1], cex = 0.8)
}

# ======= 14. Conclusion =======

# Find the best model
best_model_index <- which.max(comparison$AUC)
best_model_name <- as.character(comparison$Model[best_model_index])
best_auc <- comparison$AUC[best_model_index]

cat("\n============ CONCLUSION ============\n")
cat("Best performing model:", best_model_name, "\n")
cat("Best AUC:", best_auc, "\n")
cat("Full performance metrics for", best_model_name, ":\n")

best_perf <- switch(best_model_name,
                    "Logistic Regression" = logistic_perf,
                    "Naive Bayes" = nb_perf,
                    "Decision Tree" = tree_perf,
                    "Random Forest" = rf_perf,
                    "AdaBoost" = adaboost_perf,
                    "Gradient Boosting" = gbm_perf,
                    "SVM (RBF)" = svm_perf,
                    "Neural Network" = nn_perf)

print(best_perf)

cat("\nRecommendations:\n")
cat("1. The", best_model_name, "model performs best for classifying loan applications.\n")
cat("2. Key features influencing the credit decision include:\n")

# Get top features from the best model if possible
if(best_model_name == "Logistic Regression") {
  cat("   - Based on logistic regression coefficients:\n")
  top_features <- head(feature_importance, 5)
  for(i in 1:nrow(top_features)) {
    cat("     *", top_features$Feature[i], "with coefficient", round(top_features$Coefficient[i], 3), "\n")
  }
} else if(best_model_name == "Random Forest") {
  cat("   - Based on random forest importance:\n")
  rf_imp <- varImp(rf_model)$importance
  rf_imp$Feature <- rownames(rf_imp)
  rf_imp <- rf_imp[order(rf_imp$Overall, decreasing = TRUE), ]
  top_rf_features <- head(rf_imp, 5)
  for(i in 1:nrow(top_rf_features)) {
    cat("     *", top_rf_features$Feature[i], "with importance", round(top_rf_features$Overall[i], 3), "\n")
  }
}

cat("3. For implementation, consider:\n")
cat("   - Using", best_model_name, "as the primary model\n")
cat("   - Setting appropriate probability thresholds based on business priorities\n")
cat("   - Regular model retraining with new data\n")
cat("   - Monitoring model performance over time\n")