#' Make predictions using a saved model
#' @param model_name Name of the model to use
#' @param new_data New data for predictions
#' @return Predictions from the model
predict_with_model <- function(model_name, new_data) {
  # Load model and metadata
  model_path <- file.path("results/models", model_name)
  model <- readRDS(file.path(model_path, "model.rds"))
  metadata <- readRDS(file.path(model_path, "metadata.rds"))
  
  # Preprocess new data consistently
  processed_data <- preprocess_data(new_data, is_training = FALSE)
  
  # Ensure all required features are present
  required_features <- metadata$features
  missing_features <- setdiff(required_features, names(processed_data$data))
  if(length(missing_features) > 0) {
    stop("Missing required features: ", paste(missing_features, collapse = ", "))
  }
  
  # Make predictions
  predictions <- predict(model, processed_data$data)
  
  return(predictions)
} 