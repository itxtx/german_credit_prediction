# ======= 01_data_import.R =======
# This script downloads and imports the German Credit Dataset,
# performs initial data examination, and saves the raw data.

# Source utility scripts
source("scripts/utils/setup.R")

# Set up environment
setup_environment(debug_mode = TRUE)

# Function to download and import the German Credit Dataset
download_german_credit_data <- function(save_to_file = TRUE, 
                                      data_dir = "data/raw",
                                      file_name = "german_credit.csv") {
  # Create data directory if it doesn't exist
  if(!dir.exists(data_dir)) {
    dir.create(data_dir, recursive = TRUE)
    message(paste("Created directory:", data_dir))
  }
  
  # URL for German Credit Dataset
  url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
  
  message("Downloading German Credit Dataset from UCI repository...")
  
  # Create a temporary file to download the data
  temp_file <- tempfile()
  download_result <- tryCatch({
    download.file(url, temp_file, quiet = TRUE)
    TRUE
  }, error = function(e) {
    message("ERROR downloading data: ", e$message)
    FALSE
  })
  
  if(!download_result) {
    message("Failed to download the dataset. Using backup method or local file if available.")
    
    # Check if the file already exists locally
    local_path <- file.path(data_dir, file_name)
    if(file.exists(local_path)) {
      message("Using existing local file: ", local_path)
      german_credit <- read.csv(local_path)
      return(german_credit)
    } else {
      stop("Failed to download data and no local file available.")
    }
  }
  
  # Define column names according to UCI documentation
  column_names <- c(
    "checking_status", "duration", "credit_history", "purpose", "credit_amount", 
    "savings_status", "employment", "installment_commitment", "personal_status", 
    "other_parties", "residence_since", "property_magnitude", "age", 
    "other_payment_plans", "housing", "existing_credits", "job", 
    "num_dependents", "own_telephone", "foreign_worker", "class"
  )
  
  # Read the data with the correct column names
  german_credit <- read.table(temp_file, sep = " ", header = FALSE)
  colnames(german_credit) <- column_names
  
  message("Dataset successfully loaded with ", nrow(german_credit), " rows and ", 
         ncol(german_credit), " columns.")
  
  # Save the raw data if requested
  if(save_to_file) {
    output_path <- file.path(data_dir, file_name)
    write.csv(german_credit, output_path, row.names = FALSE)
    message("Raw data saved to: ", output_path)
  }
  
  return(german_credit)
}

# Function to examine the data structure and provide a summary
examine_data <- function(data, output_dir = "data/raw") {
  message("\n=== Data Structure ===")
  str_output <- capture.output(str(data))
  cat(paste(str_output, collapse = "\n"), "\n")
  
  message("\n=== Data Summary ===")
  summary_output <- capture.output(summary(data))
  cat(paste(summary_output, collapse = "\n"), "\n")
  
  # Check for any missing values
  missing_values <- colSums(is.na(data))
  message("\n=== Missing Values ===")
  if(sum(missing_values) > 0) {
    print(missing_values[missing_values > 0])
  } else {
    message("No missing values found.")
  }
  
  # Check class distribution
  message("\n=== Class Distribution ===")
  class_table <- table(data$class)
  class_pct <- prop.table(class_table) * 100
  class_df <- data.frame(
    Count = as.numeric(class_table),
    Percentage = as.numeric(class_pct)
  )
  rownames(class_df) <- names(class_table)
  print(class_df)
  
  # Create a simple data dictionary based on the column names
  message("\n=== Data Dictionary ===")
  data_dict <- data.frame(
    Column = colnames(data),
    Type = sapply(data, class),
    Description = "",
    stringsAsFactors = FALSE
  )
  
  # Add descriptions based on column names
  descriptions <- c(
    "Status of existing checking account",
    "Duration in months",
    "Credit history",
    "Purpose of loan",
    "Credit amount",
    "Savings account/bonds status",
    "Present employment since",
    "Installment rate as percentage of disposable income",
    "Personal status and sex",
    "Other debtors / guarantors",
    "Present residence since",
    "Property (e.g. real estate)",
    "Age in years",
    "Other installment plans",
    "Housing",
    "Number of existing credits at this bank",
    "Job",
    "Number of people being liable to provide maintenance for",
    "Telephone",
    "Foreign worker",
    "Credit risk classification (1=Good, 2=Bad)"
  )
  
  # Assign descriptions if they match the number of columns
  if(length(descriptions) == ncol(data)) {
    data_dict$Description <- descriptions
  }
  
  # Print and save data dictionary
  print(data_dict)
  
  # Save data dictionary to file
  dict_path <- file.path(output_dir, "data_dictionary.csv")
  write.csv(data_dict, dict_path, row.names = FALSE)
  message("Data dictionary saved to: ", dict_path)
  
  # Save summary statistics to file
  summary_path <- file.path(output_dir, "data_summary.txt")
  writeLines(c(str_output, "", summary_output), summary_path)
  message("Summary statistics saved to: ", summary_path)
  
  return(list(
    data_dict = data_dict,
    class_distribution = class_df,
    missing_values = missing_values
  ))
}

# Function to handle categorical variables and identify unique values
explore_categorical_variables <- function(data, output_dir = "data/raw") {
  message("\n=== Categorical Variables ===")
  
  # Identify categorical columns (assuming factors and characters)
  categorical_cols <- names(data)[sapply(data, function(x) is.factor(x) || is.character(x))]
  
  if(length(categorical_cols) == 0) {
    # If no categorical variables found, check for numeric columns that might be categorical
    potential_categorical <- names(data)[sapply(data, function(x) {
      is.numeric(x) && length(unique(x)) <= 10
    })]
    
    if(length(potential_categorical) > 0) {
      message("No explicit categorical variables found, but these numeric columns might be categorical:")
      print(potential_categorical)
    } else {
      message("No categorical variables found.")
    }
    
    return(NULL)
  }
  
  # Create a list to store category information
  category_info <- list()
  
  # Examine each categorical variable
  for(col in categorical_cols) {
    # Get unique values and their frequencies
    val_table <- table(data[[col]])
    val_pct <- prop.table(val_table) * 100
    
    category_info[[col]] <- data.frame(
      Value = names(val_table),
      Count = as.numeric(val_table),
      Percentage = as.numeric(val_pct),
      stringsAsFactors = FALSE
    )
    
    message("\nVariable: ", col)
    print(category_info[[col]])
  }
  
  # Save category information to file
  cat_path <- file.path(output_dir, "categorical_variables.csv")
  
  # Combine all category info into one dataframe
  all_cats <- do.call(rbind, lapply(names(category_info), function(name) {
    df <- category_info[[name]]
    df$Variable <- name
    return(df[, c("Variable", "Value", "Count", "Percentage")]  )
  }))
  
  write.csv(all_cats, cat_path, row.names = FALSE)
  message("Categorical variable information saved to: ", cat_path)
  
  return(category_info)
}

# Function to visualize class distribution
plot_class_distribution <- function(data, output_dir = "results/figures/eda") {
  # Check if ggplot2 is available
  if(!requireNamespace("ggplot2", quietly = TRUE)) {
    message("ggplot2 package is required for visualization. Skipping plot generation.")
    return(NULL)
  }
  
  # Create output directory if it doesn't exist
  if(!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    message(paste("Created directory:", output_dir))
  }
  
  # Create class distribution plot
  class_plot <- ggplot2::ggplot(data, ggplot2::aes(x = class, fill = class)) +
    ggplot2::geom_bar() +
    ggplot2::labs(
      title = "Class Distribution in German Credit Dataset",
      x = "Credit Risk",
      y = "Count"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::scale_fill_manual(values = c("1" = "green", "2" = "red"))
  
  # Save the plot
  plot_path <- file.path(output_dir, "class_distribution.png")
  ggplot2::ggsave(plot_path, class_plot, width = 8, height = 6)
  message("Class distribution plot saved to: ", plot_path)
  
  return(class_plot)
}

# Main execution section
main <- function() {
  message("\n====== Starting German Credit Dataset Import and Examination ======\n")
  
  # Download and import the data
  german_credit <- download_german_credit_data()
  
  # Examine the data
  data_info <- examine_data(german_credit)
  
  # Explore categorical variables
  cat_info <- explore_categorical_variables(german_credit)
  
  # Plot class distribution
  class_plot <- plot_class_distribution(german_credit)
  
  message("\n====== Data Import and Initial Examination Complete ======\n")
  
  # Return the imported data
  return(german_credit)
}

# Run the main function if this script is being run directly
if(!exists("DATA_IMPORT_SOURCED") || !DATA_IMPORT_SOURCED) {
  german_credit <- main()
  DATA_IMPORT_SOURCED <- TRUE
} else {
  message("01_data_import.R has been sourced. Use main() to run the import process.")
}