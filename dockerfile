# Use the R base image
FROM rocker/r-ver:4.2.0

# Install system dependencies needed for R packages
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    make \
    gcc \
    g++ \
    cmake \
    libnlopt-dev \
    libgit2-dev \
    pandoc \
    pandoc-citeproc \
    here \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install required R packages
RUN R -e "install.packages(c('readr', 'dplyr', 'tidyr', 'ggplot2', 'caret', 'rpart', 'rpart.plot', \
                            'randomForest', 'neuralnet', 'ROCR', 'pROC', 'e1071', \
                            'kernlab', 'MASS', 'recipes', 'xgboost', 'devtools', \
                            'BiocManager', 'rmarkdown', 'knitr', 'naivebayes', 'fmsb', 'e1071', \
                            'pdp', 'vip', 'gridExtra'), \
                            repos='https://cloud.r-project.org/')"
# Create directories for the project structure
RUN mkdir -p /app/data/raw \
    /app/data/processed \
    /app/scripts/utils \
    /app/scripts/03_models \
    /app/results/figures/eda \
    /app/results/figures/models \
    /app/results/figures/roc \
    /app/results/models \
    /app/results/performance \
    /app/reports

# Set the working directory
WORKDIR /app

# Copy the project files (assuming they're in the current directory)
COPY . /app/

# Command to run when the container starts
#CMD ["R", "--vanilla", "-e", "source('scripts/05_analysis.R')"]
CMD ["/bin/bash"]