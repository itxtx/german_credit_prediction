#!/bin/bash

# Start the container if it's not running
docker-compose up -d r-analysis

# Execute the R script in the running container
#docker-compose exec r-analysis Rscript scripts/04_model_comparison.R
docker-compose exec r-analysis Rscript scripts/05_analysis.R
