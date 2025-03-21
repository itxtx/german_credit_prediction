#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <script_name>"
    echo "Example: $0 01_data_import.R"
    exit 1
fi

docker-compose run --rm r-analysis R --vanilla -e "source('scripts/$1')"