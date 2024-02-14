#!/bin/bash

# This script sets up the Kaggle API credentials
KAGGLE_JSON="kaggle.json"

# Destination directory for Kaggle API credentials
KAGGLE_DIR="$HOME/.kaggle"

# Check if the .kaggle directory exists, if not, create it
if [ ! -d "$KAGGLE_DIR" ]; then
    echo "Creating $KAGGLE_DIR directory..."
    mkdir -p $KAGGLE_DIR
else
    echo "$KAGGLE_DIR already exists."
fi

# Move the kaggle.json to the .kaggle directory
mv -v $KAGGLE_JSON $KAGGLE_DIR

# Change the permissions of the kaggle.json file
chmod 600 $KAGGLE_DIR/$KAGGLE_JSON

echo "Kaggle API credentials are set up."