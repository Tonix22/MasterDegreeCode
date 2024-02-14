# README for Kaggle Dataset Download Script

This script is designed to help you set up your Kaggle API credentials and download a specific dataset from Kaggle. Below are the instructions and requirements to use this script successfully.

## Prerequisites

Before you run the script, ensure you have the following:

1. **Kaggle Account**: You must have a Kaggle account. If you don't have one, you can sign up at [Kaggle](https://www.kaggle.com).

2. **Kaggle API Token**: 
   - Go to your Kaggle account settings.
   - Scroll to the API section and click "Create New API Token".
   - This will download a `kaggle.json` file containing your API credentials and **copy it to this folder**

3. **Bash Shell**: Run SetupKaglle.sh script to setup your Kaggle credential, so you need a Unix-like environment to run it. Linux and macOS should work out of the box. Windows users can use WSL (Windows Subsystem for Linux) or a similar Unix-like environment.

4. **Kaggle Python Package**: Ensure the Kaggle Python package is installed. You can install it using pip:
   ```bash
   pip install kaggle

5. **Download** : This will download data set for Line of Sight and Non Line of sight
    ```bash
    ./Download.sh
    ```

    ```
    kaggle_dataset
    ├── v2v80211p_LOS.mat
    └── v2v80211p_NLOS.mat
    ```


