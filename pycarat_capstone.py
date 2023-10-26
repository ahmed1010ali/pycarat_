#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sqlite3

from pycaret.classification import setup, compare_models
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg

def load_and_preprocess_data():
    try:
        # Prompt the user to specify the data format (CSV, Excel, SQL)
        data_format = input("Enter the data format (CSV, Excel, SQL): ")
        
        if data_format == "csv":
            file_path = input("Enter the path to your CSV file: ")
            df = pd.read_csv(file_path)
        elif data_format == "excel":
            file_path = input("Enter the path to your Excel file: ")
            df = pd.read_excel(file_path)
        elif data_format == "sql":
            database_path = input("Enter the path to your SQLite database: ")
            query = input("Enter your SQL query: ")
            conn = sqlite3.connect(database_path)
            df = pd.read_sql(query, conn)
            conn.close()
        else:
            print("Invalid data format selected.")
            return

        # Print the loaded dataframe
        print("Loaded DataFrame:")
        print(df)
        
        # Select the target column
        chosen_target = input("Choose the Target Column: ")
        
        if chosen_target not in df.columns:
            print(f"Column '{chosen_target}' not found in the dataset.")
            return

        # Detect the task type (regression or classification)
        if np.issubdtype(df[chosen_target].dtype, np.number):
            task_type = "Regression"
        else:
            task_type = "Classification"
        
        print(f"Task Type: {task_type}")
        
        # Handle missing values and impute based on data type
        for column in df.columns:
            if column != chosen_target:
                if np.issubdtype(df[column].dtype, np.number):
                    impute_method = input(f"Select imputation method for {column} (mean/median/mode): ")
                    if impute_method == "mean":
                        df[column].fillna(df[column].mean(), inplace=True)
                    elif impute_method == "median":
                        df[column].fillna(df[column].median(), inplace=True)
                    else:
                        df[column].fillna(df[column].mode()[0], inplace=True)
                else:
                    impute_method = input(f"Select imputation method for {column} (most frequent/additional class): ")
                    if impute_method == "most frequent":
                        df[column].fillna(df[column].value_counts().idxmax(), inplace=True)
                    else:
                        df[column].fillna("Missing", inplace=True)
        
        # Select and drop columns
        columns_to_drop = input("Select columns to drop (comma-separated): ").split(",")
        columns_to_drop = [col.strip() for col in columns_to_drop]
        valid_columns = [col for col in columns_to_drop if col in df.columns]

        if valid_columns:
            df.drop(columns=valid_columns, inplace=True)
        else:
            print("No valid columns selected for dropping.")

        # Use PyCaret for model selection and evaluation
        print("Using PyCaret for model selection and evaluation:")
        if task_type == "Classification":
            setup_df = setup(data=df, target=chosen_target)
            best_model = compare_models()
            print("Best Classification Model:")
            print(best_model)
        else:
            setup_df = setup_reg(data=df, target=chosen_target)
            best_model_reg = compare_models_reg()
            print("Best Regression Model:")
            print(best_model_reg)
        
       
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to start the process
load_and_preprocess_data()


# In[ ]:




