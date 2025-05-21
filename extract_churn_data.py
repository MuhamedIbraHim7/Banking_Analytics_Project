import pandas as pd 
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Read all sheets from the Excel file
file_path = 'Banking_Analytics_Dataset.xlsx'
df_customers = pd.read_excel(file_path, sheet_name='Customers')
df_accounts = pd.read_excel(file_path, sheet_name='Accounts')
df_transactions = pd.read_excel(file_path, sheet_name='Transactions')
df_loans = pd.read_excel(file_path, sheet_name='Loans')
df_cards = pd.read_excel(file_path, sheet_name='Cards')
df_support = pd.read_excel(file_path, sheet_name='SupportCalls')

# Define key dates
snapshot_date = pd.to_datetime('2024-11-15')
churn_period_end = pd.to_datetime('2025-05-15')

# --- Target Variable: Churn ---
# Identify active customers (those with transactions in the churn period)
active_transactions = df_transactions[
    (df_transactions['TransactionDate'] >= snapshot_date) & 
    (df_transactions['TransactionDate'] <= churn_period_end)
]
active_account_ids = active_transactions['AccountID'].unique()
active_customers = df_accounts[df_accounts['AccountID'].isin(active_account_ids)]['CustomerID'].unique()

# Create target DataFrame
df_target = df_customers[['CustomerID']].copy()
df_target['churn'] = np.where(df_target['CustomerID'].isin(active_customers), 0, 1)

# --- Feature Extraction ---
# 1. Tenure (from Customers)
df_customers['JoinDate'] = pd.to_datetime(df_customers['JoinDate'])
df_features = df_customers[['CustomerID']].copy()
df_features['tenure'] = (snapshot_date - df_customers['JoinDate']).dt.days

# 2. Number of Accounts and Total Balance (from Accounts)
accounts_per_customer = df_accounts.groupby('CustomerID')['AccountID'].count().reset_index(name='number_of_accounts')
total_balance = df_accounts.groupby('CustomerID')['Balance'].sum().reset_index(name='total_balance')
df_features = pd.merge(df_features, accounts_per_customer, on='CustomerID', how='left')
df_features = pd.merge(df_features, total_balance, on='CustomerID', how='left')

# 3. Transaction Features: Recency, Frequency, Monetary (from Transactions)
df_trans_acc = pd.merge(
    df_transactions[df_transactions['TransactionDate'] < snapshot_date], 
    df_accounts, 
    on='AccountID'
)
trans_grouped = df_trans_acc.groupby('CustomerID').agg(
    last_transaction_date=('TransactionDate', 'max'),
    frequency=('TransactionID', 'count'),
    monetary=('Amount', 'sum')
).reset_index()
df_features = pd.merge(df_features, trans_grouped, on='CustomerID', how='left')
df_features['recency'] = (snapshot_date - df_features['last_transaction_date']).dt.days
df_features['recency'] = df_features['recency'].fillna(9999)  # Large value for no transactions
df_features['frequency'] = df_features['frequency'].fillna(0)
df_features['monetary'] = df_features['monetary'].fillna(0)
df_features = df_features.drop(columns=['last_transaction_date'])

# 4. Has Active Loan (from Loans)
df_loans['LoanEndDate'] = pd.to_datetime(df_loans['LoanEndDate'])
active_loans = df_loans[(df_loans['LoanEndDate'].isna()) | (df_loans['LoanEndDate'] > snapshot_date)]
has_active_loan = active_loans['CustomerID'].unique()
df_features['has_active_loan'] = df_features['CustomerID'].isin(has_active_loan).astype(int)

# 5. Number of Cards (from Cards)
cards_per_customer = df_cards.groupby('CustomerID')['CardID'].count().reset_index(name='number_of_cards')
df_features = pd.merge(df_features, cards_per_customer, on='CustomerID', how='left')
df_features['number_of_cards'] = df_features['number_of_cards'].fillna(0)

# 6. Support Call Features: Frequency and Resolution Rate (from SupportCalls)
df_support['CallDate'] = pd.to_datetime(df_support['CallDate'])
support_before_snapshot = df_support[df_support['CallDate'] < snapshot_date]
support_grouped = support_before_snapshot.groupby('CustomerID').agg(
    support_call_frequency=('CallID', 'count'),
    resolved_count=('Resolved', lambda x: (x == 'Yes').sum())
).reset_index()
support_grouped['resolution_rate'] = support_grouped['resolved_count'] / support_grouped['support_call_frequency']
df_features = pd.merge(
    df_features, 
    support_grouped[['CustomerID', 'support_call_frequency', 'resolution_rate']], 
    on='CustomerID', 
    how='left'
)
df_features['support_call_frequency'] = df_features['support_call_frequency'].fillna(0)
df_features['resolution_rate'] = df_features['resolution_rate'].fillna(1)  # Assume resolved if no calls

# --- Combine Features and Target ---
df_final = pd.merge(df_features, df_target, on='CustomerID')

# Fill any remaining NaN values appropriately
df_final['number_of_accounts'] = df_final['number_of_accounts'].fillna(0)
df_final['total_balance'] = df_final['total_balance'].fillna(0)

# Display the final DataFrame (for verification)
print(df_final.head())

# Save to CSV for model training
df_final.to_csv('churn_prediction_data.csv', index=False)


df = pd.read_csv('churn_prediction_data.csv')
df.info()
# Check for missing values  
missing_values = df.isnull().sum()
print("Missing values in each column:") 
print(missing_values[missing_values > 0])
# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
# Check for data types
data_types = df.dtypes
print("Data types of each column:")
print(data_types)
df.describe()

#  Data Preprocessing
# 1. Fill Missing Values
numerical_features = ['tenure', 'total_balance', 'recency', 'support_call_frequency']
for feature in numerical_features:
    df[feature] = df[feature].fillna(df[feature].median())


# 4. Scale Numerical Features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Save preprocessed data
df.to_csv('preprocessed_churn_data.csv', index=False)
