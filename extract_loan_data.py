import pandas as pd
import numpy as np
from datetime import datetime

# Load sheets from Excel file
file_path = "Banking_Analytics_Dataset.xlsx"
customers_df = pd.read_excel(file_path, sheet_name="Customers")
accounts_df = pd.read_excel(file_path, sheet_name="Accounts")
transactions_df = pd.read_excel(file_path, sheet_name="Transactions")
loans_df = pd.read_excel(file_path, sheet_name="Loans")
supportcalls_df = pd.read_excel(file_path, sheet_name="SupportCalls")

# Feature Engineering
# Calculate Tenure from Customers
current_date = pd.Timestamp.now()
customers_df['Tenure'] = (current_date - pd.to_datetime(customers_df['JoinDate'])).dt.days / 365.25

# Aggregate Accounts: TotalBalance per customer
total_balance_df = accounts_df.groupby('CustomerID')['Balance'].sum().reset_index(name='TotalBalance')

# Aggregate Transactions: TotalPayments per customer (assuming "Payment" transactions)
transactions_with_customer = pd.merge(
    transactions_df,
    accounts_df[['AccountID', 'CustomerID']],
    on='AccountID',
    how='left'
)
payment_transactions = transactions_with_customer[transactions_with_customer['TransactionType'] == 'Payment']
total_payments_df = payment_transactions.groupby('CustomerID')['Amount'].sum().reset_index(name='TotalPayments')

# Aggregate SupportCalls: SupportIssueFrequency per customer
support_issue_freq_df = supportcalls_df.groupby('CustomerID').size().reset_index(name='SupportIssueFrequency')

# Calculate LoanTerm from Loans
loans_df['LoanStartDate'] = pd.to_datetime(loans_df['LoanStartDate'])
loans_df['LoanEndDate'] = pd.to_datetime(loans_df['LoanEndDate'])
loans_df['LoanTerm'] = (loans_df['LoanEndDate'] - loans_df['LoanStartDate']).dt.days / 365.25

# Merge DataFrames
merged_df = loans_df[['LoanID', 'CustomerID','LoanAmount', 'InterestRate', 'LoanType', 'LoanTerm']]
merged_df = pd.merge(merged_df, customers_df[['CustomerID', 'Tenure']], on='CustomerID', how='left')
merged_df = pd.merge(merged_df, total_balance_df, on='CustomerID', how='left')
merged_df = pd.merge(merged_df, total_payments_df, on='CustomerID', how='left', suffixes=('', '_transactions'))
merged_df = pd.merge(merged_df, support_issue_freq_df, on='CustomerID', how='left')

# Handle missing values
numerical_cols = ['LoanAmount', 'InterestRate', 'LoanTerm', 'TotalBalance', 'TotalPayments', 'Tenure', 'SupportIssueFrequency']
for col in numerical_cols:
    merged_df[col] = merged_df[col].fillna(merged_df[col].median())

# Save integrated dataset
merged_df.to_csv('integrated_loan_data.csv', index=False)
print("Integrated dataset saved as 'integrated_loan_data.csv'")


# Preprocessing and Feature Engineering
import pandas as pd

# 1. Load data
df = pd.read_csv('integrated_loan_data.csv')

# get the dummies and store it in a variable
dummies = pd.get_dummies(df.LoanType, dtype=np.int8)

# Concatenate the dummies to original dataframe
merged = pd.concat([df, dummies], axis='columns')

# 2. Drop identifier columns
df = merged.drop(['LoanID', 'CustomerID', 'LoanType'], axis=1)

# 11. Save cleaned and preprocessed data
df.to_csv('loan_data_preprocessed.csv', index=False)
#df.to_parquet('loan_data_preprocessed.parquet', index=False)