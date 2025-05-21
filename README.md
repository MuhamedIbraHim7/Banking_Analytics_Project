# Banking Analytics Project

## Overview
This project provides advanced analytics and machine learning solutions for the banking sector, focusing on two main areas:
- **Customer Churn Prediction**: Identifying customers likely to leave the bank, enabling proactive retention strategies.
- **Loan Default Prediction**: Assessing the risk of loan default using integrated customer and loan data.

## Features
- End-to-end data extraction, preprocessing, and feature engineering for both churn and loan models.
- Machine learning model training, evaluation, and interpretation (Random Forest, Logistic Regression, Gradient Boosting).
- Ready-to-use Jupyter Notebooks for reproducible analysis and business insights.
- Model serialization for deployment and integration.
- Azure Machine Learning integration for scalable model training (loan prediction).

## Project Structure
```
Banking_Analytics_Project/
│
├── Churn_Model_Prediction/
│   ├── Bank_Churn_Complete_Notebook.ipynb   # Full churn prediction workflow
│   ├── extract_churn_data.py                # Data extraction & feature engineering
│   ├── churn_prediction_data.csv            # Processed churn dataset
│   ├── best_rf_churn_model.pkl              # Trained churn model
│
├── Loan_Model_Prediction/
│   ├── script_run_notebook.ipynb            # Azure ML training workflow
│   ├── extract_loan_data.py                 # Data extraction & feature engineering
│   ├── integrated_loan_data.csv             # Integrated raw loan dataset
│   ├── loan_data_preprocessed.csv           # Preprocessed loan dataset
│   ├── loan_data_with_labels.csv            # Labeled loan data
│   ├── model.pkl                            # Trained loan model
│
├── LICENSE
├── README.md
```

## Getting Started
### Prerequisites
- Python 3.8+
- Recommended: Create a virtual environment
- Install required packages:
  ```sh
  pip install -r requirements.txt
  ```
  (See notebooks/scripts for specific dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, Azure ML SDK, etc.)

### Data Preparation
- Place the raw banking datasets (Excel/CSV) in the appropriate folders.
- Run the extraction scripts (`extract_churn_data.py`, `extract_loan_data.py`) to generate processed datasets.

### Model Training & Evaluation
- Use the provided Jupyter Notebooks for step-by-step model training, evaluation, and interpretation.
- For churn prediction, see `Churn_Model_Prediction/Bank_Churn_Complete_Notebook.ipynb`.
- For loan prediction, see `Loan_Model_Prediction/script_run_notebook.ipynb` (includes Azure ML integration).

### Model Deployment
- Trained models are saved as `.pkl` files for easy deployment.
- Example prediction functions are provided in the notebooks.

## Usage
- Modify and extend the notebooks/scripts for your own data and business requirements.
- Integrate the trained models into production systems as needed.

## Authors
- [Mohamed Ibrahim/Team26/EYouth Hackathon]

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
