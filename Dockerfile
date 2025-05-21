FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
COPY Loan_Model_Prediction/app.py .
COPY Loan_Model_Prediction/model.pkl .

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
