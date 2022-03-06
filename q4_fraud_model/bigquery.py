from google.cloud import bigquery
import pandas as pd
from google.cloud import storage
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from q4_fraud_model.fraudmodel import clean_data


def create_table():
    df = pd.read_pickle("/dev_data_not_important/clean_fraud_data.pkl")
    print(df.columns)

    dataset_ref = bigquery.DatasetReference("chiayi-xfers-technical-test", "fraud_feature_engineering")
    table_ref = dataset_ref.table('results_table')
    # 31 features
    schema = [
        bigquery.SchemaField("accountNumber", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("customerId", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("creditLimit", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("availableMoney", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("transactionAmount", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("merchantName", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("acqCountry", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("merchantCountryCode", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("posEntryMode", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("posConditionCode", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("merchantCategoryCode", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("cardCVV", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("enteredCVV", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("cardLast4Digits", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("transactionType", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("currentBalance", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("cardPresent", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("expirationDateKeyInMatch", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("lastAddressChangeYear", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("lastAddressChangeMonth", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("lastAddressChangeDay", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("transactionYear", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("transactionMonth", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("transactionDay", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("transactionHour", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("transactionMinute", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("transactionSecond", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("accountOpenYear", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("accountOpenMonth", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("currentExpMonth", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("currentExpYear", "STRING", mode="REQUIRED"),
    ]

    table = bigquery.Table(table_ref, schema=schema)
    client = bigquery.Client()
    table = client.create_table(table)
    print(table)


def upload_csv_to_gcs():

    df = pd.read_pickle("/dev_data_not_important/clean_fraud_data.pkl")
    df = df.drop(columns=["isFraud"])
    df.to_csv("fe_results.csv", index=False)

    client = storage.Client()

    bucket = client.get_bucket("chiayi-xfers-technical-test-bucket")
    blob = bucket.blob("fe_results.csv")
    blob.upload_from_filename("/Users/cseow/PycharmProjects/chiayi-test/q4_fraud_model/fe_results.csv")


def populate_table():
    uri = "gs://chiayi-xfers-technical-test-bucket/fe_results.csv"

    client = bigquery.Client()

    job_config = bigquery.LoadJobConfig(
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
    )

    load_job = client.load_table_from_uri(
        uri, "chiayi-xfers-technical-test.fraud_feature_engineering.results_table", job_config=job_config)  # Make an API request.

    load_job.result()  # Waits for the job to complete.

    destination_table = client.get_table("chiayi-xfers-technical-test.fraud_feature_engineering.results_table")  # Make an API request.
    print("Loaded {} rows.".format(destination_table.num_rows))


def dag():
    default_args = {
            'owner': 'airflow',
            'depends_on_past': False,
            'start_date': datetime.today(),
            'email': ['airflow@airflow.com'],
            'email_on_failure': False,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
    }

    dag = DAG('tutorial', default_args=default_args, schedule_interval=timedelta(days=1))

    PythonOperator(dag=dag,
                   task_id='my_move_task',
                   provide_context=False,
                   python_callable=process_data())


if __name__ == '__main__':
    dag()
