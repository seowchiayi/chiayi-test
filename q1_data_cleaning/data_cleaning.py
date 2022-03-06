from google.cloud import bigquery
import pandas as pd
from google.cloud import storage


def get_data_from_bigquery_client_and_upload_to_gcs():
    client = bigquery.Client()

    project = "bigquery-public-data"
    dataset_id = "new_york"
    table_id = "311_service_requests"

    destination_uri = "gs://{}/{}".format("chiayi-xfers-technical-test-bucket", "chiayi_311_service_requests*.csv")
    dataset_ref = bigquery.DatasetReference(project, dataset_id)
    table_ref = dataset_ref.table(table_id)

    extract_job = client.extract_table(
        table_ref,
        destination_uri,
        location="US",
    )
    extract_job.result()

    print(
        "Exported {}:{}.{} to {}".format(project, dataset_id, table_id, destination_uri)
    )


def retrieve_csvs_from_gcs():
    num = 0
    first = '%012d' % num

    path = "gs://{}/{}".format("chiayi-xfers-technical-test-bucket", "chiayi_311_service_requests" + str(first) + ".csv")

    client = storage.Client()
    get_gcs_bucket = client.bucket("chiayi-xfers-technical-test-bucket")
    blob = get_gcs_bucket.blob("chiayi_311_service_requests" + str(first) + ".csv")

    # Remove Irrelevant Values
    # Get Rid of Duplicate Values
    # Avoid Typos( and similar errors)
    # Convert Data Types
    # Take Care of Missing Values

    # 594576 rows
    df = pd.read_csv(blob.open("r"))
    df = df.drop_duplicates()
    s = df.memory_usage(deep=True)

    # Convert Data Types

    # convert these 23 cols to category type because they have less than 50% of unique values
    df["resolution_description"] = df["resolution_description"].astype('category')
    df["agency_name"] = df["agency_name"].astype('category')
    df["agency"] = df["agency"].astype('category')
    df["complaint_type"] = df["complaint_type"].astype('category')
    df["descriptor"] = df["descriptor"].astype('category')
    df["location_type"] = df["location_type"].astype('category')
    df["street_name"] = df["street_name"].astype('category')
    df["address_type"] = df["address_type"].astype('category')
    df["city"] = df["city"].astype('category')
    df["vehicle_type"] = df["vehicle_type"].astype('category')
    df["landmark"] = df["landmark"].astype('category')
    df["facility_type"] = df["facility_type"].astype('category')
    df["status"] = df["status"].astype('category')
    df["community_board"] = df["community_board"].astype('category')
    df["park_facility_name"] = df["park_facility_name"].astype('category')
    df["park_borough"] = df["park_borough"].astype('category')
    df["open_data_channel_type"] = df["open_data_channel_type"].astype('category')
    df["taxi_company_borough"] = df["taxi_company_borough"].astype('category')
    df["taxi_pickup_location"] = df["taxi_pickup_location"].astype('category')
    df["bridge_highway_name"] = df["bridge_highway_name"].astype('category')
    df["bridge_highway_direction"] = df["bridge_highway_direction"].astype('category')
    df["road_ramp"] = df["road_ramp"].astype('category')
    df["bridge_highway_segment"] = df["bridge_highway_segment"].astype('category')

    # convert (4) created_date and closed_date and resolution_action_updated_date to datetime64
    pd.to_datetime(df["created_date"], utc=True, format='%Y-%m-%d %H:%M:%S %Z')
    pd.to_datetime(df["closed_date"], utc=True, format='%Y-%m-%d %H:%M:%S %Z')
    pd.to_datetime(df["resolution_action_updated_date"], utc=True, format='%Y-%m-%d %H:%M:%S %Z')
    pd.to_datetime(df["due_date"], utc=True, format='%Y-%m-%d %H:%M:%S %Z')

    # convert (5) numeric cols to specific numeric data types
    df["x_coordinate"] = df["x_coordinate"].fillna(0)
    df["y_coordinate"] = df["y_coordinate"].fillna(0)
    df["bbl"] = df["bbl"].fillna(0)
    df["latitude"] = df["latitude"].fillna(0)
    df["longitude"] = df["longitude"].fillna(0)

    df["x_coordinate"] = pd.to_numeric(df["x_coordinate"], downcast='signed')
    df["y_coordinate"] = pd.to_numeric(df["y_coordinate"], downcast='signed')
    df["bbl"] = pd.to_numeric(df["bbl"], downcast='unsigned')
    df["latitude"] = pd.to_numeric(df["latitude"], downcast='signed')
    df["longitude"] = pd.to_numeric(df["longitude"], downcast='signed')

    # unprocessed cols
    # incident_zip
    # incident_address
    # cross_street_1
    # cross_street_2
    # intersection_street_1
    # intersection_street_2
    # borough
    # location
    # unique_key

    df.to_pickle("data1.pkl")


def retrieve_first_data_after_cleaning():
    df = pd.read_pickle("data1.pkl")
    mem_usage_s = df.memory_usage(deep=True)


if __name__ == '__main__':
    retrieve_first_data_after_cleaning()

