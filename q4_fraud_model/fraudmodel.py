import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def predict(test_data):
    print("Starting data cleaning and feature engineering....")
    clean_test_data = process_data(test_data)
    print("Data cleaning and feature engineering done!")

    Y_test = clean_test_data["isFraud"]
    clean_test_data = clean_test_data.drop(columns=["isFraud"])
    X_test = clean_test_data

    print("Unpacking xgboost model....")
    trained_model = pickle.load(open(os.path.join(__location__, "trained_fraud_model_4.pkl"), 'rb'))
    print("Model loaded!")

    print("Predicting on your features and returning the accuracy...")
    result = trained_model.score(X_test, Y_test)
    print("Accuracy returned!")

    return {"accuracy": result}


def process_data(df):
    # Drop this col if it exists (its an index col)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Drop columns with only null values
    df = df.drop(columns=["echoBuffer", "merchantCity", "merchantState", "merchantZip", "posOnPremises", "echoBuffer",
                          "recurringAuthInd"])

    # Convert bool cols to 0s and 1s
    df["cardPresent"] = df["cardPresent"].astype('int')
    df["expirationDateKeyInMatch"] = df["expirationDateKeyInMatch"].astype('int')
    df["isFraud"] = df["isFraud"].astype('int')

    # Convert categorical columns into category type to speed up processing time
    df["merchantName"] = df["merchantName"].astype('category')
    df["acqCountry"] = df["acqCountry"].astype('category')
    df["merchantCountryCode"] = df["merchantCountryCode"].astype('category')
    df["merchantCategoryCode"] = df["merchantCategoryCode"].astype('category')
    df["transactionType"] = df["transactionType"].astype('category')

    # Transform date columns into separate features
    df["transactionDateTime"] = pd.to_datetime(df["transactionDateTime"], format='%Y-%m-%dT%H:%M:%S')
    df["accountOpenDate"] = pd.to_datetime(df["accountOpenDate"], format='%Y-%m-%d')
    df["dateOfLastAddressChange"] = pd.to_datetime(df["dateOfLastAddressChange"], format='%Y-%m-%d')

    df["lastAddressChangeYear"] = df["dateOfLastAddressChange"].dt.year
    df["lastAddressChangeMonth"] = df["dateOfLastAddressChange"].dt.month
    df["lastAddressChangeDay"] = df["dateOfLastAddressChange"].dt.day

    df["transactionYear"] = df["transactionDateTime"].dt.year
    df["transactionMonth"] = df["transactionDateTime"].dt.month
    df["transactionDay"] = df["transactionDateTime"].dt.day
    df["transactionHour"] = df["transactionDateTime"].dt.hour
    df["transactionMinute"] = df["transactionDateTime"].dt.minute
    df["transactionSecond"] = df["transactionDateTime"].dt.second

    df["accountOpenYear"] = df["accountOpenDate"].dt.year
    df["accountOpenMonth"] = df["accountOpenDate"].dt.month

    split_currentExp = df["currentExpDate"].str.split("/", n=1, expand=True)
    df["currentExpMonth"] = pd.to_numeric(split_currentExp[0])
    df["currentExpYear"] = pd.to_numeric(split_currentExp[1])

    df = df.drop(columns=["dateOfLastAddressChange", "accountOpenDate", "transactionDateTime", "currentExpDate"])

    # One hot encode categorical variables
    one_hot_cols = ["transactionType", "merchantCountryCode", "merchantCategoryCode", "acqCountry", "merchantName"]

    for col in one_hot_cols:
        mapping = {}
        unq_vals = list(df[col].unique())
        for i in range(len(unq_vals)):
            mapping[unq_vals[i]] = i
        df = df.replace({col: mapping})
        df[col] = pd.to_numeric(df[col], downcast="unsigned")

    df.to_csv("fe_results.csv", index=False)
    upload_csv_to_gcs()

    return df


# this function is where we get the file trained_fraud_model_4.pkl
def train_fraud_model():

    df = pd.read_pickle("../clean_fraud_data.pkl")

    pd.set_option("display.max_columns", 500)
    print(df.head())
    print(df.dtypes)
    print(df.columns)

    df = df.sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.2)
    Y_train = train["isFraud"]
    X_train = train.drop(columns=["isFraud"])
    Y_test = test["isFraud"]
    X_test = test.drop(columns=["isFraud"])

    Y_train.to_pickle("Y_train_data.pkl")
    X_train.to_pickle("X_train_data.pkl")
    Y_test.to_pickle("Y_test_data.pkl")
    X_test.to_pickle("X_test_data.pkl")

    EPOCHS = 5
    kf = StratifiedKFold(n_splits=EPOCHS, shuffle=True)
    y_preds = np.zeros(Y_test.shape[0])
    y_oof = np.zeros(X_train.shape[0])
    count = 0
    for tr_idx, val_idx in kf.split(X_train, Y_train):
        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=9,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            missing=-999,
            random_state=2019,
            tree_method='auto',
            n_jobs=-1

        )

        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = Y_train.iloc[tr_idx], Y_train.iloc[val_idx]

        clf.fit(X_tr, y_tr)

        y_pred_train = clf.predict_proba(X_vl)[:, 1]
        print("y_pred_train")
        print(y_pred_train)

        y_oof[val_idx] = y_pred_train

        print('ROC AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))

        y_preds += clf.predict_proba(X_test)[:, 1] / EPOCHS

        print("clf predict proba x_test")
        print(clf.predict_proba(X_test))
        print(clf.predict_proba(X_test)[:, 1])
        print(clf.predict_proba(X_test)[:, 1] / EPOCHS)

        pickle.dump(clf, open("trained_fraud_model_" + str(count) + ".pkl", 'wb'))
        count += 1


def experimental_clean_data():
    df = pd.read_csv("../ffg_fraud_model_assigment.csv")
    df = df.drop(columns=["Unnamed: 0"])
    df = df.drop(columns=["echoBuffer", "merchantCity", "merchantState", "merchantZip", "posOnPremises", "echoBuffer",
                          "recurringAuthInd"])
    df["transactionDateTime"] = pd.to_datetime(df["transactionDateTime"], format='%Y-%m-%dT%H:%M:%S')
    df["accountOpenDate"] = pd.to_datetime(df["accountOpenDate"], format='%Y-%m-%d')
    df["dateOfLastAddressChange"] = pd.to_datetime(df["dateOfLastAddressChange"], format='%Y-%m-%d')

    df["lastAddressChangeYear"] = df["dateOfLastAddressChange"].dt.year
    df["lastAddressChangeMonth"] = df["dateOfLastAddressChange"].dt.month
    df["lastAddressChangeDay"] = df["dateOfLastAddressChange"].dt.day

    df["transactionYear"] = df["transactionDateTime"].dt.year
    df["transactionMonth"] = df["transactionDateTime"].dt.month
    df["transactionDay"] = df["transactionDateTime"].dt.day
    df["transactionHour"] = df["transactionDateTime"].dt.hour
    df["transactionMinute"] = df["transactionDateTime"].dt.minute
    df["transactionSecond"] = df["transactionDateTime"].dt.second

    df["accountOpenYear"] = df["accountOpenDate"].dt.year
    df["accountOpenMonth"] = df["accountOpenDate"].dt.month

    split_currentExp = df["currentExpDate"].str.split("/", n=1, expand=True)
    df["currentExpMonth"] = pd.to_numeric(split_currentExp[0])
    df["currentExpYear"] = pd.to_numeric(split_currentExp[1])

    df["merchantName"] = df["merchantName"].astype('category')
    df["acqCountry"] = df["acqCountry"].astype('category')
    df["merchantCountryCode"] = df["merchantCountryCode"].astype('category')
    df["merchantCategoryCode"] = df["merchantCategoryCode"].astype('category')
    df["transactionType"] = df["transactionType"].astype('category')

    df = df.drop(columns=["dateOfLastAddressChange", "accountOpenDate", "transactionDateTime", "currentExpDate"])

    one_hot_cols = ["transactionType", "merchantCountryCode", "merchantCategoryCode", "acqCountry", "merchantName"]

    df["cardPresent"] = df["cardPresent"].astype('int')
    df["expirationDateKeyInMatch"] = df["expirationDateKeyInMatch"].astype('int')
    df["isFraud"] = df["isFraud"].astype('int')

    for col in one_hot_cols:
        mapping = {}
        unq_vals = list(df[col].unique())
        for i in range(len(unq_vals)):
            mapping[unq_vals[i]] = i
        df = df.replace({col: mapping})
        df[col] = pd.to_numeric(df[col], downcast="unsigned")

    df.to_pickle("clean_fraud_data.pkl")


def read_clean_data():
    df = pd.read_pickle("/dev_data_not_important/clean_fraud_data.pkl")
    print(df.columns)


if __name__ == "__main__":
    read_clean_data()