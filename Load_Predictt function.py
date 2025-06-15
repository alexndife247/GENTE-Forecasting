import pandas as pd
import json
import requests
from datetime import date, datetime, timedelta
from requests.auth import HTTPBasicAuth
from dateutil import parser
import numpy as np
import holidays
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib
import tensorflow
import pytz

def Predict_Load(tag, startdate, enddate):
    def read_data(dt_start, resolution='H', dt_end=''):
        if dt_end == '':
            dt_end = dt_start

        Timestamp1 = dt_end.strftime("%d-%b-%Y %H:%M:%S")
        Timestamp1 = Timestamp1.replace(' ', '%20')
        Timestamp2 = dt_start.strftime("%d-%b-%Y %H:%M:%S")
        Timestamp2 = Timestamp2.replace(' ', '%20')
        url = 'https://hll-api.livinglab.chalmers.se:3001/api/keyqueries?key=$2a$06$ebFvep2QAJa1uQ0OMWvmA.oV5VZP8uso08IgJ30rmKv0W.82JAx1m&start=' + Timestamp2 + '&stop=' + Timestamp1
        response = requests.get(url)
        data = pd.DataFrame(response.json(),
                            columns=['lo_time', 'hi_time', 'avg_pload', 'avg_pvp', 'avg_ulq1', 'avg_ulq2', 'avg_ulq3',
                                     'timestamp'])
        df = data.loc[:, ['lo_time', 'avg_pload']]
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

        # Set the timestamp as the index to be able to use resample for datetime dfs
        df.set_index("lo_time", inplace=True)

        # Resample the data by the resolution and the mean
        df = df.resample(resolution).mean()

        # Handle NaN values: forward fill, then backward fill as fallback
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')

        # convert the unit to kW (BES is in W, the rest are in kW)
        df.loc[:, "avg_pload"] = df.loc[:, "avg_pload"]
        df = df.loc[dt_start:dt_end]
        df.index.name = 'timestamp'

        return df

    # --------------------------Main part of program-------------------------------------

    # prediction interval
    dt_start = datetime.strptime(startdate, "%Y-%m-%d %H:%M%z")
    dt_end = datetime.strptime(enddate, "%Y-%m-%d %H:%M%z")

    # reading 24 hours before historical data
    yesterday_start = dt_start - timedelta(seconds=3600)
    yesterday_end = dt_end - timedelta(seconds=3600)

    Yestreday_df = read_data(dt_start=yesterday_start, resolution='H', dt_end=yesterday_end)
    print(Yestreday_df)

    # reading 168 hours before historical data (1 week)
    LastWeek_start = dt_start - timedelta(days=1)
    LastWeek_end = dt_end - timedelta(days=1)

    LastWeek_df = read_data(dt_start=LastWeek_start, resolution='H', dt_end=LastWeek_end)
    print(LastWeek_df)

    Load_Yes_ave = Yestreday_df.copy()
    Load_LastWeek_ave = LastWeek_df.copy()

    # Making a data frame with timestamps of forecast horizon
    DataAsli = pd.DataFrame()
    DataAsli['timestamp'] = pd.date_range(start=dt_start, end=dt_end, freq='H')

    # Set the timestamp as the index
    xx = DataAsli.copy()
    DataAsli.set_index("timestamp", inplace=True)

    # Extract calendar-based features
    UTCstart = dt_start.astimezone(pytz.UTC)
    UTCend = dt_end.astimezone(pytz.UTC)
    xx['timestamp'] = pd.date_range(start=UTCstart, end=UTCend, freq='H')
    xx.set_index("timestamp", inplace=True)

    DataAsli['hour'] = xx.index.hour
    DataAsli['dayofweek'] = DataAsli.index.weekday
    DataAsli['date'] = DataAsli.index.date

    Swe_hol = holidays.Sweden(years=[2019, 2024])
    DataAsli['is_holiday'] = [date in Swe_hol for date in DataAsli['date']]

    DataAsli['Load1'] = Load_Yes_ave.values / 1000
    DataAsli['Load24'] = Load_LastWeek_ave.values / 1000

    data = DataAsli.copy()
    DataSet = data[['hour', 'is_holiday', 'dayofweek', 'Load1', 'Load24']]
    print(DataSet)

    # Prepare the test data
    DataTest = DataSet
    DataTest = DataSet[0:len(DataTest)]
    X_test = DataTest[['hour', 'is_holiday', 'dayofweek', 'Load1', 'Load24']]

    # Load the scaler for predictors (handle missing file scenario)
    try:
        sc1 = joblib.load('scaler_predictors_' + tag + '.save')
    except FileNotFoundError:
        print(f"Error: 'scaler_predictors_{tag}.save' file not found. Please check the file path.")
        return None

    # Scale the dataset
    dataset = sc1.transform(DataSet)

    # Scale the test data
    DataTest_Scal = dataset[0:len(DataTest)]
    X_test = DataTest_Scal
    X_test = tensorflow.expand_dims(X_test, axis=1)

    # Load the trained model
    try:
        model = load_model('trainedANN' + tag + '.h5')
    except FileNotFoundError:
        print(f"Error: 'trainedANN_{tag}.h5' file not found. Please check the file path.")
        return None

    # Predict with the loaded model
    yhat_te = model.predict(X_test)

    # Unscale with loaded scale metrics for Load (handle missing file scenario)
    try:
        sc3 = joblib.load('Scalar_Load_' + tag + '.save')
    except FileNotFoundError:
        print(f"Error: 'Scalar_Load_{tag}.save' file not found. Please check the file path.")
        return None

    Y_Predicted = sc3.inverse_transform(yhat_te)

    # Load the statistics of the prediction and pass it to output
    try:
        stats = joblib.load('stats_' + tag + '.save')
    except FileNotFoundError:
        print(f"Error: 'stats_{tag}.save' file not found. Please check the file path.")
        return None

    mean_temp = stats['mean'].loc[0]
    std_temp = stats['standard deviation'].loc[0]

    # Create dataframe with the predicted values and put the statistical metrics as well
    predicted = pd.DataFrame(Y_Predicted, columns=['predictedValue'])
    predicted['timestamp'] = pd.date_range(start=dt_start, end=dt_end, freq='H')

    predicted.set_index('timestamp', inplace=True)

    for i in predicted.index:
        predicted.loc[i, 'mean'] = mean_temp
        predicted.loc[i, 'standardDeviation'] = std_temp

    predicted.to_csv('Predicted_Load_' + tag + '.csv')

    return predicted


# Example of how to run the function
tag = 'HSBLL_Load'
startdate = "2024-12-08 20:00+0000"
enddate = "2024-12-09 19:59+0000"

PredictedLoad = Predict_Load(tag, startdate, enddate)
if PredictedLoad is not None:
    print(PredictedLoad)
    output_file = 'PredictedLoad.csv'
    PredictedLoad.to_csv(output_file, index=True)
