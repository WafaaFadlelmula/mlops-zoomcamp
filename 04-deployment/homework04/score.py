import os
import numpy as np
import pickle
import pandas as pd
import argparse

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    args = parser.parse_args()

    year = args.year
    month = args.month

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    data_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(data_url)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    std_pred = np.std(y_pred)
    print(f"Standard deviation of predictions: {std_pred:.2f}")
    
    mean = np.mean(y_pred)
    print(f"Mean of predictions: {mean:.2f}")
    

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['pred'] = y_pred

    output_file = 'predictions.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    size_mb = round(os.path.getsize(output_file) / 1024 / 1024)
    print(f"Output file size: {size_mb} MB")

if __name__ == '__main__':
    main()
