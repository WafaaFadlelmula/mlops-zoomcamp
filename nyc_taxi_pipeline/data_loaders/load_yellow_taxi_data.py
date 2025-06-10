import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def load_yellow_taxi_data(*args, **kwargs):
    """
    Load March 2023 Yellow taxi data - This will answer Question 3!
    """
    year = 2023
    month = 3
    
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    
    print(f"Loading data from: {url}")
    df = pd.read_parquet(url)
    
    # THIS IS THE ANSWER TO QUESTION 3!
    print(f"*** QUESTION 3 ANSWER: Loaded {len(df)} records from Yellow taxi data ***")
    
    # Process the data
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    print(f"After filtering: {len(df)} records")
    return df