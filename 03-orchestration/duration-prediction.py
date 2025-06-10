import pickle
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import mlflow
import mlflow.sklearn
from pathlib import Path
from prefect import flow, task

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

@task
def load_taxi_data(year: int, month: int):
    """Load taxi data"""
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    
    print(f"Loading data from: {url}")
    df = pd.read_parquet(url)
    
    print(f"Total records loaded: {len(df)}")
    
    return df

@task
def prepare_data(df):
    """Data preparation"""
    
    # Apply the exact logic from the homework
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df.duration = df.duration.dt.total_seconds() / 60
    
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    print(f"Records after preparation: {len(df)}")
    
    return df

@task
def train_linear_model(df):
    """Train linear regression model"""
    
    # Prepare features - use locations separately (not combined)
    categorical = ['PULocationID', 'DOLocationID']  # Separate, not combined
    numerical = ['trip_distance']
    
    # Create feature dictionaries
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    # Fit DictVectorizer
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(dicts)
    
    # Target variable
    y = df['duration'].values
    
    # Train Linear Regression with default parameters
    lr = LinearRegression()
    lr.fit(X, y)
    
    print(f"Model intercept: {lr.intercept_:.2f}")
    
    return dv, lr, X, y

@task
def register_model_mlflow(dv, lr, X, y):
    """Register model with MLflow"""
    
    with mlflow.start_run() as run:
        # Log model parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", "PULocationID, DOLocationID, trip_distance")
        
        # Log metrics
        y_pred = lr.predict(X)
        rmse = root_mean_squared_error(y, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("intercept", lr.intercept_)
        
        # Log the DictVectorizer
        with open("models/dict_vectorizer.pkl", "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact("models/dict_vectorizer.pkl")
        
        # Log the Linear Regression model
        mlflow.sklearn.log_model(
            lr, 
            "linear_regression_model",
            registered_model_name="nyc_taxi_linear_regression"
        )
        
        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")
        
        return run_id

@flow(name="NYC Taxi Complete Pipeline")
def main_flow():
    """Complete pipeline for homework"""
    print("Starting NYC Taxi Pipeline with Prefect\n")
    
    # Create models directory
    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)
    
    # Load data
    df = load_taxi_data(2023, 3)  # March 2023
    
    # Prepare data
    df_prepared = prepare_data(df)
    
    # Train linear model
    dv, lr, X, y = train_linear_model(df_prepared)
    
    # Register model
    run_id = register_model_mlflow(dv, lr, X, y)
    
    print(f"\nPipeline completed! MLflow run_id: {run_id}")
    
    return run_id

if __name__ == "__main__":
    main_flow()