# MLOps Zoomcamp Progress

## Week 1: Introduction to MLOps

### Overview
In Week 1, I focused on learning the fundamentals of MLOps and setting up my machine learning development environment. The primary goal was to understand MLOps concepts and create a baseline prediction model.

### Key Concepts Covered
- Introduction to MLOps (Machine Learning Operations)
- MLOps maturity model and workflow stages
- Development environment setup
- Creating a baseline machine learning model

### Project: NYC Taxi Trip Duration Prediction

I built a linear regression model to predict the duration of taxi trips in New York City using the Yellow Taxi Trip Records dataset. The implementation followed these steps:

1. **Data Loading & Exploration**
   - Downloaded Yellow Taxi Trip Records for January and February 2023
   - Explored the dataset structure (identified 18 columns in the dataset)
   - Performed initial data analysis

2. **Feature Engineering**
   - Created a trip duration feature by calculating the difference between pickup and dropoff times
   - Calculated standard deviation of trip durations
   - Filtered outliers to keep only trips between 1-60 minutes
   - Applied one-hot encoding to pickup and dropoff location IDs

3. **Model Building**
   - Trained a linear regression model with default parameters
   - Used the DictVectorizer for feature transformation
   - Evaluated model performance using RMSE on training data

4. **Validation**
   - Applied the model to February 2023 data
   - Evaluated performance using RMSE on validation data

### Technical Implementation
- Used pandas for data manipulation and preprocessing
- Implemented scikit-learn's LinearRegression and DictVectorizer
- Built data pipelines for consistent preprocessing
- Used matplotlib and seaborn for data visualization

## Week 2: Experiment tracking and model management with MLflow  



### Resources
- [MLOps Zoomcamp GitHub Repository](https://github.com/DataTalksClub/mlops-zoomcamp)
- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)