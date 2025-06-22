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

### Overview
Week 2 focused on implementing experiment tracking and model management using MLflow. I learned how to track experiments, manage models, perform hyperparameter optimization, and use model registries to manage the complete ML model lifecycle.

### Key Concepts Covered
* MLflow fundamentals and installation
* Experiment tracking and logging
* Model autologging capabilities
* MLflow tracking server setup
* Hyperparameter optimization with tracking
* Model registry and model promotion workflows
* Backend stores and artifact management

### Project: Enhanced NYC Taxi Trip Duration Prediction with MLflow

Building upon Week 1's linear regression model, I implemented comprehensive experiment tracking using MLflow with the Green Taxi Trip Records dataset.

#### 1. MLflow Setup and Installation
* **Environment Setup**: Created a dedicated Python environment for MLflow
* **Installation**: Installed MLflow Python package using pip/conda
* **Version Verification**: Confirmed MLflow installation and version

#### 2. Data Pipeline Enhancement
* **Dataset**: Used Green Taxi Trip Records for January, February, and March 2023 (parquet format)
* **Preprocessing Script**: Used `preprocess_data.py` to:
  - Load data from specified taxi data folder
  - Fit DictVectorizer on training set (January 2023)
  - Save preprocessed datasets and DictVectorizer to disk
* **Output Management**: Generated multiple preprocessed files for downstream tasks

#### 3. Model Training with Autologging
* **Model**: RandomForestRegressor from Scikit-Learn
* **Training Script**: Used `train.py` with MLflow integration:
  - Enabled MLflow autologging
  - Wrapped training code with `mlflow.start_run()` context
  - Tracked RMSE scores on validation set
  - Logged model parameters automatically
* **Parameter Tracking**: Monitored key parameters like `min_samples_split`

#### 4. MLflow Tracking Server Configuration
* **Local Server Setup**: Launched MLflow tracking server locally
* **Backend Configuration**: 
  - Used SQLite database for backend store
  - Configured artifacts folder for artifact storage
  - Set up proper server parameters including `default-artifact-root`
* **Model Registry Access**: Enabled model registry functionality through server setup

#### 5. Hyperparameter Optimization with Tracking
* **Optimization Framework**: hyperparameter tuning using hyperopt
* **Enhanced HPO Script**: Modified `hpo.py` to:
  - Log validation RMSE for each optimization run
  - Track hyperparameter combinations
  - Store results in "random-forest-hyperopt" experiment
  - Record optimization progress without autologging
* **Performance Monitoring**: Tracked validation RMSE across multiple hyperparameter configurations

#### 6. Model Registry and Promotion
* **Model Selection**: Used `register_model.py` to:
  - Identify top 5 performing models from hyperparameter optimization
  - Evaluate selected models on test set (March 2023 data)
  - Calculate test RMSE for model comparison
* **Model Registration**: 
  - Used `mlflow.register_model()` for model registration
  - Implemented proper model URI formatting: `"runs:/<RUN_ID>/model"`
  - Created "random-forest-best-models" experiment for final evaluation
* **Best Model Promotion**: Selected and registered the model with lowest test RMSE

### Technical Implementation Highlights

#### MLflow Components Used
* **Tracking**: Experiment and run logging
* **Models**: Model packaging and versioning  
* **Model Registry**: Model lifecycle management
* **UI**: Web-based experiment exploration

#### Key MLflow Features Implemented
* Automatic parameter and metric logging
* Manual logging for hyperparameter optimization
* Model artifact storage and retrieval
* Experiment organization and comparison
* Model registration and versioning

#### Integration with ML Pipeline
* **Data Preprocessing**: Tracked data transformation steps
* **Model Training**: Logged training metrics and parameters
* **Validation**: Recorded validation performance
* **Testing**: Evaluated final model performance
* **Deployment Preparation**: Registered production-ready model

### Skills Developed
* MLflow installation and configuration
* Experiment tracking best practices
* Model lifecycle management
* Hyperparameter optimization tracking
* Model registry workflows
* Local tracking server setup and management
* Integration of MLflow with scikit-learn models

### Key Learnings
* **Experiment Organization**: Proper structuring of ML experiments for reproducibility
* **Model Versioning**: Managing different model versions through the registry
* **Performance Tracking**: Systematic logging of model performance metrics
* **Hyperparameter Management**: Efficient tracking of optimization experiments
* **Production Readiness**: Preparing models for deployment through proper registration

### Technical Stack
* **MLflow**: Experiment tracking and model management
* **Hyperopt**: Hyperparameter optimization
* **Scikit-learn**: RandomForestRegressor implementation
* **Pandas**: Data manipulation and preprocessing  
* **SQLite**: Backend storage for tracking server
* **Python**: Core programming language


## Week 3: Orchestration and ML Pipelines
### Overview
Week 3 focused on transforming ML experiments into production-ready pipelines using orchestration tools. The goal was to convert notebook-based workflows into automated, scheduled, and parameterized ML pipelines.

### Key Concepts Covered

* Introduction to ML Pipelines and orchestration
* Converting Jupyter notebooks to Python scripts
* Orchestration tools comparison and selection
* Pipeline parameterization and scheduling
* Workflow backfilling and deployment
* Production pipeline best practices

### Project: Orchestrated ML Pipeline Implementation
**1. ML Pipeline Introduction**

Understanding the need for ML pipeline orchestration
Identifying pipeline components and dependencies
Designing workflow structure for taxi trip duration prediction

**2. Notebook to Script Conversion**

Converted existing Jupyter notebooks to modular Python scripts
Separated data preprocessing, model training, and evaluation logic
Created reusable functions for each pipeline step
Implemented proper error handling and logging

**3. Orchestrator Selection and Setup**

Evaluated orchestration tools (Airflow, Prefect, Dagster, Kestra, Mage)
Selected [chosen orchestrator] based on project requirements
Configured local development environment
Implemented and tested "Hello World" workflow

**4. Pipeline Orchestration**

Created orchestrated workflow from previous week's code
Defined pipeline steps:

Data extraction and preprocessing
Model training with MLflow tracking
Model validation and evaluation
Model registration and promotion


Implemented proper task dependencies and error handling

**5. Workflow Parameterization**

Configured monthly pipeline scheduling
Implemented dynamic data loading:

Training data: Two months ago
Validation data: One month ago


Added configurable parameters for model hyperparameters
Created flexible date-based data partitioning

**6. Backfilling Implementation**

Developed backfill capabilities for historical data processing
Implemented date range processing for past months
Created batch processing for multiple time periods
Added validation for backfill operations


### Technical Implementation
#### Pipeline Architecture

Modular script design with clear separation of concerns
Parameterized workflows for flexible execution
Robust error handling and recovery mechanisms
Comprehensive logging and monitoring

#### Orchestration Features

Scheduled execution with cron-like scheduling
Dynamic parameter passing between tasks
Conditional execution based on data availability
Retry mechanisms for failed tasks

#### Data Management

Automated data ingestion from multiple sources
Data quality validation and preprocessing
Versioned data artifacts with proper lineage
Efficient data storage and retrieval

#### Skills Developed

Pipeline design and orchestration
Production-ready code development
Workflow scheduling and automation
Container orchestration with Docker
Backfill and historical data processing
Production MLflow deployment

#### Key Learnings

Pipeline Orchestration: Converting ad-hoc ML experiments into production workflows
Automation: Scheduling and parameterizing ML pipelines for regular execution
Scalability: Designing pipelines that can handle varying data volumes and time periods
Reliability: Implementing robust error handling and recovery mechanisms
Deployment: Containerizing ML services for consistent deployment

#### Technical Stack

Perfect: Pipeline orchestration and scheduling
Docker: Containerization and service deployment
MLflow: Continued experiment tracking and model management
Python: Core pipeline development
SQL/Database: Data storage and retrieval
YAML/JSON: Configuration management

## Week 4: Model Deployment

### Overview
Week 4 focused on deploying machine learning models to production using different deployment strategies. The goal was to learn various deployment approaches including web services, streaming, and batch processing.

### Key Concepts Covered
- Web services, streaming, and batch processing strategies
- Model serving with Flask and Docker
- Integration with MLflow model registry
- Real-time and batch scoring implementations
- Production deployment best practices

### Project: Multi-Modal ML Model Deployment

I deployed machine learning models using multiple strategies, integrating web service APIs, real-time streaming, and batch processing. Each deployment path was tied to realistic use cases and production-level requirements.

#### 1. Deployment Strategy Overview
* **Web Services**: Real-time prediction APIs for interactive applications  
* **Streaming**: Event-driven predictions using real-time data streams  
* **Batch Processing**: Scheduled and large-scale inference for offline workloads  
* Evaluated trade-offs, scalability, and performance needs for each approach

#### 2. Web Service Deployment with Flask and Docker

**Flask Application Development**
- Built REST API endpoints for model predictions
- Implemented input validation, error handling, and logging
- Added health checks and monitoring endpoints

**Docker Containerization**
- Created Dockerfile for Flask application
- Used multi-stage builds to optimize image size
- Managed dependencies and environment variables

**Model Integration**
- Loaded trained models from local storage
- Applied preprocessing pipeline and prediction logic
- Managed versioning and model updates within the app

#### 3. MLflow Model Registry Integration

**Model Retrieval from Registry**
- Connected Flask app to MLflow tracking server
- Loaded models from MLflow model registry
- Implemented version control and automatic updates

**Production Model Serving**
- Served production-stage models via registry
- Integrated model promotion and staging workflows
- Added model monitoring and fallback mechanisms

#### 4. Streaming Deployment (Optional)

**AWS Kinesis and Lambda Integration**
- Configured Kinesis data streams for real-time input
- Triggered Lambda functions for model predictions
- Implemented event-driven architecture and dead letter queues

**Real-time Processing**
- Applied streaming data preprocessing
- Performed real-time inference
- Streamed output to downstream systems and added monitoring

#### 5. Batch Scoring Implementation

**Scoring Script Development**
- Built batch scoring scripts for offline predictions
- Loaded and preprocessed large datasets
- Applied trained model for inference and saved outputs

**Batch Processing with Orchestration**
- Integrated scripts into orchestration tools (e.g., Airflow)
- Managed model retrieval from MLflow registry
- Scheduled batch jobs and implemented monitoring

#### 6. Production Deployment Considerations

**Scalability and Performance**
- Used load balancing and auto-scaling in web services
- Optimized model inference time and added caching
- Allocated compute resources efficiently

**Monitoring and Observability**
- Logged model metrics and service performance
- Configured alerts for failures and performance drops
- Detected model drift and deployed mitigation strategies

### Technical Implementation

#### Web Service Architecture
- Flask-based REST API with middleware and monitoring
- Docker for environment consistency
- Auto-scaling and logging enabled in deployment setup

#### Model Management
- Integrated MLflow registry for versioned models
- Implemented automated updates and rollback mechanisms
- Enabled A/B testing and performance tracking

#### Deployment Strategies
- Supported multiple environments (dev/staging/prod)
- Implemented blue-green and canary deployment
- Planned disaster recovery and backup solutions

### Skills Developed
- Web service development with Flask
- Container orchestration with Docker
- Model serving and REST API creation
- MLflow registry integration
- Streaming and batch deployment techniques
- Production monitoring and scaling

### Key Learnings
- **Deployment Patterns**: When and how to use web, batch, and streaming strategies  
- **Model Serving**: Built scalable, real-time inference systems  
- **Production Readiness**: Integrated observability and version control  
- **System Integration**: Connected ML pipelines to robust production systems  
- **Scalability**: Designed deployments that handle production-scale demands  

### Technical Stack
- **Flask**: Web service framework  
- **Docker**: Containerization for portable deployments  
- **MLflow**: Model registry and tracking  
- **AWS Kinesis & Lambda**: Streaming deployment components (optional)  
- **Orchestration Tool**: Airflow or equivalent for batch processing  
- **Monitoring Tools**: Logging and metric collection frameworks  
- **Python**: Core development language  

### Overall Progress Summary

#### Completed Modules
✅ Week 1: Introduction to MLOps and baseline model development  
✅ Week 2: Experiment tracking and model management with MLflow  
✅ Week 3: Orchestration and ML pipelines  
✅ Week 4: Model deployment strategies  

#### Key Skills Acquired
- End-to-end ML pipeline development  
- Experiment tracking and versioning  
- Production-ready code and APIs  
- Model deployment and serving strategies  
- Workflow orchestration  
- Docker containerization  
- MLflow registry integration  

### Resources
- [MLOps Zoomcamp GitHub Repository](https://github.com/DataTalksClub/mlops-zoomcamp)
- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
