🚀 Azure MLOps: Predictive Maintenance API
Technical Focus: Machine Learning Engineering, Cloud Infrastructure (Azure), MLOps, Unit Testing.

📌 Project Overview
Developed a production-ready machine learning pipeline to predict the Remaining Useful Life (RUL) of aircraft engines using the NASA CMAPSS dataset. This project demonstrates a full MLOps lifecycle: from local feature engineering and unit testing to cloud-native deployment via Azure Machine Learning Managed Endpoints.

🛠️ Tech Stack & Skills
Cloud: Microsoft Azure (ML Workspace, Managed Online Endpoints, Model Registry).

Languages & Frameworks: Python, Scikit-Learn, Pandas, NumPy.

Infrastructure as Code (IaC): YAML (declarative environment and deployment definitions).

DevOps/QA: Pytest (Unit Testing), Virtual Environments, Git.

🏗️ Architecture
The system is designed with a clear separation of concerns to mirror professional software engineering standards:

src/train.py: Handles data ingestion, custom feature engineering, and model serialization using joblib.

src/score.py: The production "driver" script that defines how the Azure container initializes and handles real-time inference requests.

env.yml & dependencies.yml: Declarative definitions for the Docker container environment, ensuring identical execution across local and cloud environments.

provision_cloud.py: An automated deployment script using the Azure ML SDK v2 to provision infrastructure programmatically.

🧪 Engineering Highlights
1. Unit Testing for Data Integrity
Implemented a suite of tests using pytest to validate the RUL calculation logic. This ensures that the core mathematical transformations are correct before the data ever touches the model.

Bash

python -m pytest
2. Declarative Cloud Infrastructure
Utilized YAML-based environment definitions to resolve complex SDK versioning conflicts. By defining the environment as a versioned asset in Azure, I ensured that the Standard_DS2_v2 compute instance has the exact dependencies required for the Scikit-learn framework.

3. Automated Deployment Pipeline
Used the Azure ML SDK to automate:

Model Versioning: Registering models in a central registry for tracking.

Managed Endpoints: Creating a secure, scalable URL for real-time predictions.

📈 Key Results
Model Performance: Achieved a functional baseline for engine degradation prediction using a Random Forest Regressor.

Deployment Efficiency: Reduced infrastructure setup time by ~90% through Python-based automation vs. manual portal configuration.

Code Quality: Maintained a modular structure allowing for easy integration into a CI/CD pipeline (e.g., GitHub Actions).
