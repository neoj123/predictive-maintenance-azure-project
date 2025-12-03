import os
import yaml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment, CodeConfiguration

SUBSCRIPTION_ID = "your-subscription-id-here"  # Replace with your Azure Subscription ID
RESOURCE_GROUP = ""
WORKSPACE_NAME = ""
ENDPOINT_NAME = "" 

try:
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
    print(f"Connected to workspace: {WORKSPACE_NAME}")
except Exception as e:
    print(f"Error connecting to Azure. Error: {e}")
    exit()

print("\nCreating/Updating Environment definition...")
custom_env_name = "rul-predictor-env"
conda_config = {
    "name": custom_env_name,
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.10",
        "pip",
    ]
}
conda_file_path = "conda.yml"
with open(conda_file_path, "w") as f:
    yaml.dump(conda_config, f)

# Create Environment Object
deployment_environment = Environment(
    name=custom_env_name,
    description="Custom environment for Scikit-learn RUL predictor",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    conda_file=conda_file_path,
)

# Register and Capture Result 
registered_env = ml_client.environments.create_or_update(deployment_environment)
environment_path = f"azureml:{registered_env.name}:{registered_env.version}"
print(f"Environment registered/updated: {environment_path}")
print("\nRegistering model...")
model = Model(
    path="model/rul_model.pkl",
    name="nasa-rul-predictor",
    description="Predicts engine failure using Random Forest",
    type="custom_model"
)
registered_model = ml_client.models.create_or_update(model)
print(f"Model registered: {registered_model.name} (Version {registered_model.version})")
print(f"\nCreating Endpoint: {ENDPOINT_NAME}...")
endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    description="Real-time RUL prediction",
    auth_mode="key"
)

try:
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Check: Endpoint {ENDPOINT_NAME} exists or created.")
except Exception as e:
    print(f"Warning: Endpoint creation check encountered an issue (might already exist). Continuing to deployment... \nDetails: {e}")

# --- 7. DEPLOYMENT ---
print("Deploying Model (This takes ~10-15 mins)...")

# CHANGE THIS TO v3
deployment_name = "blue-deployment-v1" 

deployment = ManagedOnlineDeployment(
    name=deployment_name, 
    endpoint_name=ENDPOINT_NAME,
    model=registered_model,
    instance_type="Standard_DS2_v2", 
    instance_count=1,
    environment=environment_path,
    code_configuration=CodeConfiguration(
        code="src",          
        scoring_script="score.py", 
    )
)

# Blocking call - this will take time
ml_client.online_deployments.begin_create_or_update(deployment).result()

# --- 8. TRAFFIC ASSIGNMENT (New Step) ---
# Once deployed, we must tell the endpoint to send 100% of traffic to this new deployment
print(f"Assigning 100% traffic to {deployment_name}...")
endpoint.traffic = {deployment_name: 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print("\n\n#####################################################")
print("Done! Your API is live.")
print("#####################################################")