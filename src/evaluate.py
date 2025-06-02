import yaml
import mlflow
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os


# Now use the environment variable
load_dotenv()
MLFLOW_TRACKING_URI =  os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')


params = yaml.safe_load(open('params.yaml'))['evaluate']

def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X=data.drop(['Outcome'],axis=1)
    y=data['Outcome']
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load the model from the disk

    model = pickle.load(open(model_path, 'rb'))
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # Log metrics to MLFlow
    mlflow.log_metric('accuracy', accuracy)
    print('Model Accuracy: ' ,accuracy)
if __name__ == "__main__":
    evaluate(params['data'],params['model'])