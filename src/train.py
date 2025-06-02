import mlflow
import pandas as pd
import numpy as np
import pickle
import yaml
import sys
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
from urllib.parse import urlparse


load_dotenv()
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_TRACKING_USERNAME= os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD= os.getenv('MLFLOW_TRACKING_PASSWORD')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# PARAMS
params = yaml.safe_load(open('params.yaml'))['train']

def hyperparameter_tuning(X_train,y_train,param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid, cv=5,n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search



def train(data_path,model_path):
    data=pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run():
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        signature = infer_signature(X_train,y_train)
        param_grid = {
            'n_estimators':[100,200],
            'max_depth':[5,10,None],
            'min_samples_split':[2,5,10],
            'min_samples_leaf':[1,2]
        }
        # perform Hyperparameter TUning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Predict and Evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_pred, y_test)
        print("Accuracy: ",accuracy_score)
        
        #log the metrics
        mlflow.log_metric('accuracy: ',accuracy)
        mlflow.log_param('best_n_estimators',grid_search.best_estimator_.n_estimators)
        mlflow.log_metric('best_max_depth: ',grid_search.best_estimator_.max_depth)
        mlflow.log_metric('best_min_samples_split: ',grid_search.best_estimator_.min_samples_split)
        mlflow.log_metric('best_min_samples_leaf: ',grid_search.best_estimator_.min_samples_leaf)


        # Log the confusion and classification report
        cm = confusion_matrix(y_test,y_pred)
        cr = classification_report(y_test,y_pred)
        mlflow.log_text(str(cm),'confusionMetrix.txt')
        mlflow.log_text(cr,'classificationReport.txt')

        # Tracking URI TypeStore
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="models",
                registered_model_name='Best_Model_v1',
                input_example=X_train.iloc[:20],
                signature=signature
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path='models',
                input_example=X_train.iloc[:5],
                signature=signature
            )

    # Create directory to save the model
    os.makedirs(os.path.dirname(model_path),exist_ok=True)
    filename = model_path
    pickle.dump(best_model,open(filename,'wb'))
    print(f'Model saved to {model_path}')



if __name__ == '__main__':
    train(params['data'],params['model'])

