import pandas as pd
import os
import sys
import yaml

# Load pramas from yaml.load

params = yaml.safe_load(open('params.yaml'))['preprocess']

def preprocess(input_path,output_path):
    data = pd.read_csv(input_path)

    os.makedirs(os.path.dirname(output_path),exist_ok=True)

    data.to_csv(output_path,header=True,index=False)
    
    print(f"processed file saved to {output_path}")


    
if __name__=='__main__':
    preprocess(params['input'],params['output'])