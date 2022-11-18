import json
import numpy as np
import pandas as pd
import os
import mlflow
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
pandas_sample_input = PandasParameterType(pd.DataFrame({'location': ['loc_0', 'loc_0'], 'car_type': ['comfort', 'comfort'], 'hour':[15,16], 'count':[2,3]}))

# This is a nested input sample, any item wrapped by `ParameterType` will be described by schema
sample_input = StandardPythonParameterType({'data': pandas_sample_input})
sample_output = StandardPythonParameterType([1, -1])
outputs = StandardPythonParameterType({'Results':sample_output}) # 'Results' is case sensitive


# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_dir =os.getenv('AZUREML_MODEL_DIR')
    model_file = os.listdir(model_dir)[0]
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_file)
    model = mlflow.sklearn.load_model(model_path)
# Called when a request is received
@input_schema('Inputs', sample_input) 
# 'Inputs' is case sensitive
@output_schema(outputs)

def run(Inputs):
    try:
        # Get the input data 
        data=Inputs['data']
        # Get a prediction from the model
        predictions = model.predict(data)
        return predictions.tolist()
    except Exception as e:
        error= str(e)
        return error
