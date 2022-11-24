import json
import numpy as np
import pandas as pd
import os
import torch as T
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
pandas_sample_input = PandasParameterType(pd.DataFrame({'location': ['loc_0', 'loc_0'], 'car_type': ['comfort', 'comfort'], 'hour':[15,16], 'count':[2,3]}))

# This is a nested input sample, any item wrapped by `ParameterType` will be described by schema
sample_input = StandardPythonParameterType({'data': pandas_sample_input})
sample_output = StandardPythonParameterType([0, 1])
outputs = StandardPythonParameterType({'Results':sample_output}) # 'Results' is case sensitive
THRESHOLD = 40
TIME_STEPS = 20
class Autoencoder(T.nn.Module):  # 65-32-8-32-65
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.fc1 = T.nn.Conv2d(1,32,7)
    self.fc2 = T.nn.Conv2d(32,16,7)
    self.fc3 = T.nn.ConvTranspose2d(16,32,7)
    self.fc4 = T.nn.ConvTranspose2d(32,1,7)
    # self.fc5 = T.nn.ConvTranspose2d(32,1,7)

  def encode(self, x):  # 65-32-8
    z = T.tanh(self.fc1(x))
    z = T.tanh(self.fc2(z))  # latent in [-1,+1]
    return z  

  def decode(self, x):  # 8-32-65
    z = T.tanh(self.fc3(x))
    z = T.sigmoid(self.fc4(z))
    # z = T.sigmoid(self.fc5(z))  # [0.0, 1.0]
    return z
    
  def forward(self, x):
    z = self.encode(x) 
    z = self.decode(z) 
    return z  # in [0.0, 1.0]
# Called when the service is loaded
def create_sequences(values, time_steps=TIME_STEPS):
    output = []  
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)
def pre_process(input_df):
       mean = 2.2031823072902032e-17
       std = 1.0
       input_df['count'] = (input_df['count']- mean)/std
       transformed_data= transformer.transform(input_df)
       transformed_input = create_sequences(transformed_data)
       transformed_input = np.expand_dims(transformed_input, 1)
       transformed_input = T.tensor(np.float32(transformed_input), dtype=T.float32).to("cpu")
       return transformed_input
def init():
    global model, transformer
    # Get the path to the deployed model file and load it
    model_dir =os.getenv('AZUREML_MODEL_DIR')
    model_file = os.listdir(model_dir)[0]
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_file)
    model = Autoencoder()
    model.load_state_dict(T.load(model_path))
    model.eval()
    #load transformer
    locations =['loc_0', 'loc_1', 'loc_10', 'loc_11', 'loc_12', 'loc_13', 'loc_14',
        'loc_15', 'loc_16', 'loc_17', 'loc_18', 'loc_19', 'loc_2', 'loc_3',
        'loc_4', 'loc_5', 'loc_6', 'loc_7', 'loc_8', 'loc_9']
    car_types =['comfort', 'green', 'x', 'xl','comfort', 'green', 'x', 'xl','comfort', 'green', 'x', 'xl','comfort', 'green', 'x', 'xl','comfort', 'green', 'x', 'xl']

    transformer = make_column_transformer(
        (OneHotEncoder(sparse=False), ['location', 'car_type']),
        remainder='passthrough')
    transformer.fit(pd.DataFrame({"location":locations, "car_type":car_types, "count":range(20)}))

def score(model, input, threshold ):
    transformed_data = pre_process(input)
    Y = model(transformed_data)  # should be same as X
    errs = T.sum((transformed_data-Y)*(transformed_data-Y), dim=[1,2,3]).detach().numpy().tolist()  #
    anomalies = [int(err>threshold) for err in errs]
    anomalous_data_indices = []
    for data_idx in range(TIME_STEPS - 1, len(transformed_data) - TIME_STEPS + 1):
        if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)
    anomalies =np.array([-1]*input.shape[0])
    anomalies[anomalous_data_indices] =1
    return anomalies.tolist()

# Called when a request is received
@input_schema('Inputs', sample_input) 
# 'Inputs' is case sensitive
@output_schema(outputs)

def run(Inputs):
    try:
        # Get the input data 
        data=Inputs['data']
        # Get a prediction from the model
        
        predictions = score(model, data,THRESHOLD)
        return predictions
    except Exception as e:
        error= str(e)
        return error
