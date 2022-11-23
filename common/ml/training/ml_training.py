import pandas as pd
import numpy as np
import os
import argparse
import mlflow
import mlflow.sklearn
from azureml.core import Run, Dataset,Datastore, Workspace
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import torch as T
import joblib

def parse_args():
    # arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--prep_data", default="../../client/", type=str, help="Path to prepped data, default to local folder")
    parser.add_argument("--model_folder", type=str,default="../model/", help="Path of model ouput folder, default to local folder")
    parser.add_argument("--input_file_name", type=str, default="simulated_demand.csv")
    parser.add_argument("--run_mode", type=str, default="local")
    # parse args
    args = parser.parse_args()
    # return args
    return args


#TIME_STEPS = 20
# Generated training sequences for use in the model.
def create_sequences(values, time_steps=20):
    output = []  
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

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




  

def make_err_list(model, ds):
  # assumes model.eval()
  result_lst = []
  n_features = len(ds[0])  # 65
  for i in range(len(ds)):
    X = ds[i]
    with T.no_grad():
      Y = model(X)  # should be same as X
    err = T.sum((X-Y)*(X-Y)).item()  # SSE all features
    err = err / n_features           # sort of norm'ed SSE 
    result_lst.append( (i,err) )     # idx of data item, err
  return result_lst 


def train(ae, ds, bs, me, le, lr):
  # autoencoder, dataset, batch_size, max_epochs,
  # log_every, learn_rate
  # assumes ae.train() has been set
      data_ldr = T.utils.data.DataLoader(ds, batch_size=bs,shuffle=True)
      loss_func = T.nn.MSELoss()
      opt = T.optim.SGD(ae.parameters(), lr=lr)
      print("\nStarting training")
      for epoch in range(0, me):
          epoch_loss = 0.0
          for (batch_idx, batch) in enumerate(data_ldr):
              X = batch  # inputs
              Y = batch  # targets (same as inputs)

              opt.zero_grad()                # prepare gradients
              oupt = ae(X)                   # compute output/target
              # print("oupt shape", oupt.shape)
              loss_val = loss_func(oupt, Y)  # a tensor
              epoch_loss += loss_val.item()  # accumulate for display
              loss_val.backward()            # compute gradients
              opt.step()                     # update weights

          if epoch % le == 0:
              print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))
          #os.makedirs("models", exist_ok=True)
          #T.save(ae.state_dict(), args.model_folder)



# -----------------------------------------------------------

def main(args):

    
    device = "cpu"
    demand_data = pd.read_csv(os.path.join(args.prep_data,args.input_file_name))
    demand_data.starttime = pd.to_datetime(demand_data.starttime)
    demand_data=demand_data.groupby(["location", "car_type"]).resample("5min", on="starttime")["id"].count().reset_index(name="count")

    mean =demand_data['count'].mean()
    std = demand_data['count'].std()
    demand_data['count'] = (demand_data['count']- mean)/std
    demand_data_encoded = pd.get_dummies(demand_data, columns=['location', 'car_type'])
    demand_data_encoded =demand_data_encoded.drop("starttime", axis=1)
    time_steps=20
    x_train = create_sequences(demand_data_encoded.values, time_steps)
    # x_train= np.float32(x_train)
    x_train = np.expand_dims(x_train, 1)
    x_train = T.tensor(np.float32(x_train), dtype=T.float32).to(device) 
    # 0. get started
    print("\nBegin autoencoder anomaly demo ")
    T.manual_seed(1)
    np.random.seed(1)

    # 2. create autoencoder net
    print("\nCreating a 65-32-8-32-65 autoencoder ")
    autoenc = Autoencoder().to(device)
    autoenc.train()   # set mode

    # 3. train autoencoder model
    bat_size = 10
    max_epochs = 20
    log_interval = 10
    lrn_rate = 0.005

    print("\nbat_size = %3d " % bat_size)
    print("max epochs = " + str(max_epochs))
    print("loss = MSELoss")
    print("optimizer = SGD")
    print("lrn_rate = %0.3f " % lrn_rate)
    train(autoenc, x_train, bat_size, max_epochs, log_interval, lrn_rate) 

    mlflow.pytorch.save_model(autoenc, args.model_folder)

    # 4. compute and store reconstruction errors
    print("\nComputing reconstruction errors ")
    autoenc.eval()  # set mode
    err_list = make_err_list(autoenc, x_train)
    err_list.sort(key=lambda x: x[1], reverse=True)  # high error to low

    # 5. show most anomalous item
    print("Largest reconstruction item / error: ")
    (idx,err) = err_list[0]
    print(" [%4d]  %0.4f" % (idx, err)) 
    #   display_digit(data_ds, idx)

    print("\nEnd autoencoder anomaly detection demo \n")


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()
    # run main function
    main(args)






































#os.makedirs(args.model_folder, exist_ok=True)


#joblib.dump(fitPipeline,args.model_folder+"/"+algorithmname+".joblib")
    
#print("Training finished!. Metrics:")
#print(f"R2_{algorithmname}", r2)
#print(f"MAPE_{algorithmname}", mape)
#print(f"RMSE_{algorithmname}", rmse)
#print("Model",args.model_folder+"/"+algorithmname+".joblib","saved!")

#if args.run_mode == 'remote':
#    mlflow.log_metric(f"R2_{algorithmname}", r2)
#    mlflow.log_metric(f"MAPE_{algorithmname}", mape)
#    mlflow.log_metric(f"RMSE_{algorithmname}", rmse)
#    mlflow.sklearn.log_model(fitPipeline,f"{algorithmname}_model")

