import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

data = pd.read_csv('data/train.csv')

# metadata = {}
# metadata['info']=df.info()
# metadata['n_unique']=df.nunique()

train, val, test = np.split(data, 3)

def data_prep(df, is_training=True):
    df = df.rename(columns={'Weight Capacity (kg)': 'weight'})
    df = df.rename(columns={'Laptop Compartment': 'lap_comp'})
    df = df.rename(columns={'Waterproof': 'wp'})
    df['lap_comp'] = df['lap_comp'].map({'Yes':1}).fillna(0)
    df['weight'] = df['weight'].fillna(df['weight'].mean())
    df['wp'] = df['wp'].map({'Yes': 1}).fillna(0)

    num_vars = ['lap_comp', 'weight', 'wp']
    x = df[num_vars]
    xt = torch.tensor(x.values, dtype=torch.float).to('cpu')
    xt.requires_grad = True
    if is_training:
        y = df['Price']
        yt = torch.tensor(y.values, dtype=torch.float).to('cpu')
        yt.requires_grad = True
        return xt, yt
    return xt

class mynn(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(3, 8)
        self.active_func_layer_1 = nn.ReLU()
        self.h_layer = nn.Linear(8, 4)
        self.active_func_h_layer = nn.ReLU()
        self.output_layer = nn.Linear(4, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.active_func_layer_1(x)
        x = self.h_layer(x)
        x = self.active_func_h_layer(x)
        x = self.output_layer(x)
        return x

x_train, y_train = data_prep(train)
x_val, y_val = data_prep(val)

# model
my_nn = mynn().to('cpu')
opt = optim.Adam(my_nn.parameters(), lr=0.01)
output = my_nn(x_train).squeeze()
loss_save = []
running_loss = 0.0
epochs = 1

# Data loader
class MyDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = MyDataset(x_train, y_train)
dl = DataLoader(dataset, batch_size=150, shuffle=True) # How to define data loader??

for _ in range(epochs):
    for data in dl:
        x, y = data
        opt.zero_grad()
        loss_func = nn.MSELoss()
        loss_val = loss_func(my_nn(x).squeeze(1), y)
        loss_val.backward()
        opt.step()
        loss_save.append(loss_val.item())

# plt.plot(loss_save)
# plt.show()

test = pd.read_csv("data/test.csv")

x_test = data_prep(test, False)
test_ids = test['id']
pred = my_nn(x_test).detach().squeeze(1)
print(pred.shape)

test_out = pd.DataFrame({'id':test_ids ,'Price': pred})
test_out['Price'] = test_out['Price'].fillna(np.mean(test_out['Price']))

test_out.to_csv('test_to_submit_4.csv', index=False, encoding='UTF-8')

