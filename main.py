import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch
import torch.optim as optim
df = pd.read_csv('data/train.csv')

metadata = {}
metadata['info']=df.info()
metadata['n_unique']=df.nunique()

# print(pd.DataFrame(metadata))

# print(df['Weight Capacity (kg)'])

df = df.rename(columns={'Weight Capacity (kg)': 'weight'})
df = df.rename(columns={'Laptop Compartment': 'lap_comp'})
df['lap_comp'] = df['lap_comp'].map({'Yes':1}).fillna(0)
df['weight'] = df['weight'].fillna(df['weight'].mean())

num_vars = ['lap_comp', 'weight']
X = df[num_vars]
y = df['Price']
lr = LinearRegression()
lr.fit(X, y)

pred = lr.predict(X)

print(mean_squared_error(y, pred))

class mynn(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, 2)
        self.active_func_layer_1 = nn.ReLU()
        self.output_layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.active_func_layer_1(x)
        x = self.output_layer(x)
        return x
    
# tensors
xt = torch.tensor(X.values, dtype=torch.float).to('cpu')
yt = torch.tensor(y.values, dtype=torch.float).to('cpu')
xt.requires_grad = True
yt.requires_grad = True


# yt = yt.view(-1, 1)
# model
my_nn = mynn()
optimizer = optim.Adam(my_nn.parameters(), lr=0.01)
output = my_nn(xt).squeeze()

# forward pass
loss_fn = nn.MSELoss()
loss = loss_fn(output, yt)
# backward pass
loss.backward()
print(loss)
optimizer.step()

# forward pass
output = my_nn(xt).squeeze()
loss = loss_fn(output, yt)

loss.backward()
print(loss)




