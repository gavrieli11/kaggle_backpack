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

df, val = train_test_split(df, test_size=0.3, random_state=42)

num_vars = ['lap_comp', 'weight']
X = df[num_vars]
y = df['Price']
# lr = LinearRegression()
# lr.fit(X, y)

# pred = lr.predict(X)

# print(mean_squared_error(y, pred))

class mynn(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, 4)
        self.active_func_layer_1 = nn.ReLU()
        self.output_layer = nn.Linear(4, 1)

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


# model
my_nn = mynn().to('cpu')
opt = optim.Adam(my_nn.parameters(), lr=0.01)
output = my_nn(xt).squeeze()
loss_vals = []
epochs = 2

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

dataset = MyDataset(X.values, y.values)

dl = DataLoader(dataset, batch_size=30, shuffle=True) # How to define data loader??

for _ in range(epochs):
    for data in dl:
        x, y = data
        opt.zero_grad()
        loss_func = nn.MSELoss()
        loss_val = loss_func(my_nn(x), y)
        loss_val.backward()
        opt.step()
        loss_vals.append(loss_val.item())


test = pd.read_csv("data/test.csv")
# # print(test)
# test['Price'] = 10.0
test_ids = test['id'].values
test = test.rename(columns={'Weight Capacity (kg)': 'weight'})
test = test.rename(columns={'Laptop Compartment': 'lap_comp'})
test['lap_comp'] = df['lap_comp'].map({'Yes':1}).fillna(0)
test['weight'] = test['weight'].fillna(test['weight'].mean())
test = test[['lap_comp', 'weight']].values
test = torch.tensor(test).float()
pred = my_nn(test).detach()
pred = pred.reshape(len(pred))
test_out = pd.DataFrame({'id':test_ids ,'Price': pred})
test_out['Price'] = test_out['Price'].fillna(np.mean(test_out['Price']))
print(test_out)

# test['Price'] = my_nn(test.values)
# # print(test['Brand'].mode())
# test['Brand'] = test['Brand'].fillna('mode')
# test['Material'] = test['Material'].fillna('mode')
# test['Size'] = test['Size'].fillna('mean')
# test['Laptop Compartment'] = test['Laptop Compartment'].fillna('mode')
# test = test.fillna('mode', axis=0)
# test[['id', 'Price']].to_csv('test_to_submit.csv', index=False, encoding='UTF-8')
test_out.to_csv('test_to_submit_2.csv', index=False, encoding='UTF-8')

# print(test.isna().mean())



