import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv')

metadata = {}
metadata['info']=df.info()
metadata['n_unique']=df.nunique()

print(pd.DataFrame(metadata))