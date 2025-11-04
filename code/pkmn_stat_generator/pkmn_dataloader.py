

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv("../../code/data/cleaned_pokemon.csv")

label_encoder = LabelEncoder()
'''
type1 and type2 are categories and they overlap
and thus the encodings must line up.
'''
df['type1'] = label_encoder.fit_transform(df['type1'])
df['type2'] = label_encoder.fit_transform(df['type2'])

X = torch.tensor(df[['attack', 'defense', 'hp', 'sp_attack', 'sp_defense', 'speed', 'base_total']].values, dtype=torch.int64)
y = torch.tensor(df[['type1', 'type2']].values, dtype=torch.int64)

batch_size = 32  # Choose an appropriate batch size
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
