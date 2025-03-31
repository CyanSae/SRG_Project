import pandas as pd

df = pd.read_csv('RGCN/dataset/ponzi/1312.csv')

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('RGCN/shuffled_dataset/1312_shuffled.csv', index=False)