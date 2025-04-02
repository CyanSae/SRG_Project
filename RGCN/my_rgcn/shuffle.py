import pandas as pd

df = pd.read_csv('RGCN/dataset/ponzi/multi.csv')

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('RGCN/shuffled_dataset/multi_shuffled.csv', index=False)