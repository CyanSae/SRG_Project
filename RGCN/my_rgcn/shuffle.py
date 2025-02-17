import pandas as pd

df = pd.read_csv('RGCN/dataset/time/30_train_20231025.csv')

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('RGCN/dataset/time/30_train_20231025_shuffled.csv', index=False)