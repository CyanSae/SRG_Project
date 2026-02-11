import pandas as pd

df = pd.read_csv('RGCN/dataset/fraud/phish.csv')

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('RGCN/shuffled_dataset/shuffled_phish.csv', index=False)