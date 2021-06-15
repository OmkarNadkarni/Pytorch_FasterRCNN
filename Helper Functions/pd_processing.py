import pandas as pd

csv_path = r'/home/omkarnadkarni/od_pytorch/data/train/train.csv'

df = pd.read_csv(csv_path)
x = df['class'].value_counts()
print(x)
