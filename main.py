import numpy as np
import matplotlib.pyplot as plt
from evol import Learn, Network
from sklearn.datasets import load_boston
import pandas as pd


bos = load_boston()
bos.keys()

df = pd.DataFrame(bos.data)
df.columns = bos.feature_names
df['Price'] = bos.target
df.head()

data = df[df.columns[:-1]]
data = data.apply(
    lambda x: (x - x.mean()) / x.std()
)

data['Price'] = df.Price
X = data.drop('Price', axis=1).to_numpy()
Y = data['Price'].to_numpy()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print(X_train.shape)
print(Y_train.shape)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)


m = Learn(X_train, Y_train, epochs = 1000, batch_size = 32,
            hidden = [10], count = 128,
            kill_share = 0.1, mutate_p = 0.1, verboze = 10)

error = (100 * abs((Y_test - m.forward(X_test)) / Y_test)).mean()
print("error = {:.4f}%".format(error))
m.save_weights()
