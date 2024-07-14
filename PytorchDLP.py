import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#device checking

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('my device', device)

#load the dataset
data = load_
X = data.data
y = data.target

print(X[:2])
print(y[:2])
print(X.shape)
print(type(X))
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
#split the dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
print(X_train.shape)
print(X_test.shape)
print(type(X_train))
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


