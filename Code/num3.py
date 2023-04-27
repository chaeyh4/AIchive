import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from utils import load_data, accuracy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# circle = pd.read_csv('Code/data/circle/train.csv')

DATA_NAME = 'Circle'
train_data, test_data = load_data(DATA_NAME)
train_x, train_y = train_data
test_x, test_y = test_data

train_x = pd.DataFrame(train_x, columns=['bias', 'x1', 'x2'])
train_x['x5'] = train_x['x1'].apply(lambda x1: 1 if x1 < 0 else 0)
train_x['x6'] = train_x['x1'].apply(lambda x1: 1 if abs(x1) >= 100 else 0)
train_x['x7'] = train_x['x2'].apply(lambda x2: 1 if abs(x2) >= 100 else 0)
train_x = train_x.iloc[:,[1,2,3,4]]

test_x = pd.DataFrame(test_x, columns=['bias', 'id', 'x1'])
test_x['x2'] = test_y
test_x['x5'] = test_x['x1'].apply(lambda x1: 1 if x1 < 0 else 0)
test_x['x6'] = test_x['x1'].apply(lambda x1: 1 if abs(x1) >= 100 else 0)
test_x['x7'] = test_x['x2'].apply(lambda x2: 1 if abs(x2) >= 100 else 0)

test_x = test_x.iloc[:,[1,2,3,4,5]]
test = test_x.iloc[:,1:]

X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=4)
# 스케일 조정
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)
classifier = SVC(kernel = "rbf", gamma = 1, C = 5)
classifier.fit(X_train_scaled, Y_train)
pred = classifier.predict(test_scaled)
print(accuracy_score(Y_test,pred))

# 스케일 조정
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_x)
test_scaled = scaler.transform(test)

classifier = SVC(kernel = "rbf", gamma = 1, C = 5)
classifier.fit(X_train_scaled, train_y)



pred = classifier.predict(test_scaled)
id = np.arange(1,401)

sample = pd.DataFrame({'id': id, 'target': pred})
sample.to_csv('sample.csv', index=False)

