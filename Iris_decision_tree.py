from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


# I didn't do the downloading part again because I already have the dataset in my directory
Iris_dataset = pd.read_csv('iris.csv')

y = Iris_dataset['species']
x = Iris_dataset.drop('species', axis=1)

x_train , x_val , train_y , val_y = train_test_split(x, y, test_size=0.2, random_state=0)

model = DecisionTreeClassifier()
model.fit(x_train, train_y)

y_pred = model.predict(x_val)

error = accuracy_score(val_y, y_pred)
print(error)



