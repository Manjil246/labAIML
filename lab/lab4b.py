import numpy as np 
import pandas as pd 
PlayTennis = pd.read_csv("PlayTennis6.csv") 
print(PlayTennis)
from sklearn.preprocessing import LabelEncoder 
Le = LabelEncoder() 
PlayTennis['Outlook'] = Le.fit_transform(PlayTennis['Outlook']) 
PlayTennis['Temperature'] = Le.fit_transform(PlayTennis['Temperature']) 
PlayTennis['Humidity'] = Le.fit_transform(PlayTennis['Humidity']) 
PlayTennis['Windy'] = Le.fit_transform(PlayTennis['Windy']) 
PlayTennis['PlayTennis'] = Le.fit_transform(PlayTennis['PlayTennis']) 
print(PlayTennis)
y = PlayTennis['PlayTennis'] 
X = PlayTennis.drop(['PlayTennis'],axis=1) 
from sklearn import tree 
clf = tree.DecisionTreeClassifier(criterion = 'entropy') 
clf = clf.fit(X, y) 
tree.plot_tree(clf) 
X_pred = clf.predict(X) 
print(X_pred==y)