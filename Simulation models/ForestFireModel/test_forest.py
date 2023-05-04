# import pandas as pd
import numpy as np
# 
# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation

import os
from sklearn.tree import export_graphviz
from IPython.display import Image

#Own functions and models
import Game_of_Life
import fire
import functions_

# data = fire.array_for_ml
# number_pixels = fire.number_pixels

#this is from my last run of game of life: python Game_of_Life.py --interval 100 --savetodf --writetocsv --grid-size 50 --maxframes 10
data = np.genfromtxt('GoL_flat.csv' ,delimiter= ',')
number_pixels = 2500

print(len(data))
print(number_pixels)

df = functions_.neighbour_as_feature(data, number_pixels)

features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

print(df)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

#fitting the classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


#predicting on test set
y_pred = rf.predict(X_test)

#check accuracy of prediction
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(cm)
print("Accuracy:", accuracy)

# Extract feature names
colnames = list(df.columns.values.tolist())
feature_names = colnames[:-1]
label_name = colnames[-1]


# Extract single tree
tree = rf.estimators_[1]



#
# # Export as dot file
export_graphviz(tree, out_file='tree.dot',
                feature_names = feature_names ,
                class_names = label_name,
                rounded = True, proportion = False,
                precision = 2, filled = True)
#
# # Convert to png using system command (requires Graphviz)

os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'], shell=True, cwd=os.getcwd())
#
# # # # Display in script
from IPython.display import Image
Image(filename = 'tree.png')
