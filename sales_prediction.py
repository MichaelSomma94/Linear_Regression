# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 10:36:39 2022

@author: Michael
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score,confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pickle
pio.renderers.default = "browser"

data = pd.read_csv("advertising.csv")


# figure = px.scatter(data_frame = data, x="Radio",
#                     y="Sales", size="Radio", trendline="ols")
# figure.show()


# correlation = data.corr()
# print(correlation["Sales"].sort_values(ascending=False))

x = np.array(data[[
      "TV",  "Radio",  "Newspaper"]])
y = np.array(data["Sales"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                 test_size=0.2, 
                                                 random_state=42)

# model = LinearRegression()
# model.fit(xtrain, ytrain)
#saves the model with filename
filename = 'sales_prediciton.sav'
#pickle.dump(model, open(filename, 'wb'))
#loads the saved model 
loaded_model = pickle.load(open(filename, 'rb'))

print(loaded_model.score(xtest, ytest))
print(loaded_model.predict(xtest))

# cm1 = confusion_matrix(ytest, model.predict(xtest))
# #print('Confusion Matrix : \n', cm1)
