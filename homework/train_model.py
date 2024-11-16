"""Entrene un modelo de machine learning en scikit-learn que permita 
pronosticar el precio de una casa a partir de sus propiedades. La data
para el entrenamiento del modelo se encuenctra en el archivo `house_data.csv`. 
El modelo usa las columnas 

* "bedrooms",
  
* "bathrooms",
  
* "sqft_living",

* "sqft_lot",

* "floors",

* "waterfront",

* "condition".

El archivo con el código para entrenamiento del modelo debe llamarse 
`train_model.py`. Ejemplifique el uso del modelo desde el terminal usando
curl."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("files/input/house_data.csv", sep=",", index_col=None, header=0)

X = data[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]
y = data["price"]

model = LinearRegression()
model.fit(X, y)

with open("homework/house_predictor.pkl", "wb") as file:
    pickle.dump(model, file)
