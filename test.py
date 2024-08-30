import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load the diabetes dataset
df = datasets.load_diabetes()

# Prepare the data
X = pd.DataFrame(df.data, columns=df.feature_names)  # Features
y = pd.Series(df.target)  # Target

# Split the data into training and testing sets
trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(trainx, trainy)

# Make predictions on the test set
predictions = model.predict(testx)

# Evaluate the model
mse = mean_squared_error(testy, predictions)
print(f"Mean Squared Error: {mse:.4f}")
