import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Data Preparation
train_data = pd.read_csv("task 1\\house-prices-advanced-regression-techniques\\train.csv")
test_data = pd.read_csv("task 1\\house-prices-advanced-regression-techniques\\test.csv")

# Step 2: Exploratory Data Analysis (EDA)
print(train_data.head())  # Check the first few rows of the training data
print(train_data.info())  # Check data types and missing values

# Plotting pairplot for numerical variables
import seaborn as sns
sns.pairplot(train_data[['SalePrice', 'GrLivArea', 'BedroomAbvGr', 'FullBath']])
plt.show()

# Step 3: Model Training
X_train = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y_train = train_data['SalePrice']

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_train_pred = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
print("Training MSE:", mse_train)
print("Training R-squared:", r2_train)

# Step 5: Prediction
X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y_test_pred = model.predict(X_test)

# Save predictions to a CSV file
submission_df = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': y_test_pred})
submission_df.to_csv('task 1\\house-prices-advanced-regression-techniques\\sample_submission.csv', index=False)
