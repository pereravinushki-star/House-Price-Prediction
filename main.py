from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

#Download and load the dataset
housing = fetch_california_housing(as_frame=True)

#Store the data in dataframe
df = housing.frame 

#Display the first 5 rows
print("\n--- First 5 rows ---")
print(df.head()) 

# # Find the number of rows and columns
# print("\n ---Dataset dimensions---")
# print(df.shape)

# #Check column names and data types
# print("\n ---Dataset Information---")
# df.info()

# #Check summary statistics
# print("\n ---Summary Statistics---")
# print(df.describe())

# #Check for missing values
# print("\n ---Missing Values---")
# print(df.isnull().sum())

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# print("\n ---Features---")
# print(X.head())

# print("\n ---Target Variable---")
# print(y.head())

# print ("\n Feature dimensions: ", X.shape)
# print("Target dimensions: ", y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#Display the dimensions
# print("Training features:", X_train.shape)
# print("Training target:", y_train.shape)
# print("Testing features:", X_test.shape)
# print("Testing target:", y_test.shape)

#Create the linear regression model
model = LinearRegression()
#Train the model using the training data
model.fit(X_train, y_train)
print("\n Model training completed!")

#Make predictions using the test data 
y_pred = model.predict(X_test)

#Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n ---Model Evaluation---")
print("Mean Absolute Error (MAE):", round(mae, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))
print("R-squared (R2):", round(r2, 3))

#Compare actual and predicted values
comparison = pd.DataFrame({
    "Actual Value": y_test,
    "Predicted Value": y_pred
})

print("\n ---Actual vs Predicted Values---")
print(comparison.head(10).round(3))