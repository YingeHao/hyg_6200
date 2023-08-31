import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Data preparation
data = {
    'Response_Time': [3, 3, 3, 3, 3, 3, 3, 3, 4],
    'Animation_Effect': [3, 3, 3, 5, 5, 4, 5, 1, 2],
    'User_Feedback': [3, 4, 3, 4, 4, 3, 5, 4, 3],
    'Visual_Consistency': [2, 1, 2, 1, 3, 1, 5, 1, 1],
    'Functional_Consistency': [3, 1, 3, 2, 3, 2, 3, 1, 5],
    'Content_Consistency': [3, 3, 4, 5, 2, 2, 4, 1, 4],
    'Response_Consistency': [3, 4, 3, 3, 3, 3, 5, 1, 3],
    'Error_Recovery': [1, 4, 3, 4, 3, 4, 3, 5, 2],
    'Accessibility': [3, 3, 3, 4, 3, 3, 4, 3, 3],
    'Memorability': [3, 4, 3, 5, 3, 4, 4, 4, 5],
    'New_User_Tutorial': [3, 3, 3, 3, 4, 3, 5, 5, 4],
    'Help_and_Hints': [3, 5, 4, 4, 3, 4, 5, 4, 1],
    'Efficiency': [1, 3, 4, 4, 3, 2, 3, 1, 3],
    'Cognitive_Load_Value': [2.66, 3.16, 3.16, 3.66, 3.33, 3, 3.83, 2.66, 3]
}
df = pd.DataFrame(data)

# Define X and y
X = df.iloc[:, :-1]
y = df['Cognitive_Load_Value']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the Ridge regression model
ridge_model_train = Ridge(alpha=1.0)
ridge_model_train.fit(X_train, y_train)

# Predicting the cognitive load values on the entire dataset using the trained model
y_pred = ridge_model_train.predict(X)

# Comparing actual and predicted cognitive load values
comparison_df = pd.DataFrame({
    'Actual_Cognitive_Load': y,
    'Predicted_Cognitive_Load': y_pred
})
print("Comparing Actual vs Predicted Cognitive Load Values:\n")
print(comparison_df)

# Calculating the mean squared error for the predictions on the entire dataset
mse_all = mean_squared_error(y, y_pred)
rmse_all = np.sqrt(mse_all)
print(f"\nMean Squared Error (MSE) for entire dataset: {mse_all:.4f}")
print(f"Root Mean Squared Error (RMSE) for entire dataset: {rmse_all:.4f}\n")

# Extracting the coefficients and intercept of the Ridge model
ridge_coefficients = ridge_model_train.coef_
feature_importance = dict(zip(X.columns, ridge_coefficients))
intercept = ridge_model_train.intercept_

# Plotting with the coefficient values displayed on the bar chart
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
barplot = sns.barplot(x=list(feature_importance.values()), y=list(feature_importance.keys()))
plt.title('Feature Importance from Ridge Regression', fontsize=16)
plt.xlabel('Coefficient Value', fontsize=14)
plt.ylabel('Features', fontsize=14)

# Displaying the coefficient values on the bars
for p in barplot.patches:
    width = p.get_width()
    plt.text(width + 0.005, p.get_y() + p.get_height()/2, '{:.3f}'.format(width), va='center')

plt.show()

# Creating the formula string
formula = f"Cognitive_Load_Value = {intercept:.4f}"
for coef, feature in zip(ridge_coefficients, X.columns):
        formula += f" + ({coef:.4f} * {feature})"


# Outputting the intercept value and the complete formula
print(f"The intercept value for the Ridge Regression model is: {intercept:.4f}")
print(f"The complete formula for predicting Cognitive_Load_Value is: {formula}")
