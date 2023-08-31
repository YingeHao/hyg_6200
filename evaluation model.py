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
df_english = pd.DataFrame(data)

# Define X and y
X_eng = df_english.iloc[:, :-1]
y_eng = df_english['Cognitive_Load_Value']

# Splitting the dataset into training and testing sets
X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(X_eng, y_eng, test_size=0.3, random_state=42)

# Training the Ridge regression model
ridge_model_train_eng = Ridge(alpha=1.0)
ridge_model_train_eng.fit(X_train_eng, y_train_eng)

# Predicting the cognitive load values on the entire dataset using the trained model
y_pred_eng = ridge_model_train_eng.predict(X_eng)

# Comparing actual and predicted cognitive load values
comparison_df = pd.DataFrame({
    'Actual_Cognitive_Load': y_eng,
    'Predicted_Cognitive_Load': y_pred_eng
})
print("Comparing Actual vs Predicted Cognitive Load Values:\n")
print(comparison_df)

# Calculating the mean squared error for the predictions on the entire dataset
mse_all_eng = mean_squared_error(y_eng, y_pred_eng)
rmse_all_eng = np.sqrt(mse_all_eng)
print(f"\nMean Squared Error (MSE) for entire dataset: {mse_all_eng:.4f}")
print(f"Root Mean Squared Error (RMSE) for entire dataset: {rmse_all_eng:.4f}\n")

# Extracting the coefficients of the Ridge model
ridge_coefficients_eng = ridge_model_train_eng.coef_
feature_importance_eng = dict(zip(X_eng.columns, ridge_coefficients_eng))

# Plotting with the coefficient values displayed on the bar chart
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
barplot = sns.barplot(x=list(feature_importance_eng.values()), y=list(feature_importance_eng.keys()))
plt.title('Feature Importance from Ridge Regression', fontsize=16)
plt.xlabel('Coefficient Value', fontsize=14)
plt.ylabel('Features', fontsize=14)

# Displaying the coefficient values on the bars
for p in barplot.patches:
    width = p.get_width()
    plt.text(width + 0.005, p.get_y() + p.get_height()/2, '{:.3f}'.format(width), va='center')

plt.show()