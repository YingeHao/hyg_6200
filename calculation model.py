import pandas as pd
from sklearn.linear_model import Ridge

# Previously trained model and data (for context)
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
    'Cognitive_Load_Value': [2.66, 3.16, 3.16, 3.66, 3.33, 3, 3.83, 2.66, 3]} 
df = pd.DataFrame(data)
X = df.iloc[:, :-1]
ridge_model_train = Ridge(alpha=1.0)
ridge_model_train.fit(X, df['Cognitive_Load_Value'])

# New game's feature values
new_game_data = {
    'Response_Time': 3, 
    'Animation_Effect': 3, 
    'User_Feedback': 4,
    'Visual_Consistency': 2,  
    'Functional_Consistency': 3, 
    'Content_Consistency': 4,
    'Response_Consistency': 3, 
    'Error_Recovery': 3,
    'Accessibility': 2,  
    'Memorability': 3, 
    'New_User_Tutorial': 3, 
    'Help_and_Hints': 4, 
    'Efficiency': 4
}

# Convert the new game data to a DataFrame with feature names
new_game_df = pd.DataFrame([new_game_data])

# Predicting the cognitive load value for the new game
predicted_cognitive_load = ridge_model_train.predict(new_game_df)[0]
print(predicted_cognitive_load)
