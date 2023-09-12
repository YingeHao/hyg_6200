import pandas as pd
from sklearn.linear_model import Ridge

# Previously trained model and data (for context)
# Data preparation

df = pd.read_csv('cognitive_data.csv')

# Define X and y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

ridge_model_train = Ridge(alpha=1.0)
ridge_model_train.fit(X, df['Cognitive_Load_Value'])

# New game's feature values
new_game_data = {
    'Response_Time': 3, 
    'Animation_Effect': 3, 
    'User_Feedback': 4,
    'Visual_Consistency': 3,  
    'Functional_Consistency': 4, 
    'Content_Consistency': 3,
    'Response_Consistency': 2, 
    'Error_Recovery': 3,
    'Accessibility': 4,  
    'Memorability': 2, 
    'New_User_Tutorial': 4, 
    'Help_and_Hints': 3, 
    'Efficiency': 3
}

# Convert the new game data to a DataFrame with feature names
new_game_df = pd.DataFrame([new_game_data])

# Predicting the cognitive load value for the new game
predicted_cognitive_load = ridge_model_train.predict(new_game_df)[0]
print(predicted_cognitive_load)
