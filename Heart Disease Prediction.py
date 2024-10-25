#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('/Users/mac/Downloads/Heart_Disease_prediction.csv')

# Define features (X) and target variable (y)
X = data.drop('Heart Disease', axis=1)  # Features
y = data['Heart Disease']  # Target variable

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Specify numeric and categorical features
numeric_features = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 'Number of vessels fluro']
categorical_features = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Thallium']

# Set up a preprocessor for the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),  # Pass numeric features unchanged
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-hot encode categorical features
    ]
)

# Create a pipeline with preprocessing and the Random Forest model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Random Forest Classifier
])

# Train the model on the training set
model.fit(X_train, y_train)

# Function to gather user input for heart disease features
def get_user_input():
    print("Please enter the details of the patient:")
    return pd.DataFrame({
        'Age': [int(input("Age: "))],
        'Sex': [input("Sex (1 for male, 0 for female): ")],
        'Chest pain type': [input("Chest pain type (0-3): ")],
        'BP': [int(input("Blood Pressure: "))],
        'Cholesterol': [int(input("Cholesterol: "))],
        'FBS over 120': [int(input("Fasting Blood Sugar > 120 (1 for yes, 0 for no): "))],
        'EKG results': [input("EKG results (0-2): ")],
        'Max HR': [int(input("Maximum Heart Rate: "))],
        'Exercise angina': [int(input("Exercise Angina (1 for yes, 0 for no): "))],
        'ST depression': [float(input("ST depression: "))],  # Added input for ST depression
        'Number of vessels fluro': [int(input("Number of vessels (0-3): "))],
        'Slope of ST': [input("Slope of ST (0-2): ")],  # Added input for Slope of ST
        'Thallium': [int(input("Thallium (0-3): "))]
    })

# Get user input and predict heart disease
new_patient = get_user_input()
predicted_disease = model.predict(new_patient)

# Display the prediction result
if predicted_disease[0] == 1:
    print("\nPredicted Result: The patient is likely to have heart disease.")
else:
    print("\nPredicted Result: The patient is unlikely to have heart disease.")


# In[ ]:





# In[ ]:





# In[ ]:




