import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load data from CSV
data = pd.read_csv('heart_data.csv')

# Separating features (X) and target variable (y)
X = data.drop(columns=['target']).values
y = data['target'].values

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Save the trained model as a pickle file
pickle.dump(logistic_model, open('model.pkl', 'wb'))

# Predictions 
# y_pred_logistic = logistic_model.predict(X_test)

# Accuracy calculation
# accuracy_logistic = round(accuracy_score(y_test, y_pred_logistic) * 100, 3)
# print("Accuracy for Logistic Regression:", accuracy_logistic, "%")

# Load the saved model
# model = pickle.load(open('model.pkl', 'rb'))

#Make predictions on new data
# new_data = [(63,1,1,145,233,1,2,150,0,2.3,3,0,6),
#            (37,1,3,130,250,0,0,187,0,3.5,3,0,3),
#            (67,1,4,160,286,0,2,108,1,1.5,2,3,3)]
# predictions = model.predict(new_data)
# print(predictions)