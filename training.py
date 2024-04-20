from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Loading dataset
data = pd.read_csv('heart_data.csv')

# Separating features (X) and target variable (y)
X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Spliting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training LR
logistic_model = LogisticRegression(max_iter=1000) 
logistic_model.fit(X_train, y_train)

# Predictions 
y_pred_logistic = logistic_model.predict(X_test)

# Accuracy calculation
accuracy_logistic = round(accuracy_score(y_test, y_pred_logistic) * 100, 3)
print("Accuracy for Logistic Regression:", accuracy_logistic, "%")

# Training SVM
svm_model = SVC(max_iter=1000)
svm_model.fit(X_train, y_train)

# Predictions
y_pred_svm = svm_model.predict(X_test)

# Accuracy calculation
accuracy_svm = round(accuracy_score(y_test, y_pred_svm) * 100, 3)
print("Accuracy for Support Vector Machine (SVM):", accuracy_svm, "%")

# Training NN
nn_model = MLPClassifier(max_iter=1000)
nn_model.fit(X_train, y_train)

# Predictions
y_pred_nn = nn_model.predict(X_test)

# Accuracy calculation
accuracy_nn = round(accuracy_score(y_test, y_pred_nn) * 100, 3)
print("Accuracy for Neural Network:", accuracy_nn, "%")

# Training GBT
gbt_model = GradientBoostingClassifier()
gbt_model.fit(X_train, y_train)

# Predictions
y_pred_gbt = gbt_model.predict(X_test)

# Accuracy calculation
accuracy_gbt = round(accuracy_score(y_test, y_pred_gbt) * 100, 3)
print("Accuracy for Gradient Boosting Trees:", accuracy_gbt, "%")