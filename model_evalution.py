from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('heart_data.csv')

# Separate features (X) and target variable (y)
X = data.iloc[:, :-1]  # Features are all columns except the last one
y = data.iloc[:, -1]   # Target variable is the last column

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# TRAINING MODELS AND OBTAINING ACCURACY, PRECISION, RECALL
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Initialize and train the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)  # Increase max_iter here if needed
logistic_model.fit(X_train, y_train)

# Make predictions on the testing set for Logistic Regression
y_pred_logistic = logistic_model.predict(X_test)

# Calculate accuracy, precision, and recall for Logistic Regression
accuracy_logistic = round(accuracy_score(y_test, y_pred_logistic) * 100, 3)
precision_logistic = round(precision_score(y_test, y_pred_logistic) * 100, 3)
recall_logistic = round(recall_score(y_test, y_pred_logistic) * 100, 3)

print("Accuracy for Logistic Regression:", accuracy_logistic, "%")
print("Precision for Logistic Regression:", precision_logistic, "%")
print("Recall for Logistic Regression:", recall_logistic, "%")

# Initialize and train the Support Vector Machine (SVM) classifier
svm_model = SVC(max_iter=1000)
svm_model.fit(X_train, y_train)

# Make predictions on the testing set for SVM
y_pred_svm = svm_model.predict(X_test)

# Calculate accuracy, precision, and recall for SVM
accuracy_svm = round(accuracy_score(y_test, y_pred_svm) * 100, 3)
precision_svm = round(precision_score(y_test, y_pred_svm) * 100, 3)
recall_svm = round(recall_score(y_test, y_pred_svm) * 100, 3)

print("Accuracy for Support Vector Machine (SVM):", accuracy_svm, "%")
print("Precision for Support Vector Machine (SVM):", precision_svm, "%")
print("Recall for Support Vector Machine (SVM):", recall_svm, "%")

# Initialize and train the Neural Network classifier
nn_model = MLPClassifier(max_iter=1000)
nn_model.fit(X_train, y_train)

# Make predictions on the testing set for Neural Network
y_pred_nn = nn_model.predict(X_test)

# Calculate accuracy, precision, and recall for Neural Network
accuracy_nn = round(accuracy_score(y_test, y_pred_nn) * 100, 3)
precision_nn = round(precision_score(y_test, y_pred_nn) * 100, 3)
recall_nn = round(recall_score(y_test, y_pred_nn) * 100, 3)

print("Accuracy for Neural Network:", accuracy_nn, "%")
print("Precision for Neural Network:", precision_nn, "%")
print("Recall for Neural Network:", recall_nn, "%")

# Initialize and train the Gradient Boosting Trees classifier
gbt_model = GradientBoostingClassifier()
gbt_model.fit(X_train, y_train)

# Make predictions on the testing set for Gradient Boosting Trees
y_pred_gbt = gbt_model.predict(X_test)

# Calculate accuracy, precision, and recall for Gradient Boosting Trees
accuracy_gbt = round(accuracy_score(y_test, y_pred_gbt) * 100, 3)
precision_gbt = round(precision_score(y_test, y_pred_gbt) * 100, 3)
recall_gbt = round(recall_score(y_test, y_pred_gbt) * 100, 3)

print("Accuracy for Gradient Boosting Trees:", accuracy_gbt, "%")
print("Precision for Gradient Boosting Trees:", precision_gbt, "%")
print("Recall for Gradient Boosting Trees:", recall_gbt, "%")

# ROC CURVE
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Initialize the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine (SVM)": SVC(max_iter=1000, probability=True),
    "Neural Network": MLPClassifier(max_iter=1000),
    "Gradient Boosting Trees": GradientBoostingClassifier()
}

# Plot ROC curves for each classifier
plt.figure(figsize=(10, 8))
for name, classifier in classifiers.items():
    # Train the classifier
    classifier.fit(X_train, y_train)
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot random guessing line
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guessing')

# Set plot labels and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# Show plot
plt.show()

# CONFUSION MATRIX
import seaborn as sns

# Calculate and visualize confusion matrix for each classifier
for name, classifier in classifiers.items():
    # Train the classifier
    classifier.fit(X_train, y_train)
    
    # Make predictions on the testing set
    y_pred = classifier.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# CROSS-VALIDATION SCORES
import numpy as np

# Initialize lists to store mean and standard deviation of cross-validation scores
mean_scores = []
std_scores = []

# Perform cross-validation for each classifier
for name, classifier in classifiers.items():
    # Perform 10-fold cross-validation
    scores = cross_val_score(classifier, X_scaled, y, cv=10)
    
    # Calculate mean and standard deviation of cross-validation scores
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Append mean and standard deviation to lists
    mean_scores.append(mean_score)
    std_scores.append(std_score)
    
    # Print mean and standard deviation of cross-validation scores
    print(f"{name}: Mean Cross-Validation Score = {mean_score:.3f}, Standard Deviation = {std_score:.3f}")

# Plot the cross-validation score graph as a bar graph
plt.figure(figsize=(10, 6))
plt.bar(list(classifiers.keys()), mean_scores, yerr=std_scores, capsize=5)
plt.title('Cross-Validation Scores of Classifiers')
plt.xlabel('Classifier')
plt.ylabel('Mean Cross-Validation Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ACCURACY, PRECISION AND RECALL BAR GRAPH
# Define the classifiers and their names
classifiers = [logistic_model, svm_model, nn_model, gbt_model]
classifier_names = ["Logistic Regression", "Support Vector Machine (SVM)", "Neural Network", "Gradient Boosting Trees"]

# Initialize lists to store metric values
accuracy_scores = []
precision_scores = []
recall_scores = []

# Calculate metrics for each classifier
for classifier in classifiers:
    # Predictions on the test set
    y_pred = classifier.predict(X_test)
    # Accuracy
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    # Precision
    precision_scores.append(precision_score(y_test, y_pred))
    # Recall
    recall_scores.append(recall_score(y_test, y_pred))

# Set the width of the bars
bar_width = 0.3

# Set the positions for the bars
r1 = np.arange(len(classifier_names))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Define mild colors
accuracy_color = '#87CEEB'  # Light sky blue
precision_color = '#FFA07A'  # Light salmon
recall_color = '#98FB98'  # Pale green

# Plot the bar graphs for accuracy, precision, and recall
plt.figure(figsize=(10, 6))

# Accuracy
plt.bar(r1, accuracy_scores, color=accuracy_color, width=bar_width, edgecolor='grey', label='Accuracy')

# Precision
plt.bar(r2, precision_scores, color=precision_color, width=bar_width, edgecolor='grey', label='Precision')

# Recall
plt.bar(r3, recall_scores, color=recall_color, width=bar_width, edgecolor='grey', label='Recall')

# Add xticks on the middle of the group bars
plt.xlabel('Classifier', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(classifier_names))], classifier_names, rotation=45)

# Create legend & Show graphic
plt.legend()
plt.title('Accuracy, Precision, and Recall for Classifiers')
plt.tight_layout()
plt.show()

# EXECUTION TIME 
import time

# Define the classifiers and their names
classifiers = [logistic_model, svm_model, nn_model, gbt_model]
classifier_names = ["Logistic Regression", "Support Vector Machine (SVM)", "Neural Network", "Gradient Boosting Trees"]

# Initialize a list to store execution times
execution_times = []

# Measure execution time for each classifier
for classifier in classifiers:
    start_time = time.time()
    classifier.fit(X_train, y_train)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)

# Plot the execution times
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, execution_times, color='skyblue')
plt.xlabel('Classifier')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time for Different Classifiers')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()
