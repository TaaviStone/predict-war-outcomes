from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('combinedData.csv')

Train, Test = train_test_split(data, test_size=0.2, random_state=69)

# Features (X) and target variable (y)
X_train = Train.drop('wina', axis=1)
y_train = Train['wina']
X_test = Test.drop('wina', axis=1)
y_test = Test['wina']

# Logistic Regression
logreg_model = LogisticRegression(random_state=69)
logreg_model.fit(X_train, y_train)

# Predictions
logreg_predictions = logreg_model.predict(X_test)

# Evaluate the model
print("Logistic Regression Model:")
print("Accuracy:", accuracy_score(y_test, logreg_predictions))
print("Classification Report:\n", classification_report(y_test, logreg_predictions))

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=123)
rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
print("\nRandom Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_true, y_prob, title):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

# Plot confusion matrix for Logistic Regression
plot_confusion_matrix(y_test, logreg_predictions, 'Logistic Regression Confusion Matrix')

# Plot ROC curve for Logistic Regression
logreg_probs = logreg_model.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, logreg_probs, 'Logistic Regression ROC Curve')

# Plot confusion matrix for Random Forest Classifier
plot_confusion_matrix(y_test, rf_predictions, 'Random Forest Confusion Matrix')

# Plot ROC curve for Random Forest Classifier
rf_probs = rf_model.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, rf_probs, 'Random Forest ROC Curve')