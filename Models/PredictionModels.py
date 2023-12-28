import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

data = pd.read_csv('combined_data.csv')

Train, Test = train_test_split(data, test_size=0.2, random_state=69)

X_train = Train.drop('wina', axis=1)
y_train = Train['wina']

X_test = Test.drop('wina', axis=1)
y_test = Test['wina']

logreg_model = LogisticRegression(random_state=69)
logreg_model.fit(X_train, y_train)

logreg_predictions = logreg_model.predict(X_test)

print("Logistic Regression Model:")
print("Accuracy:", accuracy_score(y_test, logreg_predictions))
print("Classification Report:\n", classification_report(y_test, logreg_predictions))

rf_model = RandomForestClassifier(random_state=123)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
print("\nRandom Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))
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

plot_confusion_matrix(y_test, logreg_predictions, 'Logistic Regression Confusion Matrix')

logreg_probs = logreg_model.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, logreg_probs, 'Logistic Regression ROC Curve')

plot_confusion_matrix(y_test, rf_predictions, 'Random Forest Confusion Matrix')
rf_probs = rf_model.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, rf_probs, 'Random Forest ROC Curve')