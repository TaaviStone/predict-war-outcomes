import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('combined_data.csv')

Train, Test = train_test_split(data, test_size=0.2, random_state=69)

X_train = Train.drop('wina', axis=1)
y_train = Train['wina']

X_test = Test.drop('wina', axis=1)
y_test = Test['wina']

logreg_model = LogisticRegression(random_state=69, max_iter=1000)
logreg_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=123)
rf_model.fit(X_train, y_train)
feature_names = X_train.columns
feature_importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
plt.yticks(rotation=45, labels=feature_names, ticks=range(0, len(feature_names)))
plt.ylabel('Feature Importance')
plt.title('Random Forest Classifier - Feature Importance')
plt.show()

