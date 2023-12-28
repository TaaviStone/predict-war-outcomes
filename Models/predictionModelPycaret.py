import pandas as pd
from pycaret.classification import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('combined_data.csv')

Train, Test = train_test_split(data, test_size=0.2, random_state=69)

# Setup PyCaret classification environment
clf = setup(data=Train, target='wina')

# Compare models and select the top 2 based on AUC
best_3 = compare_models(sort='AUC', n_select=3)

# Blend the top 3 models
blended = blend_models(estimator_list=best_3, fold=5, method='soft')

# Finalize the blended model
final_model = finalize_model(blended)

# Make predictions on the test set
Predictions = predict_model(final_model, data=Test)

# Display confusion matrix for the final model
plot_model(final_model, plot='confusion_matrix')

# Display ROC curve for the final model
plot_model(final_model, plot='auc')

# Print separator for better visualization
print("=======================================")

# Display the actual vs. predicted values
Result = Predictions[['wina', 'prediction_label']].astype('int')
print(Result)

# Calculate and print the model accuracy
model_score = sum(Result['wina'] == Result['prediction_label']) / len(Result)
print(f"Model Accuracy: {model_score:.4f}")

# Show the plots
plt.show()
