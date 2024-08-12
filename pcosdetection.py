import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your dataset
file_path = 'CLEAN- PCOS SURVEY SPREADSHEET.csv'
data = pd.read_csv(file_path)

# Calculate BMI (Body Mass Index)
data['Height (in M)'] = data['Height (in Cm / Feet)'] / 100
data['BMI'] = data['Weight (in Kg)'] / (data['Height (in M)'] ** 2)

# Create a lifestyle score: Higher score indicates a healthier lifestyle
data['Lifestyle_Score'] = data['Do you exercise on a regular basis ?'] - data['Do you eat fast food regularly ?']

# Combine period regularity and duration into a single feature
data['Period_Health_Score'] = data['Are your periods regular ?'] * data['How long does your period last ? (in Days)\nexample- 1,2,3,4.....']

# Drop the redundant columns used in creating these features
data = data.drop(columns=['Height (in Cm / Feet)', 'Height (in M)', 'Do you exercise on a regular basis ?', 'Do you eat fast food regularly ?', 'Are your periods regular ?', 'How long does your period last ? (in Days)\nexample- 1,2,3,4.....'])

# Select features for scaling
features_to_scale = ['Age (in Years)', 'Weight (in Kg)', 'BMI', 'Lifestyle_Score', 'Period_Health_Score']

# Initialize the scaler and scale the features
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Save the processed dataset to a new CSV file
processed_file_path = 'Processed_PCOS_Dataset.csv'
data.to_csv(processed_file_path, index=False)

print(f"Processed data saved to {processed_file_path}")
#IMPORTING OF THE CSV FILE
#
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Select features for scaling
features_to_scale = ['Age (in Years)', 'Weight (in Kg)', 'BMI', 'Lifestyle_Score', 'Period_Health_Score']

# Initialize the scaler
scaler = StandardScaler()

# Scale the features
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Separate features and target variable
X = data.drop(columns=['Have you been diagnosed with PCOS/PCOD?'])
y = data['Have you been diagnosed with PCOS/PCOD?']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the first few rows of the training set
print(X_train.head())
print(y_train.head())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the Random Forest model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

from sklearn.model_selection import GridSearchCV

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5-fold cross-validation
                           n_jobs=-1,
                           verbose=2)

# Fit the GridSearchCV object
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
from sklearn.model_selection import cross_val_score

# Initialize the RandomForestClassifier with the best parameters found from GridSearchCV
best_model = grid_search.best_estimator_

# Perform cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5)  # 5-fold cross-validation

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())
importances = best_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for better visualization
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importances)

import pandas as pd

# Define feature names and importances based on your previous output
feature_names = [
    'Period_Health_Score', 'After how many months do you get your periods?', 'BMI',
    'Weight (in Kg)', 'Age (in Years)', 'Do you have excessive body/facial hair growth?',
    'Are you noticing skin darkening recently?', 'Can you tell us your blood group?',
    'Do have hair loss/hair thinning/baldness?', 'Lifestyle_Score', 'Have you gained weight recently?',
    'Do you have pimples/acne on your face/jawline?', 'Do you experience mood swings?'
]

importances = [
    0.323713, 0.139094, 0.118982, 0.091259, 0.088948, 0.079715,
    0.043713, 0.027590, 0.022984, 0.018661, 0.018273, 0.016226, 0.010842
]

# Create a DataFrame
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Save the table to a CSV file
feature_importances.to_csv('Feature_Importances.csv', index=False)

# Display the table
print(feature_importances)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# Create DataFrame for confusion matrix
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Negative', 'Positive'], columns=['Predicted Negative', 'Predicted Positive'])

# Save the confusion matrix and classification report to CSV files
conf_matrix_df.to_csv('Confusion_Matrix.csv')
pd.DataFrame(class_report).transpose().to_csv('Classification_Report.csv')

# Display the metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix_df)
print("Classification Report:")
print(pd.DataFrame(class_report).transpose())
import matplotlib.pyplot as plt
import seaborn as sns

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('Feature_Importances.png')  # Save the figure
plt.show()  # Display the figure
import seaborn as sns

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('Confusion_Matrix.png')  # Save the figure
plt.show()  # Display the figure

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='--')
plt.title('Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores) + 1))
plt.tight_layout()
plt.savefig('Cross_Validation_Scores.png')  # Save the figure
plt.show()  # Display the figure

import joblib

# Save the trained model to a file
joblib.dump(model, 'random_forest_model.pkl')

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your dataset
file_path = 'CLEAN- PCOS SURVEY SPREADSHEET.csv'
data = pd.read_csv(file_path)

# Calculate BMI and create additional features
data['Height (in M)'] = data['Height (in Cm / Feet)'] / 100
data['BMI'] = data['Weight (in Kg)'] / (data['Height (in M)'] ** 2)
data['Lifestyle_Score'] = data['Do you exercise on a regular basis ?'] - data['Do you eat fast food regularly ?']
data['Period_Health_Score'] = data['Are your periods regular ?'] * data['How long does your period last ? (in Days)\nexample- 1,2,3,4.....']
data = data.drop(columns=['Height (in Cm / Feet)', 'Height (in M)', 'Do you exercise on a regular basis ?', 'Do you eat fast food regularly ?', 'Are your periods regular ?', 'How long does your period last ? (in Days)\nexample- 1,2,3,4.....'])

# Select features for scaling
features_to_scale = ['Age (in Years)', 'Weight (in Kg)', 'BMI', 'Lifestyle_Score', 'Period_Health_Score']

# Initialize the scaler, fit it, and scale the features
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Separate features and target variable
X = data.drop(columns=['Have you been diagnosed with PCOS/PCOD?'])
y = data['Have you been diagnosed with PCOS/PCOD?']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'random_forest_model.pkl')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your dataset
file_path = 'CLEAN- PCOS SURVEY SPREADSHEET.csv'
data = pd.read_csv(file_path)

# Calculate BMI and create additional features
data['Height (in M)'] = data['Height (in Cm / Feet)'] / 100
data['BMI'] = data['Weight (in Kg)'] / (data['Height (in M)'] ** 2)
data['Lifestyle_Score'] = data['Do you exercise on a regular basis ?'] - data['Do you eat fast food regularly ?']
data['Period_Health_Score'] = data['Are your periods regular ?'] * data['How long does your period last ? (in Days)\nexample- 1,2,3,4.....']
data = data.drop(columns=['Height (in Cm / Feet)', 'Height (in M)', 'Do you exercise on a regular basis ?', 'Do you eat fast food regularly ?', 'Are your periods regular ?', 'How long does your period last ? (in Days)\nexample- 1,2,3,4.....'])

# Select features for scaling
features_to_scale = ['Age (in Years)', 'Weight (in Kg)', 'BMI', 'Lifestyle_Score', 'Period_Health_Score']

# Initialize the scaler, fit it, and scale the features
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Save the scaler and the feature names
joblib.dump(scaler, 'scaler.pkl')

# Save the feature names
with open('features.txt', 'w') as f:
    for feature in features_to_scale:
        f.write(f"{feature}\n")

# Separate features and target variable
X = data.drop(columns=['Have you been diagnosed with PCOS/PCOD?'])
y = data['Have you been diagnosed with PCOS/PCOD?']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'random_forest_model.pkl')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)