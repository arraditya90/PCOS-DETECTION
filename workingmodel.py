import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the expected feature names
with open('features.txt', 'r') as f:
    expected_features = [line.strip() for line in f]


# Function to calculate BMI
def calculate_bmi(weight, height):
    height_meters = height / 100  # Convert height to meters
    return weight / (height_meters ** 2)


# Function to calculate lifestyle score
def calculate_lifestyle_score(exercise_regularly, eat_fast_food_regularly):
    return exercise_regularly - eat_fast_food_regularly


# Function to calculate period health score
def calculate_period_health_score(regular_periods, duration):
    regularity_score = 1 if regular_periods else 0
    return regularity_score * duration


# Function to get user input and make predictions
def predict():
    age = float(input("Enter age (in years): "))
    weight = float(input("Enter weight (in kg): "))
    height = float(input("Enter height (in cm): "))
    exercise_regularly = int(input("Do you exercise on a regular basis? (1 for Yes, 0 for No): "))
    eat_fast_food_regularly = int(input("Do you eat fast food regularly? (1 for Yes, 0 for No): "))
    regular_periods = bool(int(input("Are your periods regular? (1 for Yes, 0 for No): ")))
    period_duration = float(input("How long does your period last? (in days): "))

    # Calculate BMI and other scores
    bmi = calculate_bmi(weight, height)
    lifestyle_score = calculate_lifestyle_score(exercise_regularly, eat_fast_food_regularly)
    period_health_score = calculate_period_health_score(regular_periods, period_duration)

    # Prepare data for prediction
    new_data = pd.DataFrame({
        'Age (in Years)': [age],
        'Weight (in Kg)': [weight],
        'BMI': [bmi],
        'Lifestyle_Score': [lifestyle_score],
        'Period_Health_Score': [period_health_score]
    })

    # Ensure new_data has the same columns as the training data
    new_data = new_data.reindex(columns=expected_features, fill_value=0)  # Fill missing columns with 0

    # Scale the new data
    scaled_new_data = scaler.transform(new_data)

    # Make a prediction
    prediction = model.predict(scaled_new_data)

    print("Prediction:", prediction[0])


# Run the prediction function
predict()