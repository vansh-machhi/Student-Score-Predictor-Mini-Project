# Student Score Predictor Mini Project

# Step 1: Load Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Create Dataset
data = pd.DataFrame({
    'study_hours': [2, 3, 4, 5, 6, 7],
    'sleep_hours': [7, 6, 8, 5, 7, 6],
    'practice_tests': [1, 2, 2, 3, 3, 4],
    'score': [50, 55, 65, 70, 78, 85]
})

print("Dataset:")
print(data)

# Step 3: Split Features & Target
X = data[['study_hours', 'sleep_hours', 'practice_tests']]
y = data['score']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print(X_test)
# Step 5: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Step 6: Make Predictions
predictions = model.predict(X_test)
print("\nPredicted scores:", predictions)

# Step 7: Evaluate Model
r2_score = model.score(X_test, y_test)
print("\nModel Accuracy (R² score):", r2_score)

# Step 8: Custom Prediction
custom_prediction = model.predict([[6, 7, 3]])
print("\nCustom Prediction (Study=6, Sleep=7, Tests=3):", custom_prediction)

# Step 9: Visualization (Optional)
plt.scatter(data['study_hours'], data['score'])

plt.plot(
    data['study_hours'],
    model.predict(data[['study_hours', 'sleep_hours', 'practice_tests']]),
    linestyle='dashed'
)

plt.xlabel("Study Hours")
plt.ylabel("Score")   # score = a*study + b*sleep + c*tests + constant
plt.title("Study Hours vs Score Prediction")
plt.show()
