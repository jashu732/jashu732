-import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
data = pd.read_csv('crop_data.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Step 2: Preprocess the data
# Fill missing values
data.fillna(data.mean(), inplace=True)

# Assume 'Yield' is the target variable
X = data.drop('Yield', axis=1)
y = data['Yield']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 7: Make predictions on new data
new_data = pd.DataFrame({
    'Temperature': [30],  # Replace with appropriate values
    'Rainfall': [200],    # Replace with appropriate values
    'SoilQuality': [5],   # Replace with appropriate values
    # Add more features as per your dataset
})

predicted_yield = model.predict(new_data)
print(f'Predicted Yield for new data: {predicted_yield[0]}')
 👋 Hi, I’m @jashu732
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
jashu732/jashu732 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
