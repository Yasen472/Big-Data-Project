<<<<<<< HEAD
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./car_data.csv')  # Read the CSV file

# Remove leading and trailing whitespaces from 'car_make' column
df['car_make'] = df['car_make'].str.strip()

# Separate features and target variable
X = df.drop('car_price', axis=1)
y = df['car_price']

# Convert categorical variables into dummy/indicator variables
categorical_columns = ['car_make', 'car_model', 'car_type', 'car_color', 'fuel_type', 'transmission_type']

X = pd.get_dummies(X, columns=categorical_columns)  # Use drop_first=True to avoid multicollinearity

    # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

    # Predict prices for the testing set (2024 vehicles)
predicted_prices = model.predict(X_test)


# Predict prices for the testing set (2024 vehicles)
predicted_prices = model.predict(X_test)

# Calculate average predicted prices for each unique combination of car make and model
unique_combinations = df[['car_make', 'car_model']].drop_duplicates()
# Sort the unique combinations alphabetically
unique_combinations = unique_combinations.sort_values(by=['car_make', 'car_model'])
# print(unique_combinations)
average_predicted_prices = []

for _, row in unique_combinations.iterrows():
    make = row['car_make']
    model = row['car_model']

    # Check if the dummy variable columns exist in the test set
    make_col = f'car_make_{make}'
    model_col = f'car_model_{model}'

    if make_col in X_test.columns and model_col in X_test.columns:
        filter_condition = (X_test[make_col] == 1) & (X_test[model_col] == 1)
        
        # Use the boolean array to filter predicted prices
        predicted_prices_subset = predicted_prices[filter_condition.values]
        
        average_predicted_prices.append(np.mean(predicted_prices_subset))
    else:
        # Handle the case where the columns are not found in the test set
        print(f"{model_col} is not found") 
        average_predicted_prices.append(np.nan)

# Plot the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.8
index = np.arange(len(unique_combinations))

plt.bar(index, average_predicted_prices, bar_width, label='Average Predicted Prices', alpha=0.7)

plt.xlabel('Car Make and Model')
plt.ylabel('Average Predicted Prices')
plt.title('Average Predicted Prices for Each Car Make and Model in 2024')
plt.xticks(index, unique_combinations.apply(lambda x: f"{x['car_make']} {x['car_model']}", axis=1), rotation='vertical')
plt.legend()

plt.show()
=======
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('D:/Python_experiment/car_data.csv')  # Read the CSV file

# Remove leading and trailing whitespaces from 'car_make' column
df['car_make'] = df['car_make'].str.strip()

# Separate features and target variable
X = df.drop('car_price', axis=1)
y = df['car_price']

# Convert categorical variables into dummy/indicator variables
categorical_columns = ['car_make', 'car_model', 'car_type', 'car_color', 'fuel_type', 'transmission_type']

X = pd.get_dummies(X, columns=categorical_columns)  # Use drop_first=True to avoid multicollinearity

    # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

    # Predict prices for the testing set (2024 vehicles)
predicted_prices = model.predict(X_test)


# Predict prices for the testing set (2024 vehicles)
predicted_prices = model.predict(X_test)

# Calculate average predicted prices for each unique combination of car make and model
unique_combinations = df[['car_make', 'car_model']].drop_duplicates()
# Sort the unique combinations alphabetically
unique_combinations = unique_combinations.sort_values(by=['car_make', 'car_model'])
# print(unique_combinations)
average_predicted_prices = []

for _, row in unique_combinations.iterrows():
    make = row['car_make']
    model = row['car_model']

    # Check if the dummy variable columns exist in the test set
    make_col = f'car_make_{make}'
    model_col = f'car_model_{model}'

    if make_col in X_test.columns and model_col in X_test.columns:
        filter_condition = (X_test[make_col] == 1) & (X_test[model_col] == 1)
        
        # Use the boolean array to filter predicted prices
        predicted_prices_subset = predicted_prices[filter_condition.values]
        
        average_predicted_prices.append(np.mean(predicted_prices_subset))
    else:
        # Handle the case where the columns are not found in the test set
        print(f"{model_col} is not found") 
        average_predicted_prices.append(np.nan)

# Plot the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.8
index = np.arange(len(unique_combinations))

plt.bar(index, average_predicted_prices, bar_width, label='Average Predicted Prices', alpha=0.7)

plt.xlabel('Car Make and Model')
plt.ylabel('Average Predicted Prices')
plt.title('Average Predicted Prices for Each Car Make and Model (2024) - Random Forest Regressor')
plt.xticks(index, unique_combinations.apply(lambda x: f"{x['car_make']} {x['car_model']}", axis=1), rotation='vertical', fontsize=6)
plt.legend()

plt.show()
>>>>>>> 8c28d8af1778262cc9c3fddedf3c13c60d488013
