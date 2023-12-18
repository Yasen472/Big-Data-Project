import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('./car_data_2022.csv')

# Filter data for vehicles manufactured in 2023
df_2022 = df[df['car_year'] == 2022]

# Group by car make and model, and calculate the average price for each combination
average_prices_2022 = df_2022.groupby(['car_make', 'car_model'])['car_price'].mean().reset_index()

# Sort the DataFrame by 'car_make' and 'car_model' alphabetically
average_prices_2023 = average_prices_2022.sort_values(by=['car_make', 'car_model'])

# Plot the bar chart
plt.figure(figsize=(12, 6))
plt.bar(average_prices_2023.apply(lambda x: f"{x['car_make']} {x['car_model']}", axis=1), average_prices_2023['car_price'], color='blue')
plt.xlabel('Car Model')
plt.ylabel('Average Prices')
plt.title('Average Prices of Vehicles Manufactured in 2022')
plt.xticks(rotation='vertical', fontsize=6)
plt.show()
