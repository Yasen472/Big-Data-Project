import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = 'car_data_2022.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Plotting
fig, ax = plt.subplots(figsize=(3, 3))  # Adjust the figsize to control the size of the entire figure

# Convert 'fuel_type' column to lowercase
df['fuel_type'] = df['fuel_type'].str.lower()

# Group by fuel type and count the occurrences
fuel_type_counts = df['fuel_type'].value_counts()

# Define colors for each fuel type
colors = {'diesel': 'darkgrey', 'electric': 'lightblue', 'hybrid': 'green', 'gasoline': 'yellow'}

# Plot the counts for each fuel type as a pie chart
fuel_type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax, colors=[colors[fuel_type] for fuel_type in fuel_type_counts.index])

plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular

# Set the title with additional space
plt.title('Percentage of Vehicles by Fuel Type Manufactured in 2022', pad=20)  # Adjust pad value to add space

# Remove y-axis label
ax.set_ylabel('')

plt.show()
