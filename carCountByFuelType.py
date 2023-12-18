import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
# Read the CSV file
file_path = 'car_data.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)
 
# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
 
# Group by fuel type and count the occurrences
fuel_type_counts = df['fuel_type'].value_counts()
 
# Plot the counts for each fuel type
fuel_type_counts.plot(kind='bar', color=['blue', 'green', 'orange', 'red'], ax=ax)
 
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.title('Count of Vehicles by Fuel Type')
plt.xticks(rotation=0)  # Do not rotate x-axis labels
 
plt.tight_layout()
plt.show()