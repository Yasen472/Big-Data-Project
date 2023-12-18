<<<<<<< HEAD
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
file_path = 'car_data.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Get unique combinations of car make and model
unique_combinations = df[['car_make', 'car_model']].drop_duplicates()

# Sort unique makes and models alphabetically
unique_makes = sorted(unique_combinations['car_make'].unique())
unique_models = sorted(unique_combinations['car_model'].unique())

# Create a dictionary to map each unique model to a unique color
model_colors = {model: color for model, color in zip(unique_models, sns.color_palette('husl', n_colors=len(unique_models)))}

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

for make in unique_makes:
    make_data = unique_combinations[unique_combinations['car_make'] == make]
    
    for model in unique_models:
        if model in make_data['car_model'].values:
            make_model_data = df[(df['car_make'] == make) & (df['car_model'] == model)]
            
            count = make_model_data.shape[0]
            
            # Assign a unique color to each bar
            color = model_colors[model]
            
            # Plot each bar with a label
            bar_label = f"{make} {model}"
            plt.bar(bar_label, count, color=color)

# Create a legend outside the plot area
handles = [plt.Rectangle((0,0),1,1, color=model_colors[model], ec="k") for model in unique_models]
plt.legend(handles, unique_models, title='Car Models', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel('Car Make and Model')
plt.ylabel('Count')
plt.title('Count of Vehicles by Make and Model Manufactured in 2023')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()
plt.show()
=======
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
file_path = 'car_data.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Get unique combinations of car make and model
unique_combinations = df[['car_make', 'car_model']].drop_duplicates()

# Sort unique makes and models alphabetically
unique_makes = sorted(unique_combinations['car_make'].unique())
unique_models = sorted(unique_combinations['car_model'].unique())

# Create a dictionary to map each unique model to a unique color
model_colors = {model: color for model, color in zip(unique_models, sns.color_palette('husl', n_colors=len(unique_models)))}

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

for make in unique_makes:
    make_data = unique_combinations[unique_combinations['car_make'] == make]
    
    for model in unique_models:
        if model in make_data['car_model'].values:
            make_model_data = df[(df['car_make'] == make) & (df['car_model'] == model)]
            
            count = make_model_data.shape[0]
            
            # Assign a unique color to each bar
            color = model_colors[model]
            
            # Plot each bar with a label
            bar_label = f"{make} {model}"
            plt.bar(bar_label, count, color=color)

# Create a legend outside the plot area
handles = [plt.Rectangle((0,0),1,1, color=model_colors[model], ec="k") for model in unique_models]
plt.legend(handles, unique_models, title='Car Models', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel('Car Make and Model')
plt.ylabel('Count')
plt.title('Count of Vehicles by Make and Model Manufactured in 2023')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()
plt.show()
>>>>>>> 8c28d8af1778262cc9c3fddedf3c13c60d488013
