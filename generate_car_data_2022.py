import pandas as pd
import random
import numpy as np

# Define car makes, models, and their corresponding types
car_makes_and_models = [

    #Audi vehicles
    ("Audi", "A1", ["hatchback", "sedan"]),
    ("Audi", "A3", ["hatchback", "sedan"]),
    ("Audi", "A4", ["sedan"]),
    ("Audi", "A5", ["sedan"]),
    ("Audi", "A6", ["sedan", "combie"]),
    ("Audi", "A7", ["sedan"]),
    ("Audi", "A8", ["sedan"]),
    ("Audi", "Q3", ["SUV"]),
    ("Audi", "Q5", ["SUV"]),
    ("Audi", "Q7", ["SUV"]),

    #BMW vehicles
    ("BMW", "Series 1", ["hatchback"]),
    ("BMW", "Series 2", ["coupe"]),
    ("BMW", "Series 3", ["sedan", "combie"]),
    ("BMW", "Series 4", ["coupe"]),
    ("BMW", "Series 5", ["sedan", "combie"]),
    ("BMW", "Series 6", ["sedan", "combie"]),
    ("BMW", "Series 7", ["sedan"]),
    ("BMW", "Series 8", ["coupe", "sedan"]),
    ("BMW", "M3", ["sedan"]),
    ("BMW", "M5", ["sedan"]),

    #Mercedes-Benz vehicles
    ("Mercedes-Benz", "A Class", ["hatchback"]),
    ("Mercedes-Benz", "B Class", ["hatchback"]),
    ("Mercedes-Benz", "C Class", ["sedan", "combie"]),
    ("Mercedes-Benz", "E Class", ["sedan", "combie"]),
    ("Mercedes-Benz", "R Class", ["minivan"]),
    ("Mercedes-Benz", "S Class", ["sedan"]),
    ("Mercedes-Benz", "Vito", ["van"]),
    ("Mercedes-Benz", "GLA", ["SUV"]),
    ("Mercedes-Benz", "GLC", ["SUV"]),
    ("Mercedes-Benz", "GLS", ["SUV"])

]

# Create an empty list to store data
data = []

# Generate 10,000 rows of data
for _ in range(10000):
    car_make, car_model, car_types = random.choice(car_makes_and_models)
    base_car_price = random.randint(10000, 45000)
    
    # Introduce variability within each car model
    model_price_variation = random.uniform(-5000, 6000)
    car_price = round(base_car_price + model_price_variation)
    car_year = 2022
    car_type = random.choice(car_types)
    car_color = random.choice(["Red", "Blue", "Black", "White", "Silver", "Gray"])
    number_of_doors = random.choice([3, 5])
    
    # Calculate random weights for fuel types
    weights = [random.uniform(0, 1) for _ in range(4)]  # Adjusted to 4 choices (Gasoline, Diesel, Electric, Hybrid)
    total_weight = sum(weights)
    
    # Scale the weights to cover the desired price range
    scaled_weights = [w / total_weight for w in weights]
    
    # Use numpy.random.choice to select fuel type with probabilities
    fuel_type = np.random.choice(["Gasoline", "Diesel", "Electric", "Hybrid"], p=[0.4, 0.2, 0.2, 0.2])
    
    transmission_type = random.choice(["Automatic", "Manual"])
    purchase_count = random.randint(1, 50)  # Adjusted range for a more varied distribution of purchase counts

    data.append([purchase_count, car_make, car_model, car_price, car_year, car_type, car_color, number_of_doors, fuel_type, transmission_type])

# Create a DataFrame
df = pd.DataFrame(data, columns=["purchase_count", "car_make", "car_model", "car_price", "car_year", "car_type", "car_color", "number_of_doors", "fuel_type", "transmission_type"])

# Save the DataFrame to a CSV file
df.to_csv("car_data_2022.csv", index=False)

print("CSV file 'car_data_2022.csv' has been generated with 10,000 rows of data.")
