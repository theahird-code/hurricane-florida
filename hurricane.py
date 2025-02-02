import pandas as pd
import numpy as np
from numpy.linalg import eig
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
energy_population_file = pd.read_csv('/home/theahird/code/energy_pop.csv')
hurricane_file1 = pd.read_csv('/home/theahird/code/hurricane_irma.csv')
hurricane_file2 = pd.read_csv('/home/theahird/code/hurricane_idalia2.csv')
hurricane_file3 = pd.read_csv('/home/theahird/code/hurricane_fay.csv')
florida_risk_index_file = pd.read_csv('/home/theahird/code/florida_ri.csv')
county_coordinates_file = pd.read_csv('/home/theahird/code/county_coordinates.csv')

print("Datasets loaded successfully.")
# Define wind speed ranges for each hurricane category in knots (kt)
def saffir_simpson(wind_speed_kt):
    if wind_speed_kt >= 157:
        return 5  # Category 5
    elif wind_speed_kt >= 128:
        return 4  # Category 4
    elif wind_speed_kt >= 103:
        return 3  # Category 3
    elif wind_speed_kt >= 83:
        return 2  # Category 2
    elif wind_speed_kt >= 64:
        return 1  # Category 1
    else:
        return 0  # Tropical Storm or lower

# Haversine formula for distance calculation on the earth
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    # a mapping from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # latitude difference in radians
    dlat = lat2 - lat1
    # longitude difference in radians
    dlon = lon2 - lon1
    # arc length from latitude and longitude
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    # circumference of the earth using the haversine formula
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    # distance between the hurricane and the county
    return R * c

# Function to calculate hurricane proximity
def calculate_hurricane_proximity(data, hurricane_data):
    print("Calculating hurricane proximity...")
    # Using hurricane coordinates
    hurricane_coords = hurricane_data[['X', 'Y']].values
    # Using county coordinates
    county_coords = data[['lat', 'lon']].values
    # Create a list to store the proximity values
    proximity_list = []

    # Loop through each county and find the closest hurricane
    for lat1, lon1 in county_coords:
        # Initialize the minimum distance to infinity
        min_distance = float('inf')
        # Loop through each hurricane and calculate the distance
        for lat2, lon2 in hurricane_coords:
            # Use the Haversine formula
            distance = haversine(lat1, lon1, lat2, lon2)
            # Update the minimum distance
            if distance < min_distance:
                # Update the minimum distance
                min_distance = distance
        # Append the minimum distance to the list
        proximity_list.append(min_distance)

    # Add the proximity list to the dataframe
    data['Hurricane Proximity'] = proximity_list
    # returns the updated dataframe
    return data

# Function to calculate weighted score
def calculate_weighted_score(data, weights):
    print("Calculating weighted score...")
    score = sum(data[col] * weight for col, weight in weights.items())
    # returns the weighted score
    return score

# Function to derive weights from eigenvectors
def derive_weights_from_eigenvectors(data, columns):
    print("Deriving weights from eigenvectors...")
    # Min-Max normalize the data (scale to 0-1).
    normalized_data = (data[columns] - data[columns].min()) / (data[columns].max() - data[columns].min())
    print("Normalized data:\n", normalized_data.head())

    # Calculate covariance matrix and eigenvectors using np.
    covariance_matrix = np.cov(normalized_data.values.T)
    print("Covariance matrix:\n", covariance_matrix)

    eigenvalues, eigenvectors = eig(covariance_matrix)
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

    # Use the eigenvector corresponding to the maximum eigenvalue.
    max_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    print("Max eigenvector:\n", max_eigenvector)

    # Normalize the eigenvector to derive weights
    normalized_weights = max_eigenvector / sum(abs(max_eigenvector))
    print("Normalized weights:\n", normalized_weights)

    return {columns[i]: abs(normalized_weights[i]) for i in range(len(columns))}
# Function to rank hurricane risks
def rank_hurricane_risk_with_weights(
    energy_population_file, hurricane_file1, hurricane_file2, hurricane_file3, florida_risk_index_file, county_coordinates_file
):
    print("Ranking hurricane risks...")

    # Combine hurricane datasets
    hurricane_data = pd.concat([hurricane_file1, hurricane_file2, hurricane_file3], ignore_index=True)
    print("Hurricane data combined:\n", hurricane_data.head())

    # Merge datasets on shared location fields
    merged_data = pd.merge(energy_population_file, hurricane_data, on=['County'], how='inner')
    merged_data = pd.merge(merged_data, florida_risk_index_file, on=['County'], how='inner')
    merged_data = pd.merge(merged_data, county_coordinates_file, on=['County'], how='inner')
    merged_data = merged_data.drop_duplicates(subset=['County'])
    print("Merged data:\n", merged_data.head())
    
    # Calculate hurricane proximity
    merged_data = calculate_hurricane_proximity(merged_data, hurricane_data)
    
    # Min-Max normalize the data
    print("Normalizing data...")
    merged_data['Normalized Hurricane Risk Index'] = (merged_data['Hurricane Risk Index'] - merged_data['Hurricane Risk Index'].min()) / (merged_data['Hurricane Risk Index'].max() - merged_data['Hurricane Risk Index'].min())
    merged_data['Normalized Population'] = (merged_data['Population 2023'] - merged_data['Population 2023'].min()) / (merged_data['Population 2023'].max() - merged_data['Population 2023'].min())
    merged_data['Normalized Energy Capacity'] = (merged_data['Total Capacity (MW)'] - merged_data['Total Capacity (MW)'].min()) / (merged_data['Total Capacity (MW)'].max() - merged_data['Total Capacity (MW)'].min())
    merged_data['Normalized Risk Index'] = (merged_data['National Risk Index'] - merged_data['National Risk Index'].min()) / (merged_data['National Risk Index'].max() - merged_data['National Risk Index'].min())
    # the smaller the score, the closer the hurrican is, so we need to find the complement result of min-max normalization.
    merged_data['Normalized Hurricane Proximity'] = 1 - (merged_data['Hurricane Proximity'] - merged_data['Hurricane Proximity'].min()) / (merged_data['Hurricane Proximity'].max() - merged_data['Hurricane Proximity'].min())
    print("Normalized data columns added:\n", merged_data.head())

    # Derive weights using eigenvectors
    weights_cat = [
        'Normalized Hurricane Proximity', 
        'Normalized Population', 
        'Normalized Energy Capacity',
        'Normalized Risk Index',
        'Normalized Hurricane Risk Index'
    ]
    weights = derive_weights_from_eigenvectors(merged_data, weights_cat)
    print("Derived weights:\n", weights)

    # Calculate weighted scores
    merged_data['Risk Score'] = calculate_weighted_score(merged_data, weights)
    print("Risk scores calculated:\n", merged_data[['County', 'Risk Score']].head())

    # Rank counties by risk score
    ranked_counties = merged_data.sort_values(by='Risk Score', ascending=False)
    print("Ranked counties:\n", ranked_counties[['County', 'Risk Score']].head())

    # Save ranked results to a CSV
    output_csv_path = 'ranked_hurricane_risk.csv'
    ranked_counties.to_csv(output_csv_path, index=False)
    print(f"Ranked data has been saved to: {output_csv_path}")

    # Plot heatmap
    heatmap_data = merged_data.set_index('County')[
        ['Normalized Population', 'Normalized Energy Capacity', 'Normalized Risk Index', 'Normalized Hurricane Risk Index', 'Normalized Hurricane Proximity']
    ]
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        cmap='coolwarm',  
        cbar_kws={'label': 'Normalized Value'},
        linewidths=0.5,
        fmt=".2f"
    )
    plt.title('County Risk Factors Heatmap (Blue to Red)', fontsize=16)
    plt.xlabel('Risk Factors', fontsize=12)
    plt.ylabel('County', fontsize=12)
    plt.tight_layout()
    plt.show()

    return ranked_counties

# Call the function to test
ranked_results = rank_hurricane_risk_with_weights(
    energy_population_file, hurricane_file1, hurricane_file2, hurricane_file3, florida_risk_index_file, county_coordinates_file
)
print("Final ranked results:\n", ranked_results.head())