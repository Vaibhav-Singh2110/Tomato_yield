import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your dataset (replace 'your_dataset.csv' with your actual file name or path)
data = pd.read_csv('USDA_yield_county.csv')

# Assuming 'Year' and 'Country' are features, and 'Yield' is the target variable
features = data[['Year', 'Country']]  # Add other relevant features
target = data['Value']

# Create a new DataFrame for storing predicted yields for each country
predicted_data = pd.DataFrame(columns=['Country', 'Predicted_Yield'])

# Create a dictionary to store historical yields for each country
historical_yields = {}

# Iterate over unique countries
for country in data['Country'].unique():
    # Filter data for the current country
    country_data = data[data['Country'] == country]

    # Check the number of data points for the current country
    if len(country_data) < 2:  # Adjust as needed based on your requirements
        print(f'Skipping {country} due to insufficient data points.')
        continue

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(country_data[['Year']],
                                                        country_data['Value'], test_size=0.2, random_state=42)

    # Create a SARIMA model
    order = (1, 1, 1)  # (p, d, q) - Tune these parameters based on your data
    seasonal_order = (1, 1, 1, 12)  # (P, D, Q, S) - Tune these parameters based on your data
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)

    # Train the model
    model_fit = model.fit(disp=False)

    # Make predictions on the test set
    predictions = model_fit.get_forecast(steps=len(X_test))

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions.predicted_mean)
    mse = mean_squared_error(y_test, predictions.predicted_mean)
    r2 = r2_score(y_test, predictions.predicted_mean)

    print(f'Mean Absolute Error for {country}: {mae}')
    print(f'Mean Squared Error for {country}: {mse}')
    print(f'R-squared for {country}: {r2}')

    # Store historical yields for the country
    historical_yields[country] = country_data['Value'].tolist()

    # Make predictions for 2024
    forecast = model_fit.get_forecast(steps=1)
    predicted_yield = forecast.predicted_mean.values[0]
    predicted_data = predicted_data.append({'Country': country, 'Predicted_Yield': predicted_yield},
                                           ignore_index=True)

# Save the predicted data to a new CSV file
predicted_data.to_csv('predicted_yields_by_country_sarima.csv', index=False)

# Plot historical and predicted yields for each country in separate bar graphs
for country, historical_yield in historical_yields.items():
    plt.figure(figsize=(8, 5))  # Adjust the figure size as needed

    # Plot historical yields
    plt.bar(data[data['Country'] == country]['Year'], historical_yield, label='Historical')

    # Plot predicted yield for 2024
    plt.bar(2024, predicted_data.loc[predicted_data['Country'] == country, 'Predicted_Yield'].values[0],
            color='red', label='Predicted', alpha=0.7)

    plt.title(f'Yields for {country} (Historical and Predicted for 2024) - SARIMA')
    plt.xlabel('Year')
    plt.ylabel('Yield')
    plt.legend()
    plt.show()
