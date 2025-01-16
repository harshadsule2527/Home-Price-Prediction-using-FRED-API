import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap

# 1. Data Collection
def collect_data(api_key):
    fred = Fred(api_key=api_key)
    
    # Collect S&P Case-Shiller Home Price Index
    home_prices = fred.get_series('CSUSHPISA')
    
    # Collect other factors
    factors = {
        'Interest_Rate': fred.get_series('FEDFUNDS'),
        'GDP_Growth': fred.get_series('A191RL1Q225SBEA'),
        'Unemployment_Rate': fred.get_series('UNRATE'),
        'Construction_Cost': fred.get_series('WPUSI012011'),
        'Population_Growth': fred.get_series('POPTHM'),
        'Median_Household_Income': fred.get_series('MEFAINUSA646N')
    }
    
    # Combine all data into a single DataFrame
    data = pd.concat([home_prices] + list(factors.values()), axis=1)
    data.columns = ['Home_Price_Index'] + list(factors.keys())
    return data

# 2. Data Preprocessing
def preprocess_data(data):
    # Handle missing values
    data = data.fillna(method='ffill')
    
    # Ensure all data is on the same frequency (monthly)
    data = data.resample('M').last()
    
    # Create lag features
    for col in data.columns:
        if col != 'Home_Price_Index':
            data[f'{col}_lag6'] = data[col].shift(6)
            data[f'{col}_lag12'] = data[col].shift(12)
    
    # Drop rows with NaN values after creating lag features
    data = data.dropna()
    return data

# 3. Exploratory Data Analysis
def perform_eda(data):
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Features')
    plt.show()
    
    # Time series plots
    plt.figure(figsize=(15, 10))
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)
    plt.legend()
    plt.title('Time Series of All Features')
    plt.show()

# 4. Feature Selection
def select_features(X, y, k=10):
    selector = SelectKBest(f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return X_selected, selected_features

# 5. Model Building
def build_models(X_train, y_train):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return lr_model, rf_model

# 6. Model Evaluation
def evaluate_model(model, X, y, model_name):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    print(f"{model_name} - MSE: {mse:.2f}, R2: {r2:.2f}")

# 7. Interpretation and Visualization
def interpret_model(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

# Main function to run the entire process
def main():
    # Replace with your actual FRED API key
    api_key = '19f059629e826781e06720138e97c3b1'
    
    # 1. Collect Data
    data = collect_data(api_key)
    
    # 2. Preprocess Data
    processed_data = preprocess_data(data)
    
    # 3. Perform EDA
    perform_eda(processed_data)
    
    # Prepare features and target
    X = processed_data.drop('Home_Price_Index', axis=1)
    y = processed_data['Home_Price_Index']
    
    # 4. Select Features
    X_selected, selected_features = select_features(X, y)
    print("Selected features:", selected_features)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # 5. Build Models
    lr_model, rf_model = build_models(X_train, y_train)
    
    # 6. Evaluate Models
    evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # 7. Interpret Model
    interpret_model(rf_model, X_test)

if __name__ == "__main__":
    main()