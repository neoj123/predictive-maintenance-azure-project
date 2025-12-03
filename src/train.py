import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# NASA CMAPSS Data
def load_data():
    url = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/train_FD001.txt"
    col_names = ['unit', 'time', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(url, sep=r'\s+', header=None, names=col_names)
    return df

def process_data(df):
    # Calculate Remaining Useful Life
    # RUL = The max time cycle this engine reached - current time cycle
    max_cycles = df.groupby('unit')['time'].max().reset_index()
    max_cycles.columns = ['unit', 'max_time']
    df = df.merge(max_cycles, on='unit', how='left')
    df['RUL'] = df['max_time'] - df['time']
    features = [f's{i}' for i in range(1, 22)]
    X = df[features]
    y = df['RUL']
    return X, y

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    X, y = process_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Model...")
    model = RandomForestRegressor(n_estimators=50, max_depth=10) # Smaller trees for speed
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Trained. MSE: {mse}")
    
    # Save locally
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/rul_model.pkl')
    print("Model saved to model/rul_model.pkl")