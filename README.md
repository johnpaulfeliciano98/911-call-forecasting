# EMS Call Prediction

## Overview
This project analyzes emergency medical service (EMS) call data, extracts time-based features, and trains an XGBoost model to predict future EMS call locations and times. A heatmap is generated to visualize predictions.

## Installation
Install the required dependencies before running the notebook:
```bash
pip install pandas numpy matplotlib folium scikit-learn xgboost
```

## Steps
### 1. Install Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from IPython.display import display
```

### 2. Load the Dataset
```python
df = pd.read_csv("CLT_FY18-24_Categorized-copy.csv")
```

### 3. Filter for EMS Calls Only
```python
df = df[df['CauseCategory'] == 'EMS']
```

### 4. Extract Date & Time Features
```python
df['Dispatched'] = pd.to_datetime(df['Dispatched'], format="%m/%d/%Y %H:%M")
df['Year'] = df['Dispatched'].dt.year
df['Month'] = df['Dispatched'].dt.month
df['DayOfWeek'] = df['Dispatched'].dt.dayofweek
df['Hour'] = df['Dispatched'].dt.hour
```

### 5. Aggregate Data by Location & Time
```python
df_grouped = df.groupby(['Latitude', 'Longitude', 'Year', 'Month', 'DayOfWeek', 'Hour']).size().reset_index(name='EMS_Calls')
```

### 6. Generate Future Test Data
```python
future_time = pd.Timestamp("2025-01-06 08:00:00")
future_data = df_grouped[['Latitude', 'Longitude']].drop_duplicates().reset_index(drop=True)
future_data['Year'] = future_time.year
future_data['Month'] = future_time.month
future_data['DayOfWeek'] = future_time.dayofweek
future_data['Hour'] = future_time.hour
```

### 7. Train XGBoost Model
```python
X = df_grouped[['Latitude', 'Longitude', 'Year', 'Month', 'DayOfWeek', 'Hour']]
y = df_grouped['EMS_Calls']
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X, y)
```

### 8. Predict Future EMS Calls
```python
future_features = future_data[['Latitude', 'Longitude', 'Year', 'Month', 'DayOfWeek', 'Hour']]
future_predictions = model.predict(future_features)
future_data['Predicted_EMS_Calls'] = future_predictions
```

### 9. Evaluate Model Performance
```python
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
```

### 10. Generate Interactive Heatmap
```python
heat_data = [(row['Latitude'], row['Longitude'], row['Predicted_EMS_Calls']) for _, row in future_data.iterrows()]
m = folium.Map(location=[future_data['Latitude'].mean(), future_data['Longitude'].mean()], zoom_start=10)
HeatMap(heat_data, radius=8, blur=6, min_opacity=0.2).add_to(m)
m.save('future_ems_heatmap.html')
```

## Output
- `future_ems_heatmap.html`: Interactive map displaying predicted EMS call hotspots.
- `Predicted_EMS_Calls.csv`: Dataset containing future call predictions.

## License
This project is open-source. Feel free to use and modify as needed.

