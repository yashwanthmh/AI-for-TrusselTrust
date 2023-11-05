import pandas as pd
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('prophet_data_local_authorities.csv')
data['ds'] = pd.to_datetime(data['ds'])
data.sort_values('ds', inplace=True)
data['year'] = data['ds'].dt.year
data['month'] = data['ds'].dt.month
data['day'] = data['ds'].dt.day
data['dayofweek'] = data['ds'].dt.dayofweek
X = data[['year', 'month', 'day', 'dayofweek']]
y = data['y']
test_size = 24  
train_size = len(X) - test_size
X_train, X_test, y_train, y_test = X.iloc[:train_size], X.iloc[-test_size:], y.iloc[:train_size], y.iloc[-test_size:]
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
predictions_df = pd.DataFrame({
    'ds': data['ds'][-test_size:],
    'y': y_test,
    'yhat': predictions
})
predictions_df.to_csv('rf_forecast_predictions.csv', index=False)
