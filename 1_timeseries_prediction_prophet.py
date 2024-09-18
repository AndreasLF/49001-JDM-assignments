import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.diagnostics import cross_validation, performance_metrics


def cross_validation_results(model, initial='730 days', period='180 days', horizon='180 days'):
    cv_results = cross_validation(model, initial='1825 days', period='180 days', horizon='180 days')

    # Calculate cross-validation performance metrics (MAE, RMSE, etc.)
    cv_performance = performance_metrics(cv_results)

    # Display the performance metrics
    print(cv_performance[['horizon', 'mae', 'rmse', 'mape']])

    # Remove the time part from the 'horizon' column
    cv_performance['horizon'] = cv_performance['horizon'].dt.days.astype(str) + ' days'

    # Export to LaTeX without the index
    print(cv_performance[['horizon', 'mae', 'rmse', 'mape']].to_latex(index=False, float_format="%.4f"))

    # Plot cross-validation predictions vs actuals
    fig = plt.figure(figsize=(10,6))
    plt.plot(cv_results['ds'], cv_results['y'], label='Actual')
    plt.plot(cv_results['ds'], cv_results['yhat'], label='Predicted')
    plt.fill_between(cv_results['ds'], cv_results['yhat_lower'], cv_results['yhat_upper'], color='gray', alpha=0.3)
    plt.title("Cross-Validation Predictions vs Actuals")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load your CSV data
    df = pd.read_csv('data/yahoo_finance_data_AIR.PA.csv', parse_dates=['Date'], index_col='Date')

    # print head of data as latex table
    print(df.head().to_latex())

    # Ensure 'Volume' is processed correctly, as before
    df['Volume'] = df['Volume'].astype(str).replace('-', np.nan)
    df['Volume'] = df['Volume'].str.replace(',', '').astype(float)

    # Prepare the data for Prophet (Prophet expects columns 'ds' and 'y')
    stock_data = df[['Adj Close']].reset_index()
    stock_data.columns = ['ds', 'y']  # 'ds' for the date column and 'y' for the values (Adjusted Close Price)

    # plot full stock price data
    plt.figure(figsize=(10,6))
    plt.plot(stock_data['ds'], stock_data['y'])
    plt.title('Adjusted Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.savefig('figures/airbus_stock_price_data.pdf', format='pdf', bbox_inches='tight', dpi=300)


    # Initialize and fit the Prophet model
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True) 
    model.fit(stock_data)

    # Create a future dataframe for the next 365 days
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Plot the forecasted values
    fig1 = model.plot(forecast)
    plt.title("Stock Price Prediction with Prophet, 90 days")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.legend(['Historical Data', 'Predicted Data', 'Uncertainty Interval'])
    plt.savefig('figures/airbus_stock_price_prediction_90_days.pdf', format='pdf', bbox_inches='tight', dpi=300)


    # Plot the forecast components (trend, yearly seasonality, weekly seasonality)
    fig2 = model.plot_components(forecast)
    plt.savefig('figures/airbus_stock_price_components.pdf', format='pdf', bbox_inches='tight', dpi=300)

    
    cross_validation_results(model)