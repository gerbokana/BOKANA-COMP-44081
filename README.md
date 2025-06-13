# BOKANA-COMP-44081
Technical Test for Applied Scientist (Data Scientist) - Opportunity No. 44081

The Data Science Team has built a simple forecasting model that uses data related to ferry tickets for the Toronto Island Park service from the City of Toronto's Open Data Portal (https://open.toronto.ca/dataset/toronto-island-ferry-ticket-counts/) to predict the number of redemptions (i.e., people getting on the ferry) at any given time.

However, the model does not perform well, and it does not have any way to account for uncertainty in the forecast.

This submission followed the instructions at the link that follows to complete the technical test: 
https://github.com/govmb-ds/COMP-44081 

Addressing the business need for accurate daily forecasting of ticket redemptions and sales, we developed two modular forecasting systems: one for redemptions and one for sales. Both models were implemented in Python and evaluated using time series cross-validation.

Redemption Model (Improved)

We began with the original base model, which decomposes the time series using seasonal patterns based on the day of the year. While simple and intuitive, this model does not adapt well to changing trends or anomalies.

To improve accuracy, we added:

•	ARIMA (AutoRegressive Integrated Moving Average): A statistical model that captures trends and temporal dependencies.

•	SARIMA (Seasonal ARIMA): An enhanced version of ARIMA that incorporates weekly seasonality (e.g., more redemptions on weekends), using pmdarima.auto_arima to automatically find optimal model parameters.

Each model was evaluated using Mean Absolute Percentage Error (MAPE) across multiple rolling time splits. The SARIMA model consistently provided the most accurate and stable predictions, making it suitable for production use.

Sales Forecast Model (New)

We developed a parallel model to forecast daily ticket sales using the same methodology:

•	a baseline weekly seasonal model

•	ARIMA(1,1,1)

•	Auto-tuned SARIMA with 7-day seasonality

The model assumes a DatetimeIndex and a numeric target column ('sales'). It also uses time series cross-validation and produces visual forecasts to support interpretability.

Why It’s Better:

•	model accuracy: SARIMA captures both long-term trends and weekly seasonality, outperforming the base model.

•	validation approach: time series cross-validation mimics real-world forecasting, avoiding data leakage.

•	modular design: both models are implemented in reusable classes, easily integrated into existing analytics workflows.

•	business relevance: By forecasting both redemptions and sales, stakeholders can better align operations with customer behavior - improving staffing, supply planning, and promotional timing related to the Toronto Island ferry.

These improvements provide a solid foundation for accurate, data-driven forecasting of customer actions related to the Toronto Island ferry.

