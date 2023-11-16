from objects.region import Region
import pandas as pd

import xgboost as xgb
xgb.set_config(verbosity=0)

import shap

import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.model_selection import GridSearchCV

def preprocessDataframeForRegion(df, region_name, avodaco_type = 'conventional'):
    '''TODO'''
    regionValueCounts=df['region'].value_counts()
    TotalUS_region = Region(region_name, regionValueCounts[region_name], df)

    if avodaco_type == 'conventional':
        df = TotalUS_region.getConventionalDataframe()[['Date', 'AveragePrice']]
    else:
        df = TotalUS_region.getOrganicDataframe()[['Date', 'AveragePrice']]

    ## Preprocess the data
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    df.columns = ['ds', 'y']
    # df['ds'] = pd.to_datetime(df['ds'])

    return df

def trainXGBOOST(df, train_pct=0.8):
    xgboost_df = df.loc[:]

    xgboost_df['ds'] = pd.to_datetime(xgboost_df['ds'])
    xgboost_df.set_index('ds', inplace=True)

    # Assuming you have a DataFrame called 'df' with a DateTime index and a column 'weekly_data'
    # Set the frequency to daily and interpolate
    xgboost_df = xgboost_df.resample('D').asfreq().interpolate(method='linear')

    xgboost_df.reset_index(inplace=True)

    # print(xgboost_df.head())

    xgboost_df=create_lagged_features(xgboost_df, 7)

    xgboost_df['ds'] = xgboost_df['ds'].apply(lambda x: x.timestamp())
    # Split the data into training and testing sets
    train_size = int(len(xgboost_df) * train_pct)

    X_train = xgboost_df[['Lag_1','Lag_2','Lag_3','Lag_4','Lag_5','Lag_6','Lag_7']][:train_size]
    y_train = xgboost_df['y'][:train_size]

    # Convert to DMatrix
    # dtrain = xgb.DMatrix(data=X_train, label=y_train)

    # Example using numpy arrays
    # X_train = np.random.rand(500, 10) 
    # y_train = np.random.randint(2, size=500)
    # dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)

    # params = {
    #       'eta': 0.3,
    #       'max_depth': 10,
    #       'subsample': 0.9,
    #       'colsample_bytree': 0.9
    # }


    X_test = xgboost_df[['Lag_1','Lag_2','Lag_3','Lag_4','Lag_5','Lag_6','Lag_7']][train_size:]

    # model = xgb.train(params=params, dtrain=dtrain)

    # dtest = xgb.DMatrix(X_test)

    # predictions = model.predict(dtest)

    model = xgb.XGBRegressor(
        learning_rate =0.3,
        n_estimators=1000,
        max_depth=2,
        min_child_weight=1,
        gamma=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        objective= 'reg:squarederror',
        nthread=5
        # scale_pos_weight=1

        # objective='reg:squarederror', 
        # learning_rate=0.08,
        # max_depth= 50,
        # subsample= 0.8
        # colsample_bytree= 0.85
        )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # xgb.plot_importance(model)

    # #============================================
    # # Train a SHAP explainer on your XGBoost model
    # explainer_xgb = shap.Explainer(model)

    # # Explain a specific prediction or a set of predictions
    # explanation_xgb = explainer_xgb.shap_values(np.array(predictions).reshape(-1, 1))

    # # You can visualize the SHAP values as well
    # shap.summary_plot(explanation_xgb, np.array(predictions).reshape(-1, 1))

    # #============================================

    actual = xgboost_df['y'][train_size:]

    rmse = mean_squared_error(actual, predictions, squared=False)

    # MAPE 
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    times = xgboost_df['ds'][train_size:].apply(lambda x: datetime.datetime.fromtimestamp(x))

    # Print metrics
    print("train pct:",train_pct)
    print('RMSE:', rmse)
    print('MAPE:', mape)
    print()

    plt.plot(times, actual, label='Actual')
    plt.plot(times, predictions, label='Predicted')

    plt.plot(times, xgboost_df['Lag_1'][train_size:], label='Lag1')
    plt.plot(times, xgboost_df['Lag_2'][train_size:], label='Lag2')
    plt.plot(times, xgboost_df['Lag_3'][train_size:], label='Lag3')
    plt.plot(times, xgboost_df['Lag_4'][train_size:], label='Lag4')
    plt.plot(times, xgboost_df['Lag_5'][train_size:], label='Lag5')
    plt.plot(times, xgboost_df['Lag_6'][train_size:], label='Lag6')
    plt.plot(times, xgboost_df['Lag_7'][train_size:], label='Lag7')

    x_locator = MaxNLocator(nbins=5)  # Set the desired number of ticks

    # Apply the locator to the x-axis
    plt.gca().xaxis.set_major_locator(x_locator)

    plt.title('Sales Forecast')
    plt.ylabel('Sales')
    plt.xlabel('Time Period')
    plt.legend()

    plt.show()

def create_lagged_features(df, lag_count):
    '''
    Create lagged features for a given dataframe.

    Args:
    - df: pandas DataFrame
    - lag_count: int, number of lagged features to create

    Returns:
    - df: pandas DataFrame with lagged features added
    '''
    for i in range(1, lag_count+1):
        col_name = f'Lag_{i}'
        df[col_name] = df['y']-df['y'].shift(i*7)
    return df

def trainXGBOOSTWithCrossValidation(df, train_pct=0.8, num_boost_round=100, early_stopping_rounds=10, nfold=5):


    # Ignore future warning logs
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    df = create_lagged_features(df, 4)
    
    # Split data into train and test sets
    train_size = int(len(df) * train_pct)
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Define X and y
    X_train = train_df.drop(['y'], axis=1)
    y_train = train_df['y']
    X_test = test_df.drop(['y'], axis=1)
    y_test = test_df['y']

    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train.drop('ds', axis=1), label=y_train)
    
    dtest = xgb.DMatrix(X_test.drop('ds', axis=1), label=y_test)

    # Set XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    # Perform cross-validation
    cv_results = xgb.cv(params=params, dtrain=dtrain, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds, nfold=nfold)

    # Train final model
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=cv_results.shape[0])

    # Make predictions on test set
    y_pred = model.predict(dtest)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print(f'MAE: {mae:.2f}')
    print(f'MSE: {mse:.2f}')
    print(f'R2: {r2:.2f}')

    # Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    shap_values = shap.TreeExplainer(model).shap_values(X_train.drop('ds', axis=1))
    shap.summary_plot(shap_values, X_train.drop('ds', axis=1), plot_type='bar', show=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

    actual = df['y'][train_size:]

    rmse = mean_squared_error(actual, y_pred, squared=False)

    # MAPE 
    mape = np.mean(np.abs((actual - y_pred) / actual)) * 100

    times = df['ds'][train_size:].apply(lambda x: datetime.datetime.fromtimestamp(x.timestamp()))

    # Print metrics
    print("train pct:",train_pct)
    print('RMSE:', rmse)
    print('MAPE:', mape)
    print()

    plt.plot(times, actual, label='Actual')
    plt.plot(times, y_pred, label='Predicted')

    plt.plot(times, df['Lag_1'][train_size:], label='Lag1')
    plt.plot(times, df['Lag_2'][train_size:], label='Lag2')
    plt.plot(times, df['Lag_3'][train_size:], label='Lag3')
    plt.plot(times, df['Lag_4'][train_size:], label='Lag4')
    # plt.plot(times, xgboost_df['Lag_5'][train_size:], label='Lag5')
    # plt.plot(times, xgboost_df['Lag_6'][train_size:], label='Lag6')
    # plt.plot(times, xgboost_df['Lag_7'][train_size:], label='Lag7')

    x_locator = MaxNLocator(nbins=5)  # Set the desired number of ticks

    # Apply the locator to the x-axis
    plt.gca().xaxis.set_major_locator(x_locator)

    plt.title('Sales Forecast')
    plt.ylabel('Sales')
    plt.xlabel('Time Period')
    plt.legend()

    plt.show()

    return model

# def preprocessDataframeForRegion(df, region_name, avodaco_type = 'conventional'):
#     # your code here

# def trainXGBOOST(df, train_pct=0.8):
#     # your code here

# def create_lagged_features(df, lag_count):
#     # your code here

def trainXGBOOSTWithGridSearchCV(df, train_pct=0.8, num_boost_round=100, early_stopping_rounds=10, nfold=5):    
    # # Ignore future warning logs
    # warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # warnings.filterwarnings('ignore', category=ConvergenceWarning)
    # warnings.filterwarnings('ignore', category=FutureWarning)        
    # warnings.filterwarnings('ignore', category=DataConversionWarning)
    # warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
    # warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')
    # warnings.filterwarnings('ignore', message="is_sparse is deprecated and will be removed", module='xgboost')
    # warnings.filterwarnings('ignore', message="is_categorical_dtype is deprecated and will be removed", module='xgboost')

    # warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')
    # warnings.filterwarnings('ignore', message="is_sparse is deprecated and will be removed", module='xgboost')
    # warnings.filterwarnings('ignore', message="is_categorical_dtype is deprecated and will be removed", module='xgboost')
    
    # warnings.simplefilter(action='ignore', category=FutureWarning)

    df = create_lagged_features(df, 4)
    
    # Split data into train and test sets
    train_size = int(len(df) * train_pct)
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Define X and y
    X_train = train_df.drop(['y'], axis=1)
    y_train = train_df['y']
    X_test = test_df.drop(['y'], axis=1)
    y_test = test_df['y']

    print("train pct:",train_pct)
    
    # define the parameter grid to search over
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.2, 0.15, 0.1, 0.05, 0.01, 0.001],
        'n_estimators': [50, 100, 200],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        # 'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }

    # create an XGBoost regressor object
    xgb_model = xgb.XGBRegressor()

    # create a GridSearchCV object
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=nfold, n_jobs=-1, verbose=0)

    # fit the GridSearchCV object to the data
    grid_search.fit(X_train, y_train)

    # print the best parameters and score
    print("Best parameters found: ", grid_search.best_params_)
    print("Lowest RMSE found: ", np.sqrt(np.abs(grid_search.best_score_)))
    y_pred = grid_search.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("Best R2 score found: ", r2)
    
