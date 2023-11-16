from objects.region import Region
from objects.column import Column

import pandas as pd
import matplotlib.pyplot as plt

def plotRegionObservations(regions):
    '''returns the columns of the region dataframe'''
    min_value = regions.min()
    max_value = regions.max()
    plt.hist(regions, bins=max_value - min_value + 1, range=(min_value - 0.5, max_value + 0.5), edgecolor='k')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Integers')
    plt.xticks(range(min_value, max_value+1))
    plt.show()

def findRegionsByNumberObservations(regions, number_of_observations):
    for region_name, region_observations in regions.items():
        if region_observations == number_of_observations:
            print(region_name, region_observations)

def findRegionsByName(regions, name):
    for region_name, region_observations in regions.items():
        if region_name == name:
            print(region_name, region_observations)

def boxplotsOfDataframe(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
        
    plt.figure(figsize=(5,162))

    # Create plot 
    fig, axs = plt.subplots(nrows=len(numeric_cols[:-1]), figsize=(5,18))

    for i, col in enumerate(numeric_cols[:-1]):
        ax = axs[i]
        
        # Plot boxplot 
        df[col].plot.box(ax=ax)
        
        # Set title
        ax.set_title(col)
        
    # Tight layout   
    plt.tight_layout()

def boxplotOfColumn(df, col):        
    plt.figure(figsize=(5,10))
    
    # Plot boxplot 
    df[col].plot.box()
        
    plt.show()

def histogramsOfDataframe(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
        
    plt.figure(figsize=(8,315))

    # Create plot 
    fig, axs = plt.subplots(nrows=len(numeric_cols[1:-2]), figsize=(8,35))

    for i, col in enumerate(numeric_cols[1:-2]):
        ax = axs[i]
        
        # Plot histogram 
        df[col].plot.hist(ax=ax)
        
        # Set title
        ax.set_title(col)
        
    # Tight layout   
    plt.tight_layout()

def plotConventionalAvocadosAveragePriceForYears(df):
    years_df = df.loc[:]
    years_df.reset_index()

    years_df['Date'] = pd.to_datetime(years_df['Date'])
    years_df['Month'] = years_df['Date'].dt.month
    years_df['Day'] = years_df['Date'].dt.day
    years_df['DayMonth'] = years_df['Month'].astype(str) + '-' + years_df['Day'].astype(str)

    # Create a line plot
    years = years_df['year'].unique()
    fig, ax = plt.subplots()
    for year in years[:-1]:
        df_yr = years_df[years_df['year'] == year]
        ax.plot(df_yr['WeekOfYear'], df_yr['Conventional'], label=f'Conventional_{year}')

    # Add labels and a legend
    ax.set_xlabel('WeekOfYear')
    ax.set_ylabel('Average Price')
    ax.legend()
    plt.title('Price Trend for Conventional avocados')
    plt.show()

def plotOrganicAvocadosAveragePriceForYears(df):
    years_df = df.loc[:]
    years_df.reset_index()

    years_df['Date'] = pd.to_datetime(years_df['Date'])
    years_df['Month'] = years_df['Date'].dt.month
    years_df['Day'] = years_df['Date'].dt.day
    years_df['DayMonth'] = years_df['Month'].astype(str) + '-' + years_df['Day'].astype(str)

    # Create a line plot
    years = years_df['year'].unique()
    fig, ax = plt.subplots()
    for year in years[:-1]:
        df_yr = years_df[years_df['year'] == year]
        ax.plot(df_yr['WeekOfYear'], df_yr['Organic'], label=f'Organic_{year}')

    # Add labels and a legend
    ax.set_xlabel('WeekOfYear')
    ax.set_ylabel('Average Price')
    ax.legend()
    plt.title('Price Trend for Organic avocados')
    plt.show()


def plotAveragePriceTrend(result_df):
    # Calculate daily price difference  
    result_df['PriceDiffOrganic'] = result_df['Organic'].diff()
    result_df['PriceDiffConventional'] = result_df['Conventional'].diff()

    # # Calculate percentage difference
    # result_df['PriceDiffPercentOrganic'] = result_df['PriceDiffOrganic']/result_df['Organic'].shift(1) * 100
    # result_df['PriceDiffPercentConventional'] = result_df['PriceDiffConventional']/result_df['Conventional'].shift(1) * 100

    # # Plot average price over date
    plt.figure(figsize=(24,8))
    # Plot trends
    # plt.plot(result_df['Date'], result_df['PriceDiff'], label='Daily Difference')
    plt.plot(result_df['WeekOfYear'], result_df['PriceDiffOrganic'], label='Organic Daily % Change')
    plt.plot(result_df['WeekOfYear'], result_df['PriceDiffConventional'], label='Conventional Daily % Change')

    plt.legend()
    plt.ylabel('USD Price Change')
    plt.title('Avocado Price Trends')


def plotAveragePriceTrendPerYear(result_df, avocado_type):
    # Calculate daily price difference  
    result_df[f'PriceDiff{avocado_type}'] = result_df[avocado_type].diff()

    # # Calculate percentage difference
    # result_df['PriceDiffPercentOrganic'] = result_df['PriceDiffOrganic']/result_df['Organic'].shift(1) * 100
    # result_df['PriceDiffPercentConventional'] = result_df['PriceDiffConventional']/result_df['Conventional'].shift(1) * 100

    # # Plot average price over date
    plt.figure(figsize=(24,8))
    # Plot trends
    # plt.plot(result_df['Date'], result_df['PriceDiff'], label='Daily Difference')
    plt.plot(result_df[result_df.year==2015]['WeekOfYear'], result_df[result_df.year==2015][f'PriceDiff{avocado_type}'], label=f'{avocado_type} Daily % Change 2015')
    plt.plot(result_df[result_df.year==2016]['WeekOfYear'], result_df[result_df.year==2016][f'PriceDiff{avocado_type}'], label=f'{avocado_type} Daily % Change 2016')
    plt.plot(result_df[result_df.year==2017]['WeekOfYear'], result_df[result_df.year==2017][f'PriceDiff{avocado_type}'], label=f'{avocado_type} Daily % Change 2017')

    plt.legend()
    plt.ylabel('USD Price Change')
    plt.title('Avocado Price Trends')
