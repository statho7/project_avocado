from region import Region
from column import Column

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
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(8, 10)) 
            df[col].plot.box()
            plt.title(f'Boxplot for {col}')

def histogramsOfDataframe(df):
    # df_columns = list(df)

    # for column in df_columns[:-3]:
    #     col = Column(column, df[[column]])
    #     if pd.api.types.is_numeric_dtype(df[column]):
    #         col.plot()

    numeric_cols = df.select_dtypes(include=['number']).columns
        
    plt.figure(figsize=(5, 5))

    # Create plot 
    fig, axs = plt.subplots(nrows=len(numeric_cols), figsize=(12*len(numeric_cols), 8))

    for i, col in enumerate(numeric_cols):
        ax = axs[i]
        
        # Plot histogram 
        df[col].plot.hist(ax=ax)
        
        # Set title
        ax.set_title(col)
        
    # Tight layout   
    plt.tight_layout()  



