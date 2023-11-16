import seaborn as sns
import matplotlib.pyplot as plt


class Column:
    def __init__(self, name, column_df):
        self._name=name
        self._df=column_df
    
    def getName(self):
        '''returns the name of the column'''
        return self._name
    
    def getDataframe(self):
        '''returns the dataframe of the column'''
        return self._df[:]
    
    def getNullContents(self):
        '''returns the name of the column'''
        return len(self._df[self._df.isna() == True])
    
    def statistics(self):
        '''returns the name of the column'''
        mean_age = self._df[self._name].mean()
        median_age = self._df[self._name].median()
        std_deviation_age = self._df[self._name].std()

        print(f"      {self._name}")
        print(f"Mean     : {mean_age}")
        print(f"Median   : {median_age}")
        print(f"St. Dev. : {std_deviation_age}")
        print(f" "*20)
    
    def plot(self):
        '''Creates and show the plot for the distribution of the column'''

        # Plot the distribution of the 'Age' column using Seaborn
        sns.set(style="whitegrid")  # Set the style of the plot
        plt.figure(figsize=(16, 8))  # Set the figure size
        
        # Create a distribution plot (histogram with a kernel density estimate)
        sns.histplot(self._df[self._name], kde=True, bins=10, color='blue')

        # Add labels and a title
        plt.xlabel(self._name)
        plt.ylabel('Density')
        plt.title(f'Distribution of {self._name}')

        # Show the plot
        plt.show()
    
    def plotTrend(self):

        # Plot the distribution of the 'Age' column using Seaborn
        sns.set(style="whitegrid")  # Set the style of the plot
        plt.figure(figsize=(16, 8))  # Set the figure size
        
        # Create a distribution plot (histogram with a kernel density estimate)
        sns.histplot(self._df[self._name], kde=True, bins=10, color='blue')
        sns.lmplot(data=self._df, x="Date", y=self._name)

        # # Add labels and a title
        # plt.xlabel(self._name)
        # plt.ylabel('Density')
        # plt.title(f'Distribution of {self._name}')

        # Show the plot
        plt.show()