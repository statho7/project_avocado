from objects.column import Column

class Region:
    def __init__(self, name, volume, df):
        self._name=name
        self._volume=volume
        
        self._df = df[df['region']==name]
        self._conventional_df = self._df[self._df.type=='conventional']
        self._organic_df = self._df[self._df.type=='organic']

        self._df_columns = extractColumns(self._df)
        self._conventional_df_columns = extractColumns(self._conventional_df)
        self._organic_df_columns = extractColumns(self._organic_df)
    
    def getName(self):
        '''returns the name of the region'''
        return self._name
    
    def getVolume(self):
        '''returns the volume of entries of the region'''
        return self._volume
    
    def getDataframe(self):
        '''returns the region dataframe'''
        return self._df[:]
    
    def getColumns(self):
        '''returns the columns of the region dataframe'''
        return self._df_columns
    
    def plotColumns(self):
        '''shows the plot of selected columns from the dataframe of the region.
        The selected columns are: AveragePrice, Total Volume, 4046, 4225, 4770, Total Bags, Small Bags, Large Bags, XLarge Bags'''
        for col in self._df_columns[1:-4]:
            col.plot()
        
        self._df_columns[-1].plot()
    
    def plotColumn(self, name):
        '''shows the plot of selected columns from the dataframe of the region.
        The selected columns are: AveragePrice, Total Volume, 4046, 4225, 4770, Total Bags, Small Bags, Large Bags, XLarge Bags'''  
        
        col = findColumn(name, self._df_columns)
        
        if col != None:
            col.plot()
                
    def getColumnStatistics(self, name):
        col = findColumn(name, self._df_columns)
        
        if col != None:
            col.statistics()
                
    def PrintStatisticsOfColumns(self):
        for col in self._df_columns[1:-4]:
            col.statistics()
                
        self._df_columns[-1].statistics()

    def plotColumnAndPrintStatisticsOfIt(self, name):
        col = findColumn(name, self._df_columns)
        
        if col != None:
            col.plot()
            col.statistics()
                
    def plotColumnsAndPrintStatisticsOfThem(self):
        for col in self._df_columns[1:-4]:
            col.plot()
            col.statistics()

        self._df_columns[-1].plot()
        self._df_columns[-1].statistics()

    def getConventionalDataframe(self):
        '''returns the region dataframe of entries that have type of avocado conventional'''
        return self._conventional_df[:]
    
    def getOrganicDataframe(self):
        '''returns the region dataframe of entries that have type of avocado organic'''
        return self._organic_df[:]
    
    def getSalesOfConventional(self):
        return self._conventional_df['TotalSales'].sum()
    
    def getSalesOfOrganic(self):
        return self._organic_df['TotalSales'].sum()

    # def saveDataframeToCSV(self):
    #     '''saves the dataframe of the region as csv file'''
    #     self._df.to_csv(f'./data/combined/{self._name}_data.csv', index=False)
    
    # def saveConventionalDataframeToCSV(self):
    #     '''saves the dataframe of the region that has entries of type conventional as csv file'''
    #     self._conventional_df.to_csv(f'./data/conventional/{self._name}_data.csv', index=False)
    
    # def saveOrganicDataframeToCSV(self):
    #     '''saves the dataframe of the region that has entries of type organic as csv file'''
    #     self._organic_df.to_csv(f'./data/organic/{self._name}_data.csv', index=False)
    
    def getHeadOfDataframe(self, rows=5):
        '''returns the head of the region dataframe based on the rows provided'''
        return self._df.head(rows)
    
    def details(self):
        '''details of the region'''
        print(f'Region: {self._name}, Volume: {self._volume}')

def findColumn(name, df_columns):
    for col in df_columns:
        if name == col.getName():
            return col
        
    return None
            
def extractColumns(df):
    '''extracts the columns out of the df'''
    df_columns = list(df)

    _df_columns=list()

    for column in df_columns:
        col = Column(column, df[[column]])
        
        _df_columns.append(col)

    return _df_columns