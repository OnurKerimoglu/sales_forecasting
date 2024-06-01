import os
import pandas as pd


class Data:
    """
    A class used to load and clean data.
    Attributes:
        config (dict): The configuration parameters.
        category_list (pandas.DataFrame): The loaded category list data.
        item_list (pandas.DataFrame): The loaded item list data.
        shop_list (pandas.DataFrame): The loaded shop list data.
        transaction (pandas.DataFrame): The loaded transaction data.
        test (pandas.DataFrame): The loaded test data.
    Methods:
        load_raw_data(self)
        load_all_raw_data_from_csv(self)
        fix_data_schemas(self)
        merge_data(self)
        handle_dates(self, df)
        clean_data(self, df)
        find_outlier_limits_iqr(self, df, column)
        find_outlier_limits_std(self, df, column)
        invalidate_negatives(self, df, columns)
        invalidate_outliers(self, df, columns, method='std')
    """
    def __init__(self, config):
        """
        Initializes a new instance of the RawData class.
        Args:
            config (dict, optional): A dictionary of configuration parameters.
        Initializes the following instance variables:
            - config (dict)
            - category_list (pandas.DataFrame)
            - item_list (pandas.DataFrame)
            - shop_list (pandas.DataFrame)
            - transaction (pandas.DataFrame)
            - test (pandas.DataFrame)
        If the env is set to 'local', known  data schema issues are fixed.
        """
        self.config = config

        # load data
        self.load_raw_data()

        # fix data schemas
        if self.config['env'] == 'local':
            self.fix_data_schemas()

    def load_raw_data(self):
        """
        Load raw data based on the environment configuration.
        Args:
            self (RawData): The instance of the RawData class.
        """

        if self.config['env'] == 'local':
            self.load_all_raw_data_from_csv()

    def load_all_raw_data_from_csv(self):
        """
        Load all raw data from CSV files.
        Args:
            self (RawData): The instance of the RawData class.
        """
        print('Loading data..', end=' ')
        self.category_list = self.load_from_csv(self.config['fn_categories'])
        self.item_list = self.load_from_csv(self.config['fn_items'])
        self.shop_list = self.load_from_csv(self.config['fn_shops'])
        self.transactions = self.load_from_csv(self.config['fn_transactions'])
        self.test = self.load_from_csv(self.config['fn_test'])
        print('Done.')

    def load_from_csv(self, fname: str):
        """
        Load data from a CSV file.
        Args:
            fname (str): The name of the CSV file to load.
        """
        fpath = os.path.join(self.config['root_data_path'], fname)
        df = pd.read_csv(fpath)
        # remove the unnecessary columns
        cols_to_remove = df.columns.str.contains('Unnamed: 0', case=False)
        df.drop(df.columns[cols_to_remove], axis=1, inplace=True)
        return df

    def fix_data_schemas(self):
        """
        Fixes known schema issues
        Args:
            self (object): The instance of the class.
        """
        print('Fixing data schemas..', end=' ')
        self.transactions.rename(
            columns={
                'shop': 'shop_id',
                'item': 'item_id'
                },
            inplace=True
            )
        print('Done.')

    def merge_data(self):
        """
        Merges data from different dataframes into a single dataframe.
        Args:
            self (object): The instance of the class.
        Returns:
            pandas.DataFrame: The combined df with all the necessary data.

        """
        data_merged = self.transactions
        # combine item_name and category_ids from item_list
        data_merged = data_merged.merge(
            self.item_list[['item_id', 'item_name', 'item_category_id']],
            how='left',
            on='item_id')
        # combine category_name from category_list
        data_merged = data_merged.merge(
            self.category_list[['item_category_id', 'item_category_name']],
            how='left',
            on='item_category_id')
        # combine shop_names from shop_list
        data_merged = data_merged.merge(
            self.shop_list[['shop_id', 'shop_name']],
            how='left',
            on='shop_id')
        return data_merged

    def handle_dates(self, df):
        """
        Add 'month' and 'year' columns to the df.
        Args:
            df (pandas.DataFrame): The DataFrame to be modified.
        Returns:
            pandas.DataFrame: The modified DF with 'month' and 'year' columns.
        """
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
        # Add year and month
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofweek'] = df['date'].dt.dayofweek
        return df

    def clean_data(self, df):
        """
        Clean the given df by marking rows with negative values and outliers
        in the 'price' and 'amount' columns as invalid.
        Args:
            df (pandas.DataFrame): The dataframe to be cleaned.
        Returns:
            pandas.DataFrame: The cleaned dataframe
        """
        df['valid'] = True
        df = self.invalidate_negatives(df, ['price', 'amount'])
        df = self.invalidate_outliers(df, ['price', 'amount'])
        df_clean = df.loc[df['valid'], :]
        print(f'Count of cleaned rows: {df.shape[0]-df_clean.shape[0]}')
        df_clean = df_clean.drop('valid', axis=1)
        return df_clean

    @staticmethod
    def invalidate_negatives(df, columns):
        """
        Invalidates rows in a df that have negative values in given columns.
        Args:
            df (pandas.DataFrame): The DataFrame to be modified.
            columns (list): The list of columns to check for negative values.
        Returns:
            pandas.DataFrame: The DF with invalid rows marked as False
        """
        print(f'Checking for negative values in {columns}..', end=' ')
        count_invalid = 0
        for column in columns:
            ind = df[column] < 0
            df.loc[ind, 'valid'] = False
            count_invalid += ind.sum()
        print(f'Count of rows marked as invalid: {count_invalid}')
        return df

    def invalidate_outliers(self, df, columns, method='std'):
        """
        Marks the rows of a DataFrame to be invalid
        if they fall outside the lower and upper limits of the given columns.
        Args:
            df (pandas.DataFrame): The DataFrame to be modified.
            columns (list): The list of column names to check for outliers.
            method (str): The method to use for outlier detection
                ('iqr' or 'std'(default)).
        Returns:
            pandas.DataFrame: The modified DF with outliers marked as invalid.
        """
        print(f'Checking for outliers in {columns}..', end=' ')
        count_invalid = 0
        for column in columns:
            if method == 'iqr':
                lower_lim, upper_lim = self.find_outlier_limits_iqr(df, column)
            elif method == 'std':
                lower_lim, upper_lim = self.find_outlier_limits_std(df, column)
            else:
                raise ValueError('Invalid method. Choose from iqr or std.')
            ind = (df[column] < lower_lim) | (df[column] > upper_lim)
            df.loc[ind, 'valid'] = False
            count_invalid += ind.sum()
        print(f'Count of rows marked as invalid: {count_invalid}')
        return df

    @staticmethod
    def find_outlier_limits_iqr(df, column, th1=0.25, th3=0.75):
        """
        Determine the lower and upper limits for outliers
        based on the Interquartile Range (IQR) method.
        Args:
            df (pandas.DataFrame): The DataFrame.
            column (str): The name of the column to check for outliers.
            th1 (float, optional): The lower quantile value for
                determining the lower limit. Defaults to 0.25.
            th3 (float, optional): The upper quantile value for
                determining the upper limit. Defaults to 0.75.
        Returns:
            tuple: A tuple containing the lower and upper limits for outliers.
        """
        quartile1 = df[column].quantile(th1)
        quartile3 = df[column].quantile(th3)
        iqr = quartile3 - quartile1
        upper_limit = quartile3 + 1.5 * iqr
        lower_limit = quartile1 - 1.5 * iqr
        return lower_limit, upper_limit

    @staticmethod
    def find_outlier_limits_std(df, column):
        """
        Determine the lower and upper limits for outliers
        based on the std method.
        Args:
            df (pandas.DataFrame): The DataFrame.
            column (str): The name of the column to check for outliers.
        Returns:
            tuple: A tuple containing the lower and upper limits for outliers.
        """
        upper_limit = df[column].mean() + 3 * df[column].std()
        lower_limit = df[column].mean() - 3 * df[column].std()
        return lower_limit, upper_limit


# # for testing:
# if __name__ == "__main__":
#     from utils import Utils
#     data = Data(Utils.read_config_for_env())
#     # create a combined table to ease data processing and visualisation
#     data_merged = data.merge_data()
#     # create date objects and add month and year
#     data_merged = data.handle_dates(data_merged)
#     # clean the data from negative values and outliers for price and amount
#     data_cleaned = data.clean_data(data_merged)
