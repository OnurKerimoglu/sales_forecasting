import itertools
import os
import numpy as np
import pandas as pd
import parquet


class RawData:
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
        float64cols = df.select_dtypes(include='float64').columns
        int64cols = df.select_dtypes(include='int64').columns
        df = df.astype({c: np.float32 for c in float64cols})
        df = df.astype({c: np.int32 for c in int64cols})
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
        # df['month'] = df['date'].dt.month
        # df['year'] = df['date'].dt.year
        # df['dayofweek'] = df['date'].dt.dayofweek
        df['monthly_period'] = df['date'].dt.to_period('M')
        return df

    def clean_data(self, df, rem_negs=True, rem_ol=True):
        """
        Clean the given df by marking rows with negative values and outliers
        in the 'price' and 'amount' columns as invalid.
        Args:
            df (pandas.DataFrame): The dataframe to be cleaned.
            rem_negs (bool): Whether to remove rows with negative values
            rem_ol (bool): Whether to remove rows with outliers
        Returns:
            pandas.DataFrame: The cleaned dataframe
        """
        df['valid'] = True
        if rem_negs:
            df = self.invalidate_negatives(df, ['price', 'amount'])
        if rem_ol:
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

    def get_monthly_data(
            self,
            df_daily,
            splitname,
            refresh
            ):
        fn_base = os.path.join(
            self.config['root_data_path'],
            self.config[f'fn_{splitname}_base'])
        if os.path.exists(fn_base) and not refresh:
            print(f'Loading {fn_base}')
            df_base = pd.read_parquet(fn_base)
        else:
            print(f'Creating {fn_base}')
            df_base = self.prep_monthly_data(
                df_daily,
                self.shop_list,
                self.item_list[['item_id', 'item_category_id']].copy())
            df_base.to_parquet(fn_base)
        return df_base

    def prep_monthly_data(self, df_daily, df_shops, df_items):
        # Create a full Item-Amounts table
        df_items_monthly = self.create_items_df_monthly(df_daily, df_shops, df_items) 
        # Add missing category_id's
        df_items_monthly = self.add_category_to_df(df_items_monthly, df_items)
        # df_items_monthly.info()
        # Add missing prices of non-transacted items: first based on average price of items in each shop, then average price of items in all shops, then average price of categories in all shops, then the global average price.
        df_items_monthly = self.add_avg_shopitem_price_to_df(df_items_monthly, df_daily, 'mean shop-item-specific price')
        # df_items_monthly.info()
        # Add category amounts as a feature
        # Create a full Item-Amounts table
        df_categories_monthly = self.create_categories_df_monthly(df_daily, df_shops, df_items)
        # Merge the items and categories tables
        df_monthly = pd.merge(
            df_items_monthly,
            df_categories_monthly,
            how='left',
            on=['shop_id', 'item_category_id', 'monthly_period'],
            suffixes=('_item', '_cat'))
        # df_monthly.info(show_counts=True)
        # add time features
        df_monthly = self.add_time_features(df_monthly)
        # set monthly_period as index
        df_monthly.set_index('monthly_period', inplace=True)
        return df_monthly

    def create_items_df_monthly(self, df_daily, df_shops, df_items):
        df_items_monthly_transactions = self.convert_items_daily_to_monthly(
            df_daily
        )
        # df_items_monthly_transactions.info()
        columns = ['shop_id', 'item_id', 'monthly_period']
        df_items_monthly_all = self.create_df_all(
            columns=columns,
            shops=df_shops['shop_id'].unique(),
            items=df_items['item_id'].unique(),
            dates=df_items_monthly_transactions['monthly_period'].unique()
        )
        # df_items_monthly_all.info()
        df_items_monthly = self.create_df_with_zero_sales(
            df_items_monthly_transactions,
            df_items_monthly_all,
            columns)
        # df_items_monthly.info()
        return df_items_monthly

    def convert_items_daily_to_monthly(self, df):
        df_items_monthly_grouped = df.groupby(
        ['shop_id', 'item_id', 'monthly_period']
        )
        df_items_monthly = df_items_monthly_grouped.agg(
            {
            'item_category_id': 'first',
            'price': 'mean',
            'amount': 'sum',
            }
        ).reset_index()
        return df_items_monthly

    def create_df_all(self, columns, shops, items, dates):
        # Generate all possible combinations of shops, items, and dates.
        all_combinations = list(itertools.product(shops, items, dates))
        df_all = pd.DataFrame(all_combinations, columns=columns)
        return df_all

    def create_df_all_limited(self, df, coldict):
        shop_item_pairs = df[[coldict['shops'], coldict['items']]].drop_duplicates()

        dates = df[coldict['date']].unique()

        # Create a list of all possible combinations of shop-item pairs with the dates
        df_all = pd.DataFrame(
            list(itertools.product(shop_item_pairs.values, dates)),
            columns=['shop_item', coldict['date']])
        # Split the 'shop_item' column back into separate 'shop_id' and 'item_id' columns
        df_all[[coldict['shops'], coldict['items']]] = pd.DataFrame(
            df_all['shop_item'].tolist(), 
            index=df_all.index)
        # Drop the intermediate 'shop_item' column
        df_all = df_all.drop(columns=['shop_item'])
        return df_all

    def create_df_with_zero_sales(self, df, df_all, columns):
        # Merge with the original dataframe
        df_merged = pd.merge(
            df_all,
            df,
            on=columns,
            how='left')
        # Fill missing values with 0
        # df_merged['amount'].fillna(0, inplace=True)
        df_merged.fillna({'amount': 0}, inplace=True)
        return df_merged

    def create_categories_df_monthly(self, df_daily, df_shops, df_items):
        df_categories_monthly_transactions = self.convert_categories_daily_to_monthly(
            df_daily
        )
        # df_categories_monthly_transactions.info()
        columns = ['shop_id', 'item_category_id', 'monthly_period']
        df_categories_monthly_all = self.create_df_all(
            columns=columns,
            shops=df_shops['shop_id'].unique(),
            items=df_items['item_category_id'].unique(),
            dates=df_categories_monthly_transactions['monthly_period'].unique()
        )
        # df_categories_monthly_all.info()
        df_categories_monthly = self.create_df_with_zero_sales(
            df_categories_monthly_transactions,
            df_categories_monthly_all,
            columns)
        #df_categories_monthly.info()
        return df_categories_monthly

    def convert_categories_daily_to_monthly(self, df):
        df_categories_monthly_grouped = df.groupby(
        ['shop_id', 'item_category_id', 'monthly_period']
        )
        df_categories_monthly = df_categories_monthly_grouped.agg(
            {
            'amount': 'sum',
            }
        ).reset_index()
        return df_categories_monthly

    def add_category_to_df(self, df_monthly, df_items):
        df_monthly_full = df_monthly.loc[df_monthly['item_category_id'].notna(), :]  #.copy()
        df_monthly_missing = df_monthly.loc[df_monthly['item_category_id'].isna(), :].copy()
        df_monthly_missing.drop(['item_category_id'], axis=1, inplace=True)
        df_monthly_missing_filled = pd.merge(
            df_monthly_missing,
            df_items,
            on='item_id',
            how='left')
        df_items_monthly = pd.concat([df_monthly_full, df_monthly_missing_filled], ignore_index=True)
        df_items_monthly = df_items_monthly.sort_values(by = ['monthly_period', 'shop_id', 'item_id'])
        count_missing_cats = df_items_monthly.loc[df_items_monthly['item_category_id'].isna(), :].shape[0]
        print(f'after the operation, count of rows with missing categories: {count_missing_cats}')
        return df_items_monthly

    def add_avg_shopitem_price_to_df(self, df_monthly, df_daily_train, method):
        df_monthly_full = df_monthly.loc[df_monthly['price'].notna(), :]
        df_monthly_miss = df_monthly.loc[df_monthly['price'].isna(), :].copy()
        print(f'{df_monthly_full.shape[0]} and {df_monthly_miss.shape[0]} rows with filled and missing prices, respectively.')
        if df_monthly_miss.shape[0] > 0:
            print(f'Filling missing with {method}')
            df_monthly_miss.drop(['price'], axis=1, inplace=True)
            mean_price, merge_columns = self.get_mean_price(df_daily_train, method)
            # fill the missing
            df_monthly_miss_filled = pd.merge(
                df_monthly_miss, 
                mean_price,
                on=merge_columns,
                how='left')
            df_monthly = pd.concat([df_monthly_full, df_monthly_miss_filled], ignore_index=True)
            df_monthly = df_monthly.sort_values(by = ['monthly_period', 'shop_id', 'item_id'])
            count_missing_price = df_monthly.loc[df_monthly['price'].isna(), :].shape[0]
            print(f'after the operation, count of rows with missing price: {count_missing_price}')
            if count_missing_price > 0: 
                if method == 'mean shop-item-specific price':
                    df_monthly = self.add_avg_shopitem_price_to_df(df_monthly, df_daily_train, 'mean item-specific price')
                elif method == 'mean item-specific price':
                    df_monthly = self.add_avg_shopitem_price_to_df(df_monthly, df_daily_train, 'mean category-specific price')
                elif method == 'mean category-specific price':
                    print('Filling missing with global average price')
                    global_avg_price = df_daily_train[['price']].mean().values[0]
                    df_monthly.loc[df_monthly['price'].isna(), 'price'] = float(global_avg_price)
                else:
                    raise ValueError(f'Uknown method: {method}')
        else:
            df_monthly = df_monthly_full
        return df_monthly

    def get_mean_price(self, df_daily_train, method):
        if method == 'mean shop-item-specific price':
            # calculate mean shop-item price:
            mean_price = df_daily_train[['shop_id', 'item_id', 'price']].groupby(['shop_id', 'item_id']).mean().reset_index()
            merge_columns = ['shop_id', 'item_id']
        elif method == 'mean item-specific price':
            mean_price = df_daily_train[['item_id', 'price']].groupby(['item_id']).mean().reset_index()
            merge_columns = ['item_id']
        elif method == 'mean category-specific price':
            mean_price = df_daily_train[['item_category_id', 'price']].groupby(['item_category_id']).mean().reset_index()
            merge_columns = ['item_category_id']
        else:
            raise ValueError(f'Uknown method: {method}')
        return mean_price, merge_columns

    # Year and month can help capturing the trend and the seasonality, respectively
    def add_time_features(self, df_monthly):
        df_monthly['year'] = df_monthly['monthly_period'].dt.year
        df_monthly['month'] = df_monthly['monthly_period'].dt.month
        return df_monthly

    def get_ts_data(self, df_base, splitname, refresh, num_lag_mon):
        fn_ts = os.path.join(
            self.config['root_data_path'],
            self.config[f'fn_{splitname}_ts'])
        if os.path.exists(fn_ts) and not refresh:
            print(f'Loading {fn_ts}')
            df_ts = pd.read_parquet(fn_ts)
        else:
            print(f'Creating {fn_ts}')
            df_ts = self.add_lag_ma_features(
                df_base,
                lags_to_include = num_lag_mon)
            # remove the months for which lags could not be calculated
            periods_to_remove = df_ts.index.unique()[0:num_lag_mon]
            df_ts = df_ts.drop(periods_to_remove)
            df_ts.to_parquet(fn_ts)
        return df_ts

    def add_lag_ma_features(
            self,
            df_monthly,
            lags_to_include=3,
            lag_features=['price', 'amount_item', 'amount_cat'],
            mas_to_include=[2],
            ma_features=['price_l1', 'amount_item_l1', 'amount_cat_l1']
            ):
        for feature in lag_features:
            df_monthly = self.add_feature_lags(
                df_monthly,
                feature,
                lags_to_include)
        for feature in ma_features:
            df_monthly = self.add_feature_moving_averages(
                df_monthly,
                feature,
                mas_to_include)
        return df_monthly

    def add_feature_lags(self, df, column, lagcount):
        for lag in range(1, lagcount+1):
            new_column_name = column + '_l' + str(lag)
            df[new_column_name] = df.groupby(['shop_id', 'item_id'])[column].shift(lag)
        return df

    def add_feature_moving_averages(self, df, column, windows):
        for window in windows:
            new_column_name = column + '_ma' + str(window)
            df[new_column_name] = df.groupby(['shop_id', 'item_id'])[column].transform(lambda x: x.rolling(window=window).mean())
        return df

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
#     data_cleaned.info()
