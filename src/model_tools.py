from datetime import datetime
from dateutil.relativedelta import relativedelta
import joblib
import numpy as np
import os
import pandas as pd
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error as mse 
from sklearn.model_selection import train_test_split

# local imports
from data_tools import RawData, MonthlyData

class BasePredictor:
    """
    The BasePredictor class provides base attributes and methods needed by predictor classes
    Attributes:
        pred_data (PredictorData object): The data used for prediction.
    Methods:
        load_model(self, fname): Load a model from a file.
        save_model(self, model, fnameroot): Saves a model with a filename suffixed with the current date and time.
        split_transform(self, pred_data, transformer): Splits the data into training and validation sets, and transforms the data using the defined transformer.
        get_clean_feature_names_out(transformer): Get the clean feature names from the given transformer.
    """
    def __init__(
            self,
            pred_data):
        """
        Initializes the TF_NN_Predictor object with the specified parameters.
        Args:
            pred_data (PredictorData object)
        Returns:
            None
        """
        # input arguments
        self.pred_data = pred_data

    def load_model(self, fname):
        """
        Load a model from a file.
        Args:
            fname (str): The name of the file to load the model from.
        Returns:
            object: The loaded model.
        """
        fabspath = os.path.join(
            self.pred_data.config['root_data_path'],
            fname)
        model = joblib.load(fabspath)
        return model

    def save_model(self, model, fnameroot):
        """
        Saves a model with a filename suffixed with the current date and time.
        Args:
            model (object): The model to be saved.
            fnameroot (str): The root name for the file.
        Returns:
            None
        """
        v = datetime.now().strftime("%Y%m%d_%H%M%S")
        fabspath = os.path.join(
            self.pred_data.config['root_data_path'],
            f'{fnameroot}_{v}.pkl')
        print(f'Saving model to {fabspath}')
        joblib.dump(model, fabspath)

    def split_transform(self, pred_data, transformer):
        """
        Splits the data into training and validation sets,
        and transforms the training and validation using the defined transformer.
        Args:
            pred_data (PredictorData object)
            transformer (sklearn.compose.ColumnTransformer object)
        Returns:
            pred_data (PredictorData object): now containing:
                (transformed) X_train and X_val data
                y_train, y_val data
        """
        # split the data and do the scaling:
        # stores X_train, y_train, X_val, y_val in pred_data object
        print('Splitting train-val')
        pred_data.split_X_y()
        # transform the train data
        print('Fit-transforming X_train')
        pred_data.X_train = transformer.fit_transform(
            pred_data.X_train,
            pred_data.y_train)
        # Transform the val data
        print('Transforming X_val')
        pred_data.X_val = transformer.transform(
            pred_data.X_val)
        pred_data.transformed_feature_names = self.get_clean_feature_names_out(transformer)
        return pred_data
    
    @staticmethod
    def get_clean_feature_names_out(transformer):
        """
        Get the clean feature names from the given transformer.
        Args:
            transformer (sklearn.compose.ColumnTransformer): The transformer object.
        Returns:
            List[str]: The list of clean feature names.
        """
        feats_raw = transformer.get_feature_names_out()
        feats_clean = [feat_raw.replace('remainder__', '') for feat_raw in feats_raw]
        return feats_clean
        

class PredictorData:
    """
    The PredictorData class provides methods to prepare the final model data

    Attributes:
        config (dict): The configuration parameters.
        refresh_monthly (bool): Whether to refresh the monthly data.
        refresh_ts_features (bool): Whether to refresh the time series features.
        clean_strategy (str): The strategy for cleaning the data.
        split_strategy (str): The strategy for splitting the data.
        num_lag_mon (int): The number of lag months to include.
        val_ratio (float): The ratio of validation data.
        scaler_type (str): The type of scaler to use.
        raw_data (RawData): An instance of the RawData class.
        monthly_data (MonthlyData): An instance of the MonthlyData class.
        df_daily_train (pandas.DataFrame): The daily training data.
        df_daily_val (pandas.DataFrame): The daily validation data.
        df_train (pandas.DataFrame): The training data.
        df_val (pandas.DataFrame): The validation data.
        X_train (numpy.ndarray): The training features.
        y_train (numpy.ndarray): The training targets.
        X_val (numpy.ndarray): The validation features.
        y_val (numpy.ndarray): The validation targets.
        feature_names (list): The names of the features.
    Methods:
        prep_data(self): Prepares the monthly data with features for the model.
        set_feature_names(self, df): Assigns feature names to the object.
        get_numeric_features(self): Calculates a list of numeric feature names.
        fix_data_types(self, df): Fix the data types of the input DataFrame.
        prep_monthly_data_for_split(self, columns, splitname): Prepares monthly data for splitting.
        create_monthly_data(self, columns, splitname): Creates monthly data for a given split.
        prep_raw_data(self): Prepares the raw data.
        prep_merged_data(self): Prepares the merged data.
        find_num_val_mon(self, df): Finds the number of months to use for validation.
        split_train_test_dailydata(self, df): plits the given daily data into training and testing sets.
        split_train_test_dailydata_random(self, df): Splits the given daily data into training and validation sets randomly.
        split_train_test_dailydata_by_months(self, df): Splits the given daily data into training and testing sets based on months.
        split_train_test_dailydata_last_months_val(self, df, num_val_mon): Splits the given daily data into training and testing sets
        clean_daily_data(self): Cleans the daily data.
        create_ts_features(self, df_base, num_lag_mon): Create time series features for the given DataFrame.
        split_X_y(self): Splits the input data into training X,y and validation X,y
        split_train_test_data(self, df):Splits the given DataFrame into training and validation sets.
        split_train_test_data_random(self, df):  Splits the given DataFrame into training and validation sets randomly.
        split_train_test_data_by_months(self, df): Splits the given DataFrame into training and validation sets based on months.
        split_train_test_data_last_months_val(self, df, num_val_mon): Splits the given DataFrame into training and validation sets based on the last months.
        get_X_y_for_split(self, df): get X, y for a given df.
    """
    
    def __init__(
            self,
            config,
            refresh_monthly=False,
            refresh_ts_features=False,
            clean_strategy='olrem_for_all',
            split_strategy='random',
            num_lag_mon=3,
            val_ratio=0.2
            ):
        """
        Initializes a new instance of the BasePredictor class with the given configuration.

        Args:
            config (dict):
                A dictionary containing the configuration parameters.
            refresh_monthly (bool, optional):
                Whether to refresh the monthly data. Defaults to False.
            refresh_ts_features (bool, optional):
                Whether to refresh the time series features. Defaults to False.
            split_strategy (str, optional): 
                The strategy for splitting the data. Defaults to 'random'.
                Options: 'random', 'months', 'last_months_val'
            clean_strategy (str, optional): 
                The strategy for cleaning the data. Defaults to 'olrem_for_all'.
                Options: 'olrem_for_all', 'no_olrem_for_val'
            num_lag_mon (int, optional): 
                The number of lag months to include. Defaults to 3.
            val_ratio (float, optional): 
                The ratio of validation data. Defaults to 0.2.
        """
                
        # set input args
        self.config = config
        self.refresh_monthly = refresh_monthly
        self.refresh_ts_features = refresh_ts_features
        self.split_strategy = split_strategy
        self.clean_strategy = clean_strategy
        self.num_lag_mon = num_lag_mon
        self.val_ratio = val_ratio
        
        # predefined constants
        self.cols_to_drop = ['price', 'amount_item', 'amount_cat']
        self.cat_features = ['shop_id', 'item_id', 'item_category_id'] 
        self.seasonal_features = ['month']

        # define attributes to be set later
        self.raw_data = None
        self.monthly_data = None
        self.df_daily_train = None
        self.df_daily_val = None
        self.df_train = None
        self.df_val = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.feature_names = None
        self.transformed_feature_names = None
        
        # instantiate data classes
        self.raw_data = RawData(self.config)
        self.monthly_data = MonthlyData(self.config)

        # prepare monthly data
        self.prep_data()


    def prep_data(self):
        """
        Prepares the monthly data with features for the model.
        Args:
            None
        Returns:
            None
        Side Effects:
            - Sets the `df` attribute to the prepared monthly data if `clean_strategy` is 'olrem_for_all'.
            - Sets the `df_train` and `df_val` attributes to the prepared monthly data for 
              the training and validation respectively if `clean_strategy` is not 'olrem_for_all'.
            - Calls the `fix_data_types` and `set_feature_names` methods on the prepared data.
        """
        # prepare monthly data with features
        columns = ['monthly_period', 'shop_id', 'item_id', 'item_category_id', 'amount', 'price']
        if self.clean_strategy == 'olrem_for_all':
            self.df = self.prep_monthly_data_for_split(
                columns,
                'all'
            )
            self.df = self.fix_data_types(self.df)
            self.set_feature_names(self.df)
        else:
            self.df_train = self.prep_monthly_data_for_split(
                columns,
                'train'
                )
            self.df_train = self.fix_data_types(self.df_train)

            self.df_val = self.prep_monthly_data_for_split(
                columns,
                'val'
                )
            self.df_val = self.fix_data_types(self.df_val)
            self.set_feature_names(self.df_train)
            
    def set_feature_names(self, df):
        """
        Assigns feature names to the object based on the columns of the input DataFrame 'df' by excluding 'cols_to_drop'.
        Calculates the number of numeric features and assigns it to 'num_features'.
        Args:
            df (DataFrame): The input DataFrame containing the feature columns.
        Returns:
            None
        """
        # feature names
        self.feature_names = list(set(df.columns.values) - set(self.cols_to_drop))
        self.num_features = self.get_numeric_features()
    
    def get_numeric_features(self):
        """
        Calculates a list of numeric feature names.
        Returns:
            list: A list of numeric feature names.
        """
        num_feats = list(
            set(self.feature_names) - set(self.cat_features) - set(self.seasonal_features))
        return num_feats
        
    def fix_data_types(self, df):
        """
        Fix the data types of the input DataFrame by converting the 'item_category_id' column to int32.
        Args:
            df (pandas.DataFrame): The input DataFrame.
        Returns:
            pandas.DataFrame: The DataFrame with the 'item_category_id' column converted to int32.
        """
        df = df.astype({c: np.int32 for c in ['item_category_id']})
        return df
    
    def prep_monthly_data_for_split(
            self,
            columns,
            splitname):
        """
        Prepares monthly data for splitting based on the specified columns and split name.
        Args:
            columns: Columns to be used for splitting.
            splitname: Name of the split.        
        Returns:
            pandas.DataFrame: The prepared monthly data after processing.
        """

        fn_base = os.path.join(
            self.config['root_data_path'],
            self.config[f'fn_{splitname}_base'])
        fn_ts = os.path.join(
            self.config['root_data_path'],
            self.config[f'fn_{splitname}_ts'])
        
        if os.path.exists(fn_ts) and not self.refresh_ts_features:
            print(f'Loading {fn_ts}')
            df_ts = pd.read_parquet(fn_ts)
        else:
            if os.path.exists(fn_base) and not self.refresh_monthly:
                print(f'Loading {fn_base}')
                df_base = pd.read_parquet(fn_base)
            else:
                print(f'Creating {fn_base}')
                df_base = self.create_monthly_data(columns, splitname)
                df_base.to_parquet(fn_base)

            print(f'Creating {fn_ts}')
            df_ts = self.create_ts_features(df_base, self.num_lag_mon)
            df_ts.to_parquet(fn_ts)
        
        return df_ts

    def create_monthly_data(self, columns, splitname):
        """
        Creates monthly data for a given split.
        Args:
            columns (list): A list of column names to include in the monthly data.
            splitname (str): The name of the split to use. Must be one of 'train', 'val', or 'all'.
        Returns:
            pandas.DataFrame: The monthly data created for the specified split.
        Raises:
            Exception: If an unknown splitname is provided.
        """
        # read and process rawdata 
        self.prep_raw_data()

        if splitname == 'train':
            df_daily = self.df_daily_train[columns].copy()
        elif splitname == 'val':
            df_daily = self.df_daily_val[columns].copy()
        elif splitname == 'all':
            df_daily = self.df_daily[columns].copy()
        else:
            raise Exception(f'Unknown splitname: {splitname}')
        
        print (f'Creating monthly data for split {splitname}')
        df_base = self.monthly_data.prep_monthly_data(
            df_daily,
            self.raw_data.shop_list,
            self.raw_data.item_list[['item_id', 'item_category_id']].copy())

        return df_base
    
    def prep_raw_data(self):
        """
        Prepares the raw data by loading merged data, splitting train and test data, and cleaning the daily data.
        Args:
            None
        Returns:
            None
        """
        # load merged data:
        data_m = self.prep_merged_data()

        # split train and test data
        self.split_train_test_dailydata(data_m)
        
        # clean the data
        self.clean_daily_data()
        print('Prepared daily raw data.')

    def prep_merged_data(self):
        """
        Prepares the merged data by merging data and handling dates
        Args:
            None
        Returns:
            pandas.DataFrame: The merged data with 
        """
        data_merged = self.raw_data.merge_data()
        data_merged = self.raw_data.handle_dates(data_merged)
        # TODO: refactor into prep_raw_data function
        return data_merged
    
    def find_num_val_mon(self, df):
        """
        Finds the number of months to use for validation based on the given DataFrame.
        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
        Returns:
            int: The number of months to use for validation.
        """
        # find out the total number of monthly periods available
        if 'monthly_period' in df.columns:
            num_tot_mon = df.monthly_period.unique().shape[0]
        else:
            num_tot_mon = df.index.unique().shape[0]
        # total monthly periods - lag months gives effective total months
        num_efftot_mon = num_tot_mon - self.num_lag_mon
        # number of months to use for validation
        num_val_mon = round(num_efftot_mon * self.val_ratio)
        return num_val_mon
    
    def split_train_test_dailydata(self, df):
        """
        Splits the given daily data into training and testing sets based on the clean strategy and split strategy.
        Args:
            df (pandas.DataFrame): The daily data to be split.
        Returns:
            None
        Raises:
            ValueError: If the split strategy is unknown.
        """
        if self.clean_strategy == 'olrem_for_all':
            self.df_daily = df
        else:
            if self.split_strategy == 'random':
                self.split_train_test_dailydata_random(self, df)
            elif self.split_strategy == ['months', 'last_months_val']:
                self.split_train_test_dailydata_by_months(self, df)
            else:
                raise ValueError('Unknown split strategy')
        
    def split_train_test_dailydata_random(self, df):
        """
        Splits the given daily data into training and validation sets randomly based on the specified ratio.
        Args:
            df (pandas.DataFrame): The daily data to be split.
        Returns:
            None
        """
        self.df_daily_train, self.df_daily_val = train_test_split(
            df,
            test_size=self.val_ratio,
            random_state=42)
                
    def split_train_test_dailydata_by_months(self, df):
        """
        Splits the given daily data into training and testing sets based on months as the split strategy.
        Args:
            self: The object instance.
            df (pandas.DataFrame): The daily data to be split.
        Returns:
            None
        """
        num_val_mon = self.find_num_val_mon(df)
        if self.split_strategy == 'last_months_val':
            self.split_train_test_dailydata_last_months_val(
                df, 
                num_val_mon)
        elif self.split_strategy == 'months':
            self.split_train_test_dailydata_months(
                df, 
                num_val_mon)
    
    def split_train_test_dailydata_months(self, df, num_val_mon):
        """
        Splits the given daily data into training and testing sets based on months as the split strategy.
        Args:
            df (pandas.DataFrame): The daily data to be split.
            num_val_mon (int): The number of validation months.
        Returns:
            None
        """
        periods = df['monthly_period'].unique()
        val_periods = random.choices(periods, k=num_val_mon)
        # do the split
        self.df_val = df.loc[df['monthly_period'].isin(val_periods)].copy()
        self.df_train = df.loc[~df['monthly_period'].isin(val_periods)].copy()

    def split_train_test_dailydata_last_months_val(self, df, num_val_mon):
        """
        Splits the given daily data into training and testing sets based on the last months as the validation set.
        Args:
            df (pandas.DataFrame): The daily data to be split.
            num_val_mon (int): The number of validation months.
        Returns:
            None
        """
        # Determine the start date for validation
        y_last = df['date'].max().year
        m_last = df['date'].max().month
        given_date = datetime(y_last, m_last+1, 1)
        # train data extends until (excluding) this date:
        date_val_start = given_date - relativedelta(months=num_val_mon)
        # validation data starts self.num_lag_mon before, to account for lag
        rel_delta_lag_months = relativedelta(months=self.num_lag_mon)
        date_val_start_wlag = date_val_start - rel_delta_lag_months
        # Do the split
        self.df_daily_val = df.loc[df['date'] >= date_val_start_wlag, :].copy()
        self.df_daily_train = df.loc[df['date'] < date_val_start, :].copy()

    def clean_daily_data(self):
        """
        Cleans the daily data based on the specified cleaning strategy.
        This function cleans the daily data by removing negative values and outliers. 
        The cleaning strategy is determined by the `clean_strategy` attribute of the class instance.
        If the `clean_strategy` is set to `'olrem_for_all'`, 
            the entire `df_daily` dataframe is cleaned by removing negative values and outliers.
        If the `clean_strategy` is set to `'no_olrem_for_val'`, 
            the `df_daily_train` dataframe is cleaned by removing negative values and outliers, 
            while the `df_daily_val` dataframe is cleaned by removing negative values but not outliers.
        If an unknown `clean_strategy` is provided, a `ValueError` is raised.
        Args:
            None
        Returns:
            None
        """
        print('Cleaning data')
        # clean the training data from negative values and outliers
        if self.clean_strategy == 'olrem_for_all':
            print('Cleaning data')
            self.df_daily = self.raw_data.clean_data(
                self.df_daily,
                rem_negs=True,
                rem_ol=True)
        elif self.clean_strategy == 'no_olrem_for_val':
            print('Cleaning training data')
            # clean the training data from negative values and outliers
            self.df_daily_train = self.raw_data.clean_data(
                self.df_daily_train,
                rem_negs=True,
                rem_ol=True)
            print('Cleaning validation data')
            # clean the validation data from negative values (but not outliers)
            self.df_daily_val = self.raw_data.clean_data(
                self.df_daily_val,
                rem_negs=True,
                rem_ol=False)
        else:
            raise ValueError(f'Unknown clean strategy:{self.clean_strategy}')
        
    def create_ts_features(self, df_base, num_lag_mon):
        """
        Create time series features for the given DataFrame.
        Args:
            df_base (pandas.DataFrame): The base DataFrame.
            num_lag_mon (int): The number of lags to include.
        Returns:
            pandas.DataFrame: The DataFrame with the created time series features.
        """
        df_ts = self.monthly_data.add_lag_features(
            df_base,
            lags_to_include=num_lag_mon,
            lag_features=['price', 'amount_item', 'amount_cat'])
        df_ts = self.monthly_data.add_ma_features(
            df_ts,
            mas_to_include=[num_lag_mon-1],
            ma_features=['price_l1', 'amount_item_l1', 'amount_cat_l1']
        )
        # remove the months for which lags could not be calculated
        periods_to_remove = df_ts.index.unique()[0:num_lag_mon]
        df_ts = df_ts.drop(periods_to_remove)
        return df_ts

    def split_X_y(self):
        """
        Splits the input data into training X,y and validation X,y based on the clean strategy.
        This function splits the input data into training and validation sets based on the clean strategy. 
        If the clean strategy is 'olrem_for_all', it calls the `split_train_test_data` method to split the data.
        If the clean strategy is 'no_olrem_for_val', it does nothing as the train-test splitting has already been done. 
        If the clean strategy is neither 'olrem_for_all' nor 'no_olrem_for_val', it raises a ValueError
        Args:
            None
        Returns:
            None
        """
        if self.clean_strategy == 'olrem_for_all':
            self.split_train_test_data(self.df)
        elif self.clean_strategy == 'no_olrem_for_val':
            pass  # in this case train-test splitting had been already done
        else:
            raise ValueError(f'Unknown clean strategy:{self.clean_strategy}')
        self.X_train, self.y_train = self.get_X_y_for_split(self.df_train)
        self.X_val, self.y_val = self.get_X_y_for_split(self.df_val)
    
    def split_train_test_data(self, df):
        """
        Splits the given DataFrame into training and validation sets based on the specified strategy.
        Args:
            df (pandas.DataFrame): The DataFrame to be split.
        Returns:
            None
        Raises:
            ValueError: If the split strategy is unknown.
        """
        if self.split_strategy == 'random':
            self.split_train_test_data_random(df)
        elif self.split_strategy in ['months', 'last_months_val']:
            self.split_train_test_data_by_months(df)
        else:
            raise ValueError('Unknown split strategy')

    def split_train_test_data_random(self, df):
        """
        Splits the given DataFrame into training and validation sets randomly.
        Args:
            df (pandas.DataFrame): The DataFrame to be split.
        Returns:
            None
        """
        self.df_train, self.df_val = train_test_split(
            df,
            test_size=self.val_ratio,
            random_state=42)
        
    def split_train_test_data_by_months(self, df):
        """
        Splits the given DataFrame into training and validation sets based on months.
        Args:
            df (pandas.DataFrame): The DataFrame to be split based on months.
        Returns:
            None
        """
        num_val_mon = self.find_num_val_mon(df)
        if self.split_strategy == 'months':
            self.split_train_test_data_months(
                df,
                num_val_mon)
        elif self.split_strategy == 'last_months_val':
            self.split_train_test_data_last_months_val(
                df, 
                num_val_mon)
    
    def split_train_test_data_months(self, df, num_val_mon):
        """
        Splits the given DataFrame into training and validation sets based on months.
        Args:
            df (pandas.DataFrame): The DataFrame to be split based on months.
            num_val_mon (int): The number of months to use for the validation set.
        Returns:
            None
        """
        periods = df.index.unique()
        val_periods = random.choices(periods, k=num_val_mon)
        train_periods = list(set(periods)-set(val_periods))
        # do the split
        self.df_val = df.loc[val_periods].copy()
        self.df_train = df.loc[train_periods].copy()
    
    def split_train_test_data_last_months_val(self, df, num_val_mon):
        """
        Splits the given DataFrame into training and validation sets based on the last months.
        Args:
            df (pandas.DataFrame): The DataFrame to be split.
            num_val_mon (int): The number of months to use for the validation set.
        Returns:
            None

        """
        y_last = self.df.index.max().year
        m_last = self.df.index.max().month
        given_date = datetime(y_last, m_last+1, 1)
        # train data extends until (excluding) this date:
        date_val_start = given_date - relativedelta(months=num_val_mon)
        mperiod_val_start = pd.Series(date_val_start).dt.to_period('M').values[0]
        # do the split
        self.df_val = df.loc[df.index >= mperiod_val_start, :].copy()
        self.df_train = df.loc[df.index < mperiod_val_start, :].copy()
        
    def get_X_y_for_split(self, df):
        """
        Given a DataFrame `df`, this function extracts the 'amount_item' column as the target variable `y` 
        and drops the columns specified in `self.cols_to_drop`, assigning the remaining columns to `X`.
        Args:
            df (pandas.DataFrame): The input DataFrame.
        Returns:
            X (pandas.DataFrame): The DataFrame with the columns dropped.
            y (pandas.Series): The 'amount_item' column from the input DataFrame.
        """
        y = df['amount_item']
        df.drop(columns=self.cols_to_drop, axis=1, inplace=True)
        X = df
        return X, y


class TrigTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer calss that inherits both `BaseEstimator` and `TransformerMixin` and 
    applies a trigonometric (sin or cos) transformation to input data
    Attributes:
        period(int): the period of the trigonometric function
        trigfunction (str) = trigonometric function to be applied ('sin' or 'cos')
    Methods:
        fit(self, X, y): Fit the transformer to the given data.
        transform(self, X): Transforms the input data using the fitted transformer.
        get_feature_names_out(self, input_features): Returns the feature names of the output.
    """
    def __init__(self,
                 period,
                 trigfunction):
        """
        Initializes a new instance of the `TrigTransformer` class.
        Args:
            period (int): The period (in months)of the trigonometric function.
            trigfunction (str): The type of trigonometric function to apply. Must be either 'sin' or 'cos'.
        Returns:
            None
        """
        self.period = period
        self.trigfunction = trigfunction

        if trigfunction == 'sin':
            self.transformer = FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))
        elif trigfunction == 'cos':
            self.transformer = FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

    def fit(self, X, y=None):
        """
        Fit the transformer to the given data.
        Args:
            X (array-like): The input data.
            y (None, optional): The target variable. Defaults to None.
        Returns:
            self: The fitted transformer object.
        """
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        """
        Transforms the input data using the fitted transformer.
        Args:
            X (array-like): The input data to be transformed.
        Returns:
            array-like: The transformed data.
        """
        return self.transformer.transform(X)

    def get_feature_names_out(self, input_features):
        """
        Returns the feature names of the output.
        Args:
            input_features (list): A list of input feature names.
        Returns:
            numpy.ndarray: An array of output feature names.
        """
        return np.array(input_features)


# if __name__ == "__main__":
#     from utils import Utils
#     config = Utils.read_config_for_env(config_path='config/config.yml')
#     pred_data = PredictorData(
#         config,
#         refresh_monthly=False,
#         refresh_ts_features=False,
#         clean_strategy='olrem_for_all',
#         split_strategy='random',
#         num_lag_mon=3,
#         val_ratio=0.2)
#     # split the data and do the scaling
#     # stores X_train, y_train, X_val, y_val in predictor object
#     pred_data.split_X_y()
#     # encode and scale features 
#     pred_data.X_train = pred_data.preprocessor.fit_transform(
#         pred_data.X_train,
#         pred_data.y_train)