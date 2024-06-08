from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# local imports
from data_tools import RawData, MonthlyData


class TF_NN_Predictor:
    """
    A class for creating and training TensorFlow neural network predictors.
    This class provides a convenient interface for creating, compiling, and training
    TensorFlow neural network models for time series prediction. It supports two
    predefined architectures ('3Dense' and '3Dense_v2'), but can also be customized
    to use other architectures.
    Attributes:
        input_dim (int): The number of input features.
        output_dim (int, optional): The number of output features. Default is 1.
        model_name (str, optional): The name of the pre-defined architecture to use. Default is '3Dense'.
        optimizer (str, optional): The optimizer to use for training. Default is 'adam'.
        metrics (list of str, optional):The metrics to use for evaluating the model. Default is ['mse'].
    Methods:
        __init__(self, input_dim, output_dim=1, model_name='3Dense', optimizer="adam", metrics=['mse']):
            Initializes the TF_NN_Predictor object with the specified parameters.
        get_callbacks(self):
            Returns a list of callbacks for use with the model's `fit` method.
        get_model(self):
            Returns the chosen TensorFlow model object.
        nn_model_3Dense(self):
            Returns a TensorFlow model object for the '3Dense' architecture.
        nn_model_3Dense_v2(self):
            Returns a TensorFlow model object for the '3Dense_v2' architecture.
        compile_summarize_model(self):
            Compiles the model and prints a summary of its architecture.
    """
    def __init__(
            self,
            input_dim, # = X.shape[1]
            output_dim=1,
            model_name='3Dense',
            optimizer="adam",
            metrics=['mse']):
        """
        Initializes the TF_NN_Predictor object with the specified parameters.
        Args:
            input_dim (int)
            output_dim (int, optional): Default is 1.
            model_name (str, optional): Default is '3Dense'.
            optimizer (str, optional): Default is 'adam'.
            metrics (list of str, optional): Default is ['mse'].
        Returns:
            None
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.optimizer = optimizer
        self.metrics = metrics
        
        self.get_model()
        self.compile_summarize_model()
        self.get_callbacks()

    def get_model(self):
        """
        Initializes the model based on the specified model name.
        Args:
            self (object): The instance of the class.
        Returns:
            None
        """
        if self.model_name == '3Dense':
            self.model = self.nn_model_3Dense()
        elif self.model_name == '4Dense':
            self.model = self.nn_model_4Dense()
        else:
            raise ValueError('Unknown model_name')

    def nn_model_3Dense(self):
        """
        Creates a neural network model with three dense layers and a dropout layer.
        Returns:
            NN_model (keras.models.Sequential): The created neural network model.
        """
        NN_model = Sequential()
        # Input layer
        NN_model.add(Input((self.input_dim,)))
        # Hidden Layers
        NN_model.add(Dense(32, kernel_initializer='normal',activation='relu'))
        NN_model.add(Dense(32, kernel_initializer='normal',activation='relu'))
        NN_model.add(Dense(32, kernel_initializer='normal',activation='relu'))
        # Dropout Layer
        NN_model.add(Dropout(0.2))
        # Output Layer
        NN_model.add(Dense(self.output_dim, kernel_initializer='normal',activation='linear'))
        return NN_model
    
    def nn_model_4Dense(self):
        """
        Creates a neural network model with four dense layers and a dropout layer.
        Returns:
            NN_model (keras.models.Sequential): The created neural network model.
        """
        NN_model = Sequential()
        # Input layer
        NN_model.add(Input((self.input_dim,)))
        # Hidden Layers
        NN_model.add(Dense(32, kernel_initializer='normal',activation='relu'))
        NN_model.add(Dense(64, kernel_initializer='normal',activation='relu'))
        NN_model.add(Dense(64, kernel_initializer='normal',activation='relu'))
        NN_model.add(Dense(32, kernel_initializer='normal',activation='relu'))
        # Dropout Layer
        NN_model.add(Dropout(0.2))
        # Output Layer
        NN_model.add(Dense(self.output_dim, kernel_initializer='normal',activation='linear'))
        return NN_model
    
    def compile_summarize_model(self):
        """
        Compiles the neural network model with the specified 
        loss function, optimizer, and metrics.
        and prints a summary of its architecture.
        Parameters:
            None
        Returns:
            None
        """
        #Compile the network
        self.model.compile(
            loss=tf.keras.losses.mse,
            optimizer=self.optimizer,
            metrics=self.metrics)
        self.model.summary()

    def get_callbacks(self):
        """
        Initializes and returns a list of callbacks
        to be used during the training of a model.
        Returns:
            list: A list of callbacks
        """
        #Define a checkpoint callback :
        checkpoint = ModelCheckpoint(
            '%s_weights-{epoch:03d}--{val_loss:.5f}.keras'%self.model_name,
            monitor='val_loss',
            verbose = 1,
            save_best_only = True,
            mode ='auto')
        earlystop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights= True)
        self.callbacks = [checkpoint, earlystop]


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
    """
    
    def __init__(
            self,
            config,
            refresh_monthly=False,
            refresh_ts_features=False,
            clean_strategy='olrem_for_all',
            split_strategy='random',
            num_lag_mon=3,
            val_ratio=0.2,
            scaler_type = 'standard'
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
            scaler_type (str, optional): 
                The type of scaler to use. Defaults to 'standard'.
        """
                
        # set input args
        self.config = config
        self.refresh_monthly = refresh_monthly
        self.refresh_ts_features = refresh_ts_features
        self.split_strategy = split_strategy
        self.clean_strategy = clean_strategy
        self.num_lag_mon = num_lag_mon
        self.val_ratio = val_ratio
        self.scaler_type = scaler_type

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
        
        # instantiate data classes
        self.raw_data = RawData(self.config)
        self.monthly_data = MonthlyData(self.config)

        # prepare monthly data
        self.prep_data()


    def prep_data(self):
        # prepare monthly data with features
        columns = ['monthly_period', 'shop_id', 'item_id', 'item_category_id', 'amount', 'price']
        if self.clean_strategy == 'olrem_for_all':
            self.df = self.prep_monthly_data_for_split(
                columns,
                'all'
            )
        else:
            self.df_train = self.prep_monthly_data_for_split(
                columns,
                'train'
                )
            self.df_val = self.prep_monthly_data_for_split(
                columns,
                'val'
                )
    
    def prep_monthly_data_for_split(
            self,
            columns,
            splitname):
        import os

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

        # load merged data:
        data_m = self.prep_merged_data()

        # split train and test data
        self.split_train_test_dailydata(data_m)
        
        # clean the data
        self.clean_daily_data()
        print('Prepared daily raw data.')

    def prep_merged_data(self):
        data_merged = self.raw_data.merge_data()
        data_merged = self.raw_data.handle_dates(data_merged)
        # data_cleaned = data.clean_data(data_merged)
        return data_merged
    
    def find_num_val_mon(self, df):
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
        self.df_daily_train, self.df_daily_val = train_test_split(
            df,
            test_size=self.val_ratio,
            random_state=42)
                
    def split_train_test_dailydata_by_months(self, df):
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
        periods = df['monthly_period'].unique()
        val_periods = random.choices(periods, k=num_val_mon)
        # do the split
        self.df_val = df.loc[df['monthly_period'].isin(val_periods)].copy()
        self.df_train = df.loc[~df['monthly_period'].isin(val_periods)].copy()

    def split_train_test_dailydata_last_months_val(self, df, num_val_mon):
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
        
    def split_scale_X_y(self):
        if self.clean_strategy == 'olrem_for_all':
            self.split_train_test_data(self.df)
        elif self.clean_strategy == 'no_olrem_for_val':
            pass  # in this case train-test splitting had been already done
        else:
            raise ValueError(f'Unknown clean strategy:{self.clean_strategy}')
        self.X_train, self.y_train = self.get_X_y_for_split(self.df_train)
        self.X_val, self.y_val = self.get_X_y_for_split(self.df_val)
        self.feature_names = list(self.X_train.columns.values)
        self.X_train, self.X_val = self.scale_X(self.X_train, self.X_val)
    
    def split_train_test_data(self, df):
        if self.split_strategy == 'random':
            self.split_train_test_data_random(df)
        elif self.split_strategy in ['months', 'last_months_val']:
            self.split_train_test_data_by_months(df)
        else:
            raise ValueError('Unknown split strategy')

    def split_train_test_data_random(self, df):
        self.df_train, self.df_val = train_test_split(
            df,
            test_size=self.val_ratio,
            random_state=42)
        
    def split_train_test_data_by_months(self, df):
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
        periods = df.index.unique()
        val_periods = random.choices(periods, k=num_val_mon)
        train_periods = list(set(periods)-set(val_periods))
        # do the split
        self.df_val = df.loc[val_periods].copy()
        self.df_train = df.loc[train_periods].copy()
    
    def split_train_test_data_last_months_val(self, df, num_val_mon):
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
        y = df['amount_item']
        df.drop(columns=['price', 'amount_item', 'amount_cat'], axis=1, inplace=True)
        X = df
        return X, y
    
    
    def scale_X(self, X_train, X_val):
        if self.scaler_type is None:
            return X_train, X_val
        elif self.scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            raise Exception(f'Unknown scaler: {scaler}')
        print(f'Scaling X train and X val with {scaler} scaler')
        # do the scaling
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        # X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
        # X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
        return X_train, X_val

# if __name__ == "__main__":
#     from utils import Utils
#     config = Utils.read_config_for_env(config_path='config/config.yml')
#     predictor = PredictorData(
#         config,
#         refresh_monthly=False,
#         refresh_ts_features=False,
#         num_lag_mon=3,
#         val_ratio=0.2,
#         scaler_type='standard')
#     # split the data and do the scaling
#     # stores X_train, y_train, X_val, y_val and feature_names in predictor object
#     predictor.split_scale_X_y()