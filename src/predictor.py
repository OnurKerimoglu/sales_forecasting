from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler

# local imports
from data_utils import RawData, MonthlyData


class BasePredictor:
    def __init__(
            self,
            config,
            refresh=False,
            num_lag_mon=3,
            val_ratio=0.2,
            scaler_type = 'standard'
            ):
        # set input args
        self.config = config
        self.refresh = refresh
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
        
        # instantiate data classes
        self.raw_data = RawData(self.config)
        self.monthly_data = MonthlyData(self.config)

        # prepare monthly data
        self.prep_data()


    def prep_data(self):

        # read and process rawdata (TODO: do this only if needed)
        self.prep_raw_data()

        # prepare monthly data with features
        columns = ['monthly_period', 'shop_id', 'item_id', 'item_category_id', 'amount', 'price']
        self.df_train = self.prep_monthly_data_for_split(
            self.df_daily_train[columns].copy(),
            'train',
            refresh=self.refresh
            )
        self.df_val = self.prep_monthly_data_for_split(
            self.df_daily_val[columns].copy(),
            'val',
            refresh=self.refresh
            )
        
        # prepare X_train, y_train, X_val, y_val
        self.prep_X_y()

    def prep_raw_data(self):

        # load merged data:
        data_m = self.prep_merged_data()

        # split train and test data
        self.split_train_test_data(data_m)
        
        # clean the data
        self.clean_daily_data()
        print('Prepared daily raw data.')

    def prep_merged_data(self):
        data_merged = self.raw_data.merge_data()
        data_merged = self.raw_data.handle_dates(data_merged)
        # data_cleaned = data.clean_data(data_merged)
        return data_merged
    
    def prep_monthly_data_for_split(
            self,
            df_daily,
            splitname,
            refresh):
    
        # get base monthly data
        df_base = self.monthly_data.get_monthly_data(df_daily, splitname, refresh)

        # get monthly data with lag and ma features
        df_ts= self.monthly_data.get_ts_features(df_base, splitname, refresh, self.num_lag_mon)
        
        return df_ts

    def prep_X_y(self):
        self.X_train, self.y_train = self.get_X_y_for_split(self.df_train)
        self.X_val, self.y_val = self.get_X_y_for_split(self.df_val)
        self.X_train, self.X_val = self.scale_X(self.X_train, self.X_val)
    
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
        return X_train, X_val

    def split_train_test_data(self, df):

        # find out the total number of monthly periods available
        num_tot_mon = df.monthly_period.unique().shape[0]
        # total monthly periods - lag months gives effective total months
        num_efftot_mon = num_tot_mon - self.num_lag_mon
        # number of months to use for validation
        num_val_mon = round(num_efftot_mon * self.val_ratio)

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

if __name__ == "__main__":
    from utils import Utils
    config = Utils.read_config_for_env(config_path='config/config.yml')
    