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
            refresh_monthly=False,
            refresh_ts_features=False,
            num_lag_mon=3,
            val_ratio=0.2,
            scaler_type = 'standard'
            ):
        # set input args
        self.config = config
        self.refresh_monthly = refresh_monthly
        self.refresh_ts_features = refresh_ts_features
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
        # prepare monthly data with features
        columns = ['monthly_period', 'shop_id', 'item_id', 'item_category_id', 'amount', 'price']
        self.df_train = self.prep_monthly_data_for_split(
            columns,
            'train'
            )
        self.df_val = self.prep_monthly_data_for_split(
            columns,
            'val'
            )
        # prepare X_train, y_train, X_val, y_val
        self.prep_X_y()
    
    def prep_monthly_data_for_split(
            self,
            columns,
            splitname):
        import os

        fn_base = os.path.join(
            self.config['root_data_path'],
            self.config[f'fn_{splitname}_base'])
        if os.path.exists(fn_base) and not self.refresh_monthly:
            print(f'Loading {fn_base}')
            df_base = pd.read_parquet(fn_base)
        else:
            print(f'Creating {fn_base}')
            df_base = self.create_monthly_data(columns, splitname)
            df_base.to_parquet(fn_base)

        fn_ts = os.path.join(
            self.config['root_data_path'],
            self.config[f'fn_{splitname}_ts'])
        if os.path.exists(fn_ts) and not self.refresh_ts_features:
            print(f'Loading {fn_ts}')
            df_ts = pd.read_parquet(fn_ts)
        else:
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
        self.split_train_test_data(data_m)
        
        # clean the data
        self.clean_daily_data()
        print('Prepared daily raw data.')

    def prep_merged_data(self):
        data_merged = self.raw_data.merge_data()
        data_merged = self.raw_data.handle_dates(data_merged)
        # data_cleaned = data.clean_data(data_merged)
        return data_merged
    
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

if __name__ == "__main__":
    from utils import Utils
    config = Utils.read_config_for_env(config_path='config/config.yml')
    