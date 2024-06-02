from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

# local imports
from data_utils import Data


class TreePredictor:
    def __init__(
            self,
            config,
            num_lag_mon=3,
            val_ratio=0.2,
            ):
        self.config = config
        self.num_lag_mon = num_lag_mon
        self.val_ratio = val_ratio
        self.data_daily_train = None
        self.data_daily_val = None

        # prepare data
        self.prep_data()

    
    def prep_data(self):
        data_inst = Data(self.config)

        # load merged data:
        data_m = self.prep_merged_data(data_inst)

        # split train and test data
        self.split_train_test_data(data_m)
        
        # clean the data
        self.clean_daily_data(data_inst)
        print('Prepared daily data.')

        # self.prep_monthly_data()
        # print('Prepared monthly data.')

    def prep_merged_data(self, data_inst):
        data_merged = data_inst.merge_data()
        data_merged = data_inst.handle_dates(data_merged)
        # data_cleaned = data.clean_data(data_merged)
        return data_merged

    def split_train_test_data(self, df):

        # find out the total number of monthly periods available
        num_tot_mon = df.monthly_period.unique().shape[0]
        # total monthly periods - lag months gives effective total months
        num_efftot_mon = num_tot_mon - self.num_lag_mon
        # number of months to use for validation
        num_val_mon = round(num_efftot_mon * self.val_ratio)


        y_last = df['date'].max().year
        m_last = df['date'].max().month
        
        date_val_start = self.months_prior(
            given_date=datetime(y_last, m_last+1, 1),
            months=num_val_mon
            )

        self.data_daily_val = df.loc[df['date'] >= date_val_start, :].copy()
        self.data_daily_train = df.loc[df['date'] < date_val_start, :].copy()

    @staticmethod
    def months_prior(given_date, months):
        return given_date - relativedelta(months=months)
    
    def clean_daily_data(self, data_inst):
        
        print('Cleaning training data')
        # clean the training data from negative values and outliers
        self.data_daily_train = data_inst.clean_data(
            self.data_daily_train,
            rem_negs=True,
            rem_ol=True)

        print('Cleaning validation data')
        # clean the validation data from negative values (but not outliers)
        self.data_daily_val = data_inst.clean_data(
            self.data_daily_val,
            rem_negs=True,
            rem_ol=False)
    
    def prep_monthly_data(self, df_daily):
        df_items_monthly_grouped = df_daily.groupby(
            ['shop_id', 'item_id', 'monthly_period', 'category_id', 'price', 'amount']
            )
        df_items_monthly = df_items_monthly_grouped.agg(
            {'price': 'sum',
             'amount': 'sum',
             'dayofweek': 'first',
             'year': 'first'}
        ).reset_index()

        # 
        return df_items_monthly

if __name__ == "__main__":
    from utils import Utils
    config = Utils.read_config_for_env(config_path='config/config.yml')
    