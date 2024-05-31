import os
import pandas as pd


class RawData:
    def __init__(self, config):
        """
        Initializes a new instance of the RawData class.
        Args:
            config (dict, optional): A dictionary of configuration parameters.
        Initializes the following instance variables:
            - config (dict): The configuration parameters.
            - category_list (pandas.DataFrame): The loaded category list data.
            - item_list (pandas.DataFrame): The loaded item list data.
            - shop_list (pandas.DataFrame): The loaded shop list data.
            - transaction (pandas.DataFrame): The loaded transaction data.
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


# # for testing:
# if __name__ == "__main__":
#     from utils import Utils
#     raw_data = RawData(Utils.read_config_for_env())
