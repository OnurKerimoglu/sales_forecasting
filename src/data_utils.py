import os
import pandas as pd


class RawData:
    def __init__(self, config=None):
        self.config = config

        # load data
        self.load_data()

        # fix data schemas
        if self.config['env'] == 'local':
            self.fix_data_schemas()

    def load_data(self):
        if self.config['env'] == 'local':
            self.load_all_data_from_csv()

    def load_all_data_from_csv(self):
        print('Loading data..', end=' ')
        self.category_list = self.load_csv(self.config['fname_categories'])
        self.item_list = self.load_csv(self.config['fname_items'])
        self.shop_list = self.load_csv(self.config['fname_shops'])
        self.transaction = self.load_csv(self.config['fname_transactions'])
        print('Done.')

    def load_csv(self, fname: str):
        fpath = os.path.join(self.config['root_data_path'], fname)
        df = pd.read_csv(fpath)
        # remove the unnecessary columns
        cols_to_remove = df.columns.str.contains('Unnamed: 0', case=False)
        df.drop(df.columns[cols_to_remove], axis=1, inplace=True)
        return df

    def fix_data_schemas(self):
        # Here known schema issues are fixed:
        print('Fixing data schemas..', end=' ')
        self.transaction.rename(
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
