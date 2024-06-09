import lightgbm as lgb
import numpy as np
from sklearn.compose import ColumnTransformer

# local imports
from model_tools import PredictorData, BasePredictor, TrigTransformer


class LGBM_Predictor(BasePredictor):
    """
    A class for creating and training LGBM Predictor
    Attributes:
        pred_data (PredictorData object): An instance of the PredictorData class.
    Methods:
        __init__(self, pred_data):
            Initializes the LGBM_Predictor object with the specified parameters.
    """
    def __init__(
            self,
            pred_data):
        BasePredictor.__init__(self, pred_data)

        """
        Initializes the LGBM_Predictor object with the specified parameters.
        Args:
            pred_data (PredictorData object)
        Returns:
            None
        """
        # input arguments
        self.pred_data = pred_data

        # create preprocessing pipeline
        self.build_transformer_pipeline()


    def build_transformer_pipeline(self):
        """
        This method creates a ColumnTransformer object that applies:
        - for categorical features: TargetEncoder with continuous target type.
        - for seasonal features: Applies a sine and cosine transformations.
        - num_features: Applies a MinMaxScaler.
        Parameters:
            self (object): The instance of the class.
        Returns:
            None
        """
        self.transformer = ColumnTransformer(
            transformers=[
                ("sin", TrigTransformer(12, 'sin'), self.pred_data.seasonal_features),
                ("cos", TrigTransformer(12, 'cos'), self.pred_data.seasonal_features)
                ],
            remainder='passthrough'
            )
    
        

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
#     lgbm_predictor = LGBM_Predictor(
#         pred_data=pred_data)
#     lgbm_predictor.pred_data = lgbm_predictor.split_transform(
#         lgbm_predictor.pred_data,
#         lgbm_predictor.transformer)
#     print(lgbm_predictor.pred_data.X_train[:2,:])
#     print(lgbm_predictor.pred_data.transformed_feature_names)
