from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

from model_tools import BasePredictor, TrigTransformer


class TF_NN_Predictor(BasePredictor):
    """
    A class for creating and training TensorFlow neural network predictors.
    This class provides a convenient interface for creating, compiling, and training
    TensorFlow neural network models for time series prediction. It supports two
    predefined architectures ('3Dense' and '3Dense_v2'), but can also be customized
    to use other architectures.
    Attributes:
        pred_data (PredictorData object): An instance of the PredictorData class.
        output_dim (int, optional): The number of output features. Default is 1.
        model_name (str, optional): The name of the pre-defined architecture to use. Default is '3Dense'.
        optimizer (str, optional): The optimizer to use for training. Default is 'adam'.
        metrics (list of str, optional):The metrics to use for evaluating the model. Default is ['mse'].
    Methods:
        __init__(self, pred_data, output_dim=1, model_name='3Dense', optimizer="adam", metrics=['mse']):
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
            pred_data,
            output_dim=1,
            model_name='3Dense',
            optimizer="adam",
            metrics=['mse']):
        BasePredictor.__init__(self, pred_data)
        """
        Initializes the TF_NN_Predictor object with the specified parameters.
        Args:
            pred_data (PredictorData object)
            output_dim (int, optional): Default is 1.
            model_name (str, optional): Default is '3Dense'.
            optimizer (str, optional): Default is 'adam'.
            metrics (list of str, optional): Default is ['mse'].
        Returns:
            None
        """
        # input arguments
        self.pred_data = pred_data
        self.output_dim = output_dim
        self.model_name = model_name
        self.optimizer = optimizer
        self.metrics = metrics

        # declarations
        self.input_dim = None
        self.model = None
        self.callbacks = None

        # create preprocessing pipeline
        self.build_transformer_pipeline()

    def create_model(self):
        print('Creating model')
        if self.pred_data.X_train is None:
            raise ValueError('pred_data.X_train has not been set yet. Call split_transform()')
        else:
            self.input_dim = self.pred_data.X_train.shape[1]

        # create the model
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
        NN_model.add(Dense(32, kernel_initializer='normal', activation='relu'))
        NN_model.add(Dense(32, kernel_initializer='normal', activation='relu'))
        NN_model.add(Dense(32, kernel_initializer='normal', activation='relu'))
        # Dropout Layer
        NN_model.add(Dropout(0.2))
        # Output Layer
        NN_model.add(Dense(self.output_dim, kernel_initializer='normal', activation='linear'))
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
        NN_model.add(Dense(24, kernel_initializer='normal', activation='relu'))
        NN_model.add(Dense(48, kernel_initializer='normal', activation='relu'))
        NN_model.add(Dense(48, kernel_initializer='normal', activation='relu'))
        NN_model.add(Dense(24, kernel_initializer='normal', activation='relu'))
        # Dropout Layer
        NN_model.add(Dropout(0.2))
        # Output Layer
        NN_model.add(Dense(self.output_dim, kernel_initializer='normal', activation='linear'))
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
        # compile the network
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
        # define a checkpoint callback:
        checkpoint = ModelCheckpoint(
            '%s_weights-{epoch:03d}--{val_loss:.5f}.keras'%self.model_name,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='auto')
        # define an early stopping callback:
        earlystop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True)
        self.callbacks = [checkpoint, earlystop]

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
                ('TargetEncoded', TargetEncoder(target_type="continuous"), self.pred_data.cat_features),
                ("sin", TrigTransformer(12, 'sin'), self.pred_data.seasonal_features),
                ("cos", TrigTransformer(12, 'cos'), self.pred_data.seasonal_features),
                ('MinMaxScaled', MinMaxScaler(), self.pred_data.num_features)
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
#     tf_nn_predictor_4D = TF_NN_Predictor(
#         pred_data=pred_data,
#         output_dim=1,
#         model_name='4Dense',
#         optimizer="adam",
#         metrics=['mse'])
#     tf_nn_predictor_4D.pred_data = tf_nn_predictor_4D.split_transform(
#         tf_nn_predictor_4D.pred_data,
#         tf_nn_predictor_4D.transformer)
#     print(tf_nn_predictor_4D.pred_data.X_train[:2,:])
#     print(tf_nn_predictor_4D.pred_data.transformed_feature_names)
#     tf_nn_predictor_4D.create_model()
#     tf_nn_predictor_4D.model.fit(
#         tf_nn_predictor_4D.pred_data.X_train,
#         tf_nn_predictor_4D.pred_data.y_train,
#         epochs=10,
#         batch_size=4096,
#         validation_data=(tf_nn_predictor_4D.pred_data.X_val, tf_nn_predictor_4D.pred_data.y_val),
#         callbacks=tf_nn_predictor_4D.callbacks)
