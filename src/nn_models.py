from sklearn.preprocessing import TargetEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

from model_tools import PredictorData

class TF_NN_Predictor:
    """
    A class for creating and training TensorFlow neural network predictors.
    This class provides a convenient interface for creating, compiling, and training
    TensorFlow neural network models for time series prediction. It supports two
    predefined architectures ('3Dense' and '3Dense_v2'), but can also be customized
    to use other architectures.
    Attributes:
        pred_data (PredictorData object): An instance of the PredictorData class.
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
            pred_data,
            output_dim=1,
            model_name='3Dense',
            optimizer="adam",
            metrics=['mse']):
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

        # create preprocessing pipeline
        self.build_transformer_pipeline()
        
    def create_model(self):
        print ('Creating model')
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
                ('cat', TargetEncoder(target_type="continuous"), self.pred_data.cat_features),
                ("month_sin", PredictorData.sin_transformer(12), self.pred_data.seasonal_features),
                ("month_cos", PredictorData.cos_transformer(12), self.pred_data.seasonal_features),
                ('num', MinMaxScaler(), self.pred_data.num_features)
            ])
    
    def split_transform(self):
        """
        Splits the data into training and validation sets,
        and transforms the training and validation using the defined transformer.
        Parameters:
            None
        Returns:
            None
        """
        # split the data and do the scaling:
        # stores X_train, y_train, X_val, y_val self.pred_data object
        print('Splitting train-val')
        self.pred_data.split_X_y()
        # transform the train data
        print('Fit-transforming X_train')
        self.pred_data.X_train = self.transformer.fit_transform(
            self.pred_data.X_train,
            self.pred_data.y_train)
        # Transform the val data
        print('Transforming X_val')
        self.pred_data.X_val = self.transformer.transform(
            self.pred_data.X_val)