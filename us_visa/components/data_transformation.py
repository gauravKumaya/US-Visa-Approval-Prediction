import os
import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.entity.config_entity import DataTransfomationConfig
from us_visa.entity.artifact_entity import (DataIngestionArtifact,
                                            DataValidationArtifact,
                                            DataTransformationArtifact)
from us_visa.utils.main_utils import save_numpy_array_data, save_object, read_yaml, drop_columns
from us_visa.entity.estimator import TargetValueMapping

class DataTransformation:
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransfomationConfig):
        """
        param data_ingeston_artifact:       Ouput reference of data ingestion artifact
        param data_validation_artifact:     Output reference of data validation artifact
        param data_transformation_cofig:    Configuration for data transformation
        """

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys) from e
    

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys) from e
        
    
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name:    get_data_transformer_object
        Description:    This method creates and returns a data transformer object for the data

        Output:         data transfomer object
        On Failure:     Raise an exception
        """

        logging.info("Entered the get_data_transformer_object method of DataTransformation class")

        try:
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            or_encoder = OrdinalEncoder()
            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = self._schema_config.oh_columns
            or_columns = self._schema_config.or_columns
            transform_columns = self._schema_config.transform_columns
            num_features = self._schema_config.num_features
            logging.info("Got oh_columns, or_columns, transform_columns, num_features from schema config")

            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("OrdinalEncoder", or_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )
            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info("Exited the get_data_transformer_object method of DataTransformer class")

            return preprocessor
        except Exception as e:
            raise USvisaException(e, sys) from e
        
    

    def initiate_data_transformation(self) -> DataIngestionArtifact:
        """
        Method Name:    initiate_data_transformer
        Description:    This method initiates the data transformation component for the pipeline

        Output:         Returns DataIngestionArtifact data-type
        On Failure:     Raise an Exception
        """

        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")

                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object.")

                train_df = DataTransformation.read_data(self.data_ingestion_artifact.train_file_path)
                test_df = DataTransformation.read_data(self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(columns=TARGET_COLUMN)
                input_feature_test_df = test_df.drop(columns=TARGET_COLUMN)
                logging.info("Got input features of training and testing data.")

                target_feature_train_df = train_df[TARGET_COLUMN]
                target_feature_test_df = test_df[TARGET_COLUMN]
                logging.info("Got target feature of training and testing data.")

                input_feature_train_df['company_age'] = CURRENT_YEAR - input_feature_train_df['yr_of_estab']
                input_feature_test_df['company_age'] = CURRENT_YEAR - input_feature_test_df['yr_of_estab']
                logging.info("Added company_age column to the training dataset.")

                drop_cols = self._schema_config.drop_columns
                input_feature_train_df = drop_columns(input_feature_train_df, drop_cols)
                input_feature_test_df = drop_columns(input_feature_test_df, drop_cols)
                logging.info("Droped columns from training and testing data.")

                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping()._asdict()
                )
                target_feature_test_df = target_feature_test_df.replace(
                    TargetValueMapping()._asdict()
                )
                logging.info("Encoded the categorical target of training and testing data.")

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                logging.info("Fitted and transformed training input features using the preprocessing pipeline.")
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)
                logging.info("Transformed testing input features using the preprocessing pipeline.")

                smt = SMOTEENN(sampling_strategy='minority')
                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )
                logging.info("Applied SMOTEENN on trainind data")

                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]
                test_arr = np.c_[
                    input_feature_test_arr, np.array(target_feature_test_df)
                ]
                logging.info("Created train and test array")

                save_object(
                    file_path=self.data_transformation_config.transformed_object_file_path,
                    content=preprocessor
                )
                save_numpy_array_data(
                    file_path=self.data_transformation_config.transformed_train_file_path,
                    array=train_arr
                )
                save_numpy_array_data(
                    file_path=self.data_transformation_config.transformed_test_file_path,
                    array=test_arr
                )
                logging.info("Saved the preprocessor object")
                logging.info("Saved train and test array")


                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )

                logging.info("Exited initiate_data_transformation method of DataTransformation class")

                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)
            
        except Exception as e:
            raise USvisaException(e, sys) from e
        


