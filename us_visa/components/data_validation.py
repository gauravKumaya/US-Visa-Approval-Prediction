import os
import sys

import pandas as pd

from us_visa.entity.config_entity import DataValidationConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml, write_yaml
from us_visa.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        param data_ingestion_artifact: Output reference of data ingestion artifact stage
        param data_validation_config: configuration for data validation
        """

        try: 
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys) from e
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Method Name: validate_number_of_columns
        Description: This method validates the number of columns
        
        Output: Returns bool value based on validation status
        On Failure: Raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config.columns)
            logging.info(f"Is required columsn present: {status}")
            return status
        except Exception as e:
            raise USvisaException(e, sys) from e
    
    def has_required_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Method Name: has_required_columns
        Description: This method validates the existence of required columns (numerical & categorical)

        Output: Returns bool value based on validation status
        On Failure: Raise an exception
        """

        try:
            dataframe_columns = dataframe.columns
            missing_numerical_columns = []
            missing_categorical_columsn = []

            for col in self._schema_config.numerical_columns:
                if col not in dataframe_columns:
                    missing_numerical_columns.append(col)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing Numerical Columns: {missing_numerical_columns}")
            
            for col in self._schema_config.categorical_columns:
                if col not in dataframe_columns:
                    missing_categorical_columsn.append(col)
            
            if len(missing_categorical_columsn) > 0:
                logging.info(f"Missing Categorical Columns: {missing_categorical_columsn}")
            
            return False if len(missing_numerical_columns) > 0 or len(missing_categorical_columsn) > 0 else True
        except Exception as e:
            raise USvisaException(e, sys) from e
        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try: 
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys) from e
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name:    initiate_data_validation
        Description:    This method initiates the data validation components for the pipeline

        Output:         Returns bool value based on validation result
        On Failure:     Raise an exception
        """

        try:
            logging.info("Starting Data Validation")
            validation_error_message = ""
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.train_file_path),
                                DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            
            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"Training data column count validation result: {'PASSED' if status else 'FAILED'}")
            if not status:
                validation_error_message += "Training data column count validation FAILED."

            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"Testing data column count validation result: {'PASSED' if status else 'FAILED'}")
            if not status:
                validation_error_message += "Testing data column count validation FAILED."
            
            status = self.has_required_columns(dataframe=train_df)
            logging.info(f"Training data columns presence validation: {'PASSED' if status else 'FAILED'}")
            if not status:
                validation_error_message += "Required columns missing in training data"
            
            status = self.has_required_columns(dataframe=test_df)
            logging.info(f"Testing data columns presence validation: {'PASSED' if status else 'FAILED'}")
            if not status:
                validation_error_message += "Required columns missing in testing data"

            validation_status = len(validation_error_message) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_message
            )

            logging.info(f"Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
