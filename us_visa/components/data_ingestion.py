import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.data_access.usvisa_data import USvisaData


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise USvisaException(e, sys) from e
        
    def export_data_into_feature_store(self) -> pd.DataFrame:
        try:
            logging.info("Exporting data from mongodb")

            usvisa_data = USvisaData()
            df = usvisa_data.export_collection_as_dataframe(collection_name=DataIngestionConfig.collection_name)
            logging.info(f"Shape of dataframe: {df.shape}")
            
            feature_store_file_path = DataIngestionConfig.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            df.to_csv(feature_store_file_path, index=False, header=False)

            return df
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Method name: split_data_as_train_test
        Description: This methods splits the data into train and test set based on the ratio

        Output: Folder is created
        On Failure: Write an exception log and then raise an exception
        """
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ration, random_state=42)
            logging.info("Performed train-test-split on the dataframe")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test files")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False)

            logging.info("Exported train and test file path")

        except Exception as e:
            raise USvisaException(e, sys) from e
        
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        
        logging.info("Entered initiate_daata_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the dataframe from mongo db")

            self.split_data_as_train_test(dataframe=dataframe)

            logging.info("Performed train-test-split of data")
            logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path,
                                                            test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
