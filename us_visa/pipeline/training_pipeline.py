import os
import sys

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_validation import DataValidation
from us_visa.components.data_transformation import DataTransformation

from us_visa.entity.config_entity import (DataIngestionConfig,
                                           DataValidationConfig,
                                           DataTransfomationConfig)
from us_visa.entity.artifact_entity import (DataIngestionArtifact,
                                             DataValidationArtifact,
                                             DataTransformationArtifact)

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransfomationConfig()


    def start_data_ingestion(self) -> DataIngestionArtifact:
        '''
        This method of TrainingPipeline class is responsible for starting data ingestion component
        '''

        try:
            logging.info("Entered the start_data_ingestion method of TraingingPipeline class")
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Got the training & testing set from mongodb")
            logging.info("Exited the start_data ingestion method of TrainingPipeline class")

            return data_ingestion_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
    

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        This method of TrainingPipeline class is responsible for starting data ingestion components
        """

        logging.info("Entered the start_data_validation method of TrainingPipeline class")
        try:
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                              data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info(f"Performed the data validation operation")
            logging.info("Exited the start_data_validation method of TrainingPipeline class")

            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
        

    def start_data_transformation(self, data_ingestion_artifact: DataIngestion, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        This method of TrainingPipeline class is responsible for starting data ingestion components
        """
        logging.info("Entered the start_data_transformation method of TrainingPipeline class")
        try:
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=self.data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            logging.info("Exited the start_data_transformation method of TrainingPipeline")

            return data_transformation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
    
    def run_pipeline(self) -> None:
        '''
        This method of TrainingPipeline class is responsible for running complete pipeline
        '''

        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                                          data_validation_artifact=data_validation_artifact)
            
        except Exception as e:
            raise USvisaException(e, sys) from e