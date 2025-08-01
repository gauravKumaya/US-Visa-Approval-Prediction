import os
from us_visa.constants import *
from us_visa.logger import logging
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass(frozen=True)
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ration: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = COLLECTION_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(TrainingPipelineConfig.artifact_dir, DATA_VALIDATION_DIR_NAME)
    
@dataclass
class DataTransfomationConfig: 
    data_transformation_dir: str = os.path.join(TrainingPipelineConfig.artifact_dir, DATA_TARNSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, 
                                                    DATA_TRANSFORMATION_TRANSFORMED_DIR, 
                                                    TRAIN_FILE_NAME.replace('csv', 'npy'))
    transformed_test_file_path: str = os.path.join(data_transformation_dir,
                                                   DATA_TRANSFORMATION_TRANSFORMED_DIR,
                                                   TEST_FILE_NAME.replace('csv', 'npy'))
    transformed_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     PREPROCESSING_OBJECT_FILE_NAME)