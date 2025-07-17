import os
import sys

import numpy as np
import dill
import yaml
from pandas import DataFrame

from us_visa.exception import USvisaException
from us_visa.logger import logging



def read_yaml(file_path: str) -> dict:
    logging.info("Entered the read_yaml method of utils")

    try:
        with open(file_path, 'rb') as yaml_file:

            logging.info("Exited the read_yaml method of utils")

            return yaml.safe_load(yaml_file)
        
    except Exception as e:
        raise USvisaException(e, sys) from e
    


def write_yaml(file_path: str, content: object) -> None:
    logging.info("Entered the write_yaml method of utils")

    try:
        with open(file_path, 'w') as file:
            yaml.dump(content, file)

            logging.info("Exited the write_yaml method of utils")
    except Exception as e:
        raise USvisaException(e, sys) from e



def save_object(file_path: str, content: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(content, file_obj)
        
        logging("Exited the save_object method of utils")

    except Exception as e:
        raise USvisaException(e, sys) from e



def load_object(file_path: str) ->object:
    logging.info("Entered the load_object method of utils")
    
    try:
        with open(file_path, 'rb') as file:
            obj = dill.load(file)
        
        logging.info("Exited the load_object method of utils")

        return obj
    except Exception as e:
        raise USvisaException(e, sys) from e

    

def save_numpy_array_data(file_path: str, array: np.array) -> None:
    '''
    Save numpy array data to a file
    file_path: str location of file to save
    array: np.array data to save
    '''
    logging.info("Enter the save_numpy_array_data method of utils")

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)

        logging.info("Exited the save_numpy_array_data method of utils")

    except Exception as e:
        raise USvisaException(e, sys) from e
    


def load_numpy_array_data(file_path: str) -> np.array:
    '''
    Load numpy array data from a file
    file_path: str location of file to load
    return np.array data loaded
    '''

    logging.info("Entered the load_numpy_array_data method of utils")
    try:
        with open(file_path, 'rb') as file_obj:

            logging.info("Exited the load_numpy_array_data method of utils")
            return np.load(file_obj)
    
    except Exception as e:
        raise USvisaException(e, sys) from e
    


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    '''
    drop the columns from a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    '''