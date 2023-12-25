import time
from pathlib import Path
from typing import Any, List, Tuple, Union
import numpy as np
import pandas as pd
import os
import config

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"Running {func.__name__} ...", end='\r')
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} Done in {end - start:.2f} seconds")
        return result
    return wrapper

def read_file(filename: Union[str, Path]) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """read_file

    Args:
        filename (Union[str, Path]): The filename to read

    Returns:
       pd.DataFrame: The data in the file
    """
    temp_data=pd.read_csv(filename)
    bid_temp=pd.read_csv(config.IN_DIR/"bids.csv.zip")

    return temp_data,bid_temp

def write_file(output:pd.DataFrame,filename:Union[str,Path]) -> None:
    """write_file writes the data to a txt file 

    Args:
        output (pd.DataFrame): The data to write to the file
        filename (Union[str, Path]): The filename to write to
    """
    output.to_csv(filename)