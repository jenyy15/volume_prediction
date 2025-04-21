import json
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf

# load data
def read_data(filename, filters=None, columns=None, folder_path="F:/predictors"):
    """read data from parquet file
    columns and filters are useful to save memory
    """
    return pq.read_table(
        f"{folder_path}/{filename}.parquet", filters=filters, columns=columns
    ).to_pandas()


def replace_list_elements(original_list: list, mapping: Dict):
    """helper function for load_data_columns_config"""
    return [mapping.get(item, item) for item in original_list]


def load_data_columns_config(version=1) -> Dict:
    """load data columns' config"""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if version == 1:
        config_path = os.path.join(
            script_dir, "config", "final_dataset_column_names.json"
        )
    else:
        config_path = os.path.join(
            script_dir, "config", "final_dataset_column_namesv2.json"
        )

    with open(config_path, "r") as file:
        config_dict = json.load(file)
    # Overwrite the column names which are not allowed in json files
    replacement_map = {
        "lag_elt-4": "lag_≤-4",
        "lag_egt5": "lag_≥5",
        "elt-3": "≤-3",
        "egt3": "≥3",
    }
    config_dict["release_schedule_factors"] = replace_list_elements(
        config_dict["release_schedule_factors"], replacement_map
    )
    return config_dict


def load_metrics(file_path) -> Dict:
    """load pickle file to dict"""
    with open(file_path, "rb") as file:
        loaded_dict = pickle.load(file)
    return loaded_dict


@dataclass
class ModelConfig:
    """model configuration"""

    model_name: str
    model_structure: Callable[[int, int], tf.keras.Model]
    verbose: int
    lr: Optional[float] = None
    encoder: Optional[tf.keras.Model] = None
    encoder_trainable: Optional[bool] = None


class DataFeeder(ABC):
    def __init__(
        self,
        data_df: pd.DataFrame,
        window_size: int,
        batch_size: int,
        predictors_size: int,
        predictors_dates: pd.Series,
    ):
        """Initialize the DataFeeder with the dataset and predictors dates"""
        # the 1st column is ISIN, the last column is target
        # it should be sorted by date and isin
        self.data_df = data_df
        self.window_size = window_size
        self.batch_size = batch_size
        self.predictors_size = predictors_size
        self.predictors_dates = predictors_dates.values
        self.column_idx = None  # To undertsand column_idx-th feature's importance

    @abstractmethod
    def gen_data(self, dataset_df: pd.DataFrame, dates_array: np.ndarray):
        """This method will be implemented by subclasses for specific data generation."""
        pass

    @abstractmethod
    def gen_tf_dataset(self, subset_filter: np.ndarray, column_idx: int):
        """Generate TensorFlow dataset for both RNN and non-RNN training, validation, and testing"""
        pass

    @classmethod
    def shuffle_column(self, X, y):
        """Shuffle a specific column of X
        A helper function for feature importance
        ref: https://www.kaggle.com/code/cdeotte/lstm-feature-importance
        """
        # Shuffle the selected column
        shuffled_column = tf.random.shuffle(X[:, :, self.column_idx])
        # Update the column with shuffled values
        X = tf.concat(
            [
                X[:, :, : self.column_idx],
                shuffled_column[:, :, None],
                X[:, :, self.column_idx + 1 :],
            ],
            axis=2,
        )
        return X, y


class NonRNNDataFeeder(DataFeeder):
    """Non-RNN model data feeder"""

    def __init__(
        self,
        data_df: pd.DataFrame,
        window_size: int,
        batch_size: int,
        predictors_size: int,
        predictors_dates: pd.Series,
    ):
        # the first N columns are features, the last column is target
        # it should be sorted by date and isin
        super().__init__(
            data_df, window_size, batch_size, predictors_size, predictors_dates
        )
        self.data_df = self.data_df.values
        self.col_names = list(self.data_df.columns)

    def gen_data(self, dataset_df: pd.DataFrame, dates_array: np.ndarray):
        """Non-RNN (e.g., Dense) specific data generation logic"""
        dataset_size = dates_array.shape[0]
        for i in range(1, dataset_size + 1):
            X = dataset_df[i - 1 : i, :-1]
            y = dataset_df[i - 1 : i, -1]  # Assuming target is last column
            yield X, y

    def gen_tf_dataset(self, subset_filter: np.ndarray, column_idx: int = None):
        """Generate TensorFlow dataset for Non-RNN training, validation, and testing"""
        dataset_df = self.data_df[subset_filter]
        dates_array = self.predictors_dates[subset_filter]

        dataset = tf.data.Dataset.from_generator(
            lambda: self.gen_data(dataset_df, dates_array),
            output_signature=(
                tf.TensorSpec(
                    shape=(self.window_size, self.predictors_size), dtype=tf.float32
                ),  # Shape for the input sequence
                tf.TensorSpec(shape=(1), dtype=tf.float32),  # Shape for the target
            ),
        )
        # Make it faster
        dataset = dataset.cache()
        if column_idx is None:
            # Apply batching and prefetching for performance
            return dataset.batch(self.batch_size).prefetch(buffer_size=100000)
        else:
            # Shuffle the column_idx-th column
            self.column_idx = column_idx
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.map(self.shuffle_column)
            return dataset.prefetch(buffer_size=100000)


class RNNDataFeeder(DataFeeder):
    """RNN model data feeder"""

    def __init__(
        self,
        data_df: pd.DataFrame,
        window_size: int,
        batch_size: int,
        predictors_size: int,
        predictors_dates: pd.Series,
    ):
        # the first 2 columns are (date, isin), the last column is target
        # it should be sorted by date and isin
        super().__init__(
            data_df, window_size, batch_size, predictors_size, predictors_dates
        )
        # The primary keys are (date, isin)
        # Set date as index
        self.data_df = self.data_df.set_index(["date"])
        self.col_names = list(self.data_df.columns)

    def gen_data(self, dataset_df: pd.DataFrame, dates_array: np.ndarray):
        """RNN-specific data generation logic"""
        for date_idx in range(self.window_size - 1, len(dates_array)):
            date_i = dates_array[date_idx]
            # search the records backwards within 2* window_size
            # include dates <= date_i and >= min(date_i - 2 * window_size, date_idx)
            start_idx = max(date_idx - 2 * self.window_size, 0)
            valid_dates = dates_array[start_idx : date_idx + 1]

            isin_set = set(dataset_df.loc[date_i]["isin"])
            grouped_subset = dataset_df.loc[valid_dates].groupby("isin")
            for isin, isin_data in grouped_subset:
                if isin not in isin_set or len(isin_data) < self.window_size:
                    continue
                # Skip ISIN
                X = isin_data.iloc[-self.window_size :, 1:-1].values
                y = isin_data.iloc[-1, -1]
                yield X, y

    def gen_tf_dataset(self, subset_filter: np.ndarray, column_idx: int = None):
        """Generate TensorFlow dataset for RNN training, validation, and testing"""
        dataset_df = self.data_df[subset_filter]
        dates_array = np.unique(self.predictors_dates[subset_filter])

        dataset = tf.data.Dataset.from_generator(
            lambda: self.gen_data(dataset_df, dates_array),
            output_signature=(
                tf.TensorSpec(
                    shape=(self.window_size, self.predictors_size), dtype=tf.float32
                ),  # Shape for the input sequence
                tf.TensorSpec(shape=(), dtype=tf.float32),  # Shape for the target
            ),
        )
        # Make it faster
        dataset = dataset.cache()

        if column_idx is None:
            # Apply batching and prefetching for performance
            return dataset.batch(self.batch_size).prefetch(buffer_size=100000)
        else:
            # Shuffle the column_idx-th column
            self.column_idx = column_idx
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.map(self.shuffle_column)
            return dataset.prefetch(buffer_size=100000)


class RnnAEDataFeeder(DataFeeder):
    """RNN autoencoder data feeder"""

    def __init__(
        self,
        data_df: pd.DataFrame,
        window_size: int,
        batch_size: int,
        predictors_size: int,
        predictors_dates: pd.Series,
    ):
        # the first 2 columns are (date, isin), the last column is target
        # it should be sorted by date and isin
        super().__init__(
            data_df, window_size, batch_size, predictors_size, predictors_dates
        )
        # the indexes are (date, isin)
        self.data_df = self.data_df.set_index(["date"])
        # train dates include train dataset and validation dataset
        self.train_dates = sorted(np.unique(self.predictors_dates))
        # the threshold index to split the dataset into train and val
        self.threshold = len(self.train_dates) - 60 - 1
        self.col_names = list(self.data_df.columns)

    def update(self, data_df: pd.DataFrame):
        """Update attributes of data feeder given new dataset"""
        assert self.col_names == list(
            self.data_df.columns
        ), "Failed: Column names don't match after update"
        # date should be the index of data_df
        self.data_df = data_df
        self.predictors_dates = self.data_df.index.values
        self.train_dates = sorted(np.unique(self.predictors_dates))
        # Assume the validation data has 60 dates
        self.threshold = len(self.train_dates) - 60 - 1

    def gen_data(self, dataset_df: pd.DataFrame, dates_array: np.ndarray):
        """RNN-specific data generation logic"""
        for date_idx in range(self.window_size - 1, len(dates_array)):
            date_i = dates_array[date_idx]
            # Search the records backwards within 2 * window_size
            # include dates <= date_i and >= min(date_i - 2 * window_size, date_idx)
            start_idx = max(date_idx - 2 * self.window_size, 0)
            valid_dates = dates_array[start_idx : date_idx + 1]

            isin_set = set(dataset_df.loc[date_i]["isin"])
            grouped_subset = dataset_df.loc[valid_dates].groupby("isin")
            for isin, isin_data in grouped_subset:
                if isin not in isin_set or len(isin_data) < self.window_size:
                    continue
                # Skip ISIN
                X = isin_data.iloc[-self.window_size :, 1:].values
                yield X, X

    def gen_tf_dataset(self, subset_filter: np.ndarray, column_idx: int = None):
        """Generate TensorFlow dataset for RNN training, validation, and testing"""
        dataset_df = self.data_df[subset_filter]
        dates_array = np.unique(self.predictors_dates[subset_filter])

        dataset = tf.data.Dataset.from_generator(
            lambda: self.gen_data(dataset_df, dates_array),
            output_signature=(
                tf.TensorSpec(
                    shape=(self.window_size, self.predictors_size), dtype=tf.float32
                ),  # Shape for the input sequence
                tf.TensorSpec(
                    shape=(self.window_size, self.predictors_size), dtype=tf.float32
                ),  # Shape for the target
            ),
        )
        # Make it faster
        dataset = dataset.cache()

        if column_idx is None:
            # Apply batching and prefetching for performance
            return dataset.batch(self.batch_size).prefetch(buffer_size=100000)
        else:
            # Shuffle the column_idx-th column
            self.column_idx = column_idx
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.map(self.shuffle_column)
            return dataset.prefetch(buffer_size=100000)
