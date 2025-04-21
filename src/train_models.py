import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from pmdarima.model_selection import RollingForecastCV, SlidingWindowForecastCV
from tqdm import tqdm
import pickle

from .model_training_utils import (
    ModelConfig,
    DataFeeder,
    NonRNNDataFeeder,
    RNNDataFeeder,
    RnnAEDataFeeder,
)


def setup_gpu():
    """Setup gpu"""
    physical_gpus = tf.config.list_physical_devices("GPU")
    if physical_gpus:
        try:
            # Check if GPUs have already been set as visible devices
            visible_devices = tf.config.get_visible_devices()
            if not any(device.device_type == "GPU" for device in visible_devices):
                tf.config.set_visible_devices(physical_gpus[0], "GPU")
                # Restrict GPU memory to 8GB for the first GPU
                tf.config.set_logical_device_configuration(
                    physical_gpus[0],
                    # Restrict TensorFlow to only allocate 8GB of memory on the first GPU
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 8)],
                )
                logical_gpus = tf.config.list_logical_devices("GPU")
                print(
                    len(physical_gpus),
                    "Physical GPUs,",
                    len(logical_gpus),
                    "Logical GPUs",
                )
            else:
                print("GPU devices are already configured, skipping setup.")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


### For models training
@tf.function
def r2(y_true: np.ndarray[np.float32], y_pred: np.ndarray[np.float32]) -> float:
    """Custom R2 metric for Keras/TensorFlow"""
    # https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
    ss_total_train = K.sum((y_true - K.mean(y_true)) ** 2)
    ss_residual_train = K.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual_train / ss_total_train)
    return r2


def train_NN(
    data_feeder: DataFeeder,
    train_filter: pd.Series,
    val_filter: pd.Series,
    verbose: int,
    model_config: ModelConfig,
    model_type: Optional[str] = None,
):
    """train RNN on predictors and make prediction for volume"""
    # Prepare datasets for training and validation
    train_ds = data_feeder.gen_tf_dataset(train_filter)
    valid_ds = data_feeder.gen_tf_dataset(val_filter)

    # Build model
    if model_type == "transfer":
        # Transfer learning:
        # Train the rest part of model with pretrained encoder
        other_input_size = data_feeder.predictors_size-model_config.encoder.input_shape[-1]
        model = model_config.model_structure(
            model_config.encoder, data_feeder.window_size, other_input_size
        )
    else:
        # Normal mode:
        model = model_config.model_structure(
            data_feeder.window_size, data_feeder.predictors_size
        )
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            model_config.lr if model_config.lr else 0.001
        ),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=None if model_type == "AE" else [r2],  # autoencoder doesn't need r2
    )
    # Reduce learning rate when loss doesn't decrease
    reducelr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=5, verbose=0, min_delta=1e-2
    )
    # Stop early if the loss doesn't decrease
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=10, restore_best_weights=True
    )
    rst = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=500,
        batch_size=data_feeder.batch_size,
        verbose=verbose,
        callbacks=[reducelr_cb, earlystop_cb],
    )
    return model, rst.history


def fit_models_with_cross_validation(
    data_feeder: RNNDataFeeder | NonRNNDataFeeder,
    cv_spliter: RollingForecastCV | SlidingWindowForecastCV,
    train_dates: List,
    test_filter: np.ndarray,
    model_config: ModelConfig,
    model_name: str,
):
    """Fit models with cross-validation"""
    train_metrics_dict = {}
    test_metrics = {}
    cv_fold_count = 1

    for train_idx, test_idx in tqdm(cv_spliter.split(train_dates)):
        # Generate train and validation dataset filter
        train_filter = (data_feeder.predictors_dates >= train_dates[train_idx[0]]) & (
            data_feeder.predictors_dates <= train_dates[train_idx[-1]]
        )
        val_filter = (data_feeder.predictors_dates >= train_dates[test_idx[0]]) & (
            data_feeder.predictors_dates <= train_dates[test_idx[-1]]
        )

        model, train_metrics_dict[cv_fold_count] = train_NN(
            data_feeder=data_feeder,
            train_filter=train_filter,
            val_filter=val_filter,
            verbose=model_config.verbose,
            model_config=model_config,
        )
        # https://www.tensorflow.org/tutorials/keras/save_and_load
        model.save_weights(f"./checkpoints/{model_config.model_name}_CV{cv_fold_count}")
        # Test dataset evaluation
        test_ds = data_feeder.gen_tf_dataset(test_filter)
        test_metrics[cv_fold_count] = model.evaluate(test_ds, verbose=1)
        # Clear the session to release memory
        K.clear_session()
        del model, train_filter, val_filter, test_ds
        cv_fold_count += 1
        with open(
            f"./metrics/train_metrics_dict_{model_name}.pkl", "wb"
        ) as pickle_file:
            pickle.dump(train_metrics_dict, pickle_file)
        with open(f"./metrics/test_metrics_{model_name}.pkl", "wb") as pickle_file:
            pickle.dump(test_metrics, pickle_file)
    return train_metrics_dict, test_metrics

### pretrain with autoencoder
def train_AE(
    data_feeder: DataFeeder,
    train_filter: pd.Series,
    val_filter: pd.Series,
    verbose: int,
    model_config: ModelConfig,
):
    """Train autoencoder"""
    ae_model, train_metrics = train_NN(
        data_feeder=data_feeder,
        train_filter=train_filter,
        val_filter=val_filter,
        verbose=model_config.verbose,
        model_config=model_config,
        model_type="AE",
    )
    # Only keep encoder
    encoder = tf.keras.Model(inputs=ae_model.input, outputs=ae_model.layers[2].output)
    if not model_config.encoder_trainable:
        for layer in encoder.layers:
            layer.trainable = False
    return encoder, train_metrics


def update_pretrain_data_feeder(
    ae_data_feeder, data_feeder, first_train_dates, horizon
):
    data_df = data_feeder.data_df.loc[
        (data_feeder.predictors_dates < first_train_dates[1])
        & (data_feeder.predictors_dates >= first_train_dates[0]),
        ae_data_feeder.col_names,
    ].copy()
    data_df = pd.concat(
        [
            ae_data_feeder.data_df[
                ae_data_feeder.predictors_dates
                > ae_data_feeder.train_dates[horizon - 1]
            ],
            data_df,
        ]
    )
    ae_data_feeder.update(data_df)
    return ae_data_feeder


def update_pretrain_filter(ae_data_feeder):
    """Update pretrain dataset filter"""
    pretrain_filter = (
        ae_data_feeder.predictors_dates >= ae_data_feeder.train_dates[0]
    ) & (
        ae_data_feeder.predictors_dates
        <= ae_data_feeder.train_dates[ae_data_feeder.threshold]
    )
    preval_filter = (
        ae_data_feeder.predictors_dates
        >= ae_data_feeder.train_dates[ae_data_feeder.threshold + 1]
    ) & (ae_data_feeder.predictors_dates <= ae_data_feeder.train_dates[-1])
    return pretrain_filter, preval_filter


def validate_columns_idx(ae_data_feeder_cols, data_feeder_cols):
    """Ensure the column positions of encoder are corret"""
    # Skip isin, the 1 st column
    autoencoder_cols_len = len(ae_data_feeder_cols[1:])
    last_columns = data_feeder_cols[-(autoencoder_cols_len+1):-1]
    assert ae_data_feeder_cols[1:] == last_columns, (
        f"Failed: autoencoder data_feeder's column names don't match the last {autoencoder_cols_len} "
        f"columns in data feeder. Expected: {ae_data_feeder_cols[1:]}, but got: {last_columns}"
    )


def fit_models_with_cross_validation_v2(
    data_feeder: DataFeeder,
    ae_data_feeder: DataFeeder,
    cv_spliter: RollingForecastCV | SlidingWindowForecastCV,
    train_dates: List,
    test_filter: np.ndarray,
    model_config: ModelConfig,
    ae_model_config: ModelConfig,
    model_name: str,
    skip_cv_list: Optional[List]=None,
):
    """Fit version 2 models with cross-validation
    version 2 model contains pretrain process: use history data to train encoder part
    """
    validate_columns_idx(ae_data_feeder.col_names, data_feeder.col_names)
    train_metrics_dict = {}
    test_metrics = {}
    ae_metrics_dict = {}
    cv_fold_count = 1
    # [last first_train_date, current first_train_date]
    first_train_dates = [None, None]
    # Pretrain dataset
    # the 1st dataset is the pre_train_dataset
    # then the pretrain dataset moves to the later dates like a sliding window
    for train_idx, test_idx in tqdm(cv_spliter.split(train_dates)):
        # Update current first_train_date
        first_train_dates[1] = train_dates[train_idx[0]]
        if cv_fold_count > 1:
            # Update autoencoder data feeder
            ae_data_feeder = update_pretrain_data_feeder(
                ae_data_feeder, data_feeder, first_train_dates, cv_spliter.horizon
            )
        print(ae_data_feeder.train_dates[0], ae_data_feeder.train_dates[-1], f"len={len(ae_data_feeder.train_dates)}")
        # Generate pretrain dataset filter
        pretrain_filter, preval_filter = update_pretrain_filter(ae_data_feeder)
        if (skip_cv_list is None) or (cv_fold_count not in skip_cv_list):
            # Pretrain model
            model_config.encoder, ae_metrics_dict[cv_fold_count] = train_AE(
                data_feeder=ae_data_feeder,
                train_filter=pretrain_filter,
                val_filter=preval_filter,
                verbose=ae_model_config.verbose,
                model_config=ae_model_config,
            )
            model_config.encoder.save_weights(f"./checkpoints/{ae_model_config.model_name}_CV{cv_fold_count}")
            # Generate train and validation dataset filter
            train_filter = (data_feeder.predictors_dates >= first_train_dates[1]) & (
                data_feeder.predictors_dates <= train_dates[train_idx[-1]]
            )
            val_filter = (data_feeder.predictors_dates >= train_dates[test_idx[0]]) & (
                data_feeder.predictors_dates <= train_dates[test_idx[-1]]
            )
    
            model, train_metrics_dict[cv_fold_count] = train_NN(
                data_feeder=data_feeder,
                train_filter=train_filter,
                val_filter=val_filter,
                verbose=model_config.verbose,
                model_config=model_config,
                model_type="transfer",  # transfer learning: model_config.encoder != None
            )
            # https://www.tensorflow.org/tutorials/keras/save_and_load
            model.save_weights(f"./checkpoints/{model_config.model_name}_CV{cv_fold_count}")
            # Test dataset evaluation
            test_ds = data_feeder.gen_tf_dataset(test_filter)
            test_metrics[cv_fold_count] = model.evaluate(test_ds, verbose=1)
            # Clear the session to release memory
            K.clear_session()
            with open(
                f"./metrics/train_metrics_dict_{model_name}.pkl", "wb"
            ) as pickle_file:
                pickle.dump(train_metrics_dict, pickle_file)
            with open(f"./metrics/test_metrics_{model_name}.pkl", "wb") as pickle_file:
                pickle.dump(test_metrics, pickle_file)
    
            del model, train_filter, val_filter, test_ds

        cv_fold_count += 1
        # Update last first_train_date
        first_train_dates[0] = first_train_dates[1]

    return train_metrics_dict, test_metrics, ae_metrics_dict
