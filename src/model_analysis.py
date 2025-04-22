"""
This file contians tools for model results analysis:
1. To generate plots of r2 and loss during trainng and validation
2. To build summary table for metrics of training and testing
3. To load model and generate feature importance analysis
"""

from typing import Callable, Dict, List, Tuple, Optional

import pandas as pd
import plotly.express as px
import tensorflow as tf

from .model_training_utils import load_metrics
from .train_models import r2


# Analysis
def make_plot(train_metrics_dict: Dict):
    """Generate plots of r2 and loss during trainng and validation"""
    plt.figure(figsize=(5, 3))
    # Plot training loss
    for model_num, metrics in train_metrics_dict.items():
        plt.plot(metrics["loss"], label=f"model{model_num}_train loss")
        plt.plot(
            metrics["val_loss"], label=f"model{model_num}_val loss", linestyle="--"
        )
        print("loss: ", metrics["loss"][-1], metrics["val_loss"][-1])

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss for Multiple Models")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()

    # Plot training loss
    plt.figure(figsize=(5, 3))
    for model_num, metrics in train_metrics_dict.items():
        plt.plot(metrics["r2"], label=f"model{model_num}_train r2")
        plt.plot(metrics["val_r2"], label=f"model{model_num}_val r2", linestyle="--")
        print("r2: ", metrics["r2"][-1], metrics["val_r2"][-1])

    # Adding labels and title
    plt.xlabel("Epochs")
    plt.ylabel("r2")
    plt.title("Training and Validation r2 for Multiple Models")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()


def summarize_metrics(
    filename_roots: Tuple, factors_columns: List, num_list: Optional[List] = None
):
    """Build summary table for metrics of training and testing"""
    metrics_summary = pd.DataFrame()
    if num_list is None:
        num_list = list(range(1, 6))
    for num in num_list:
        train_metrics_path = f"./metrics/{filename_roots[0]}{num}.pkl"
        if num == 1 and "single_cate" in train_metrics_path:
            train_metrics_path = train_metrics_path.replace("single_cate_", "")
        train_metrics_dict = load_metrics(train_metrics_path)

        last_metrics = pd.DataFrame()
        for model_cv_num, metrics in train_metrics_dict.items():
            tmp_df = pd.DataFrame(
                [
                    [metrics["loss"][-1], metrics["val_loss"][-1]],
                    [metrics["r2"][-1], metrics["val_r2"][-1]],
                ],
                index=["MSE", "r2"],
                columns=["train", "validation"],
            ).T
            tmp_df = tmp_df.reset_index().rename(columns={"index": "type"})
            tmp_df["cv_num"] = model_cv_num
            last_metrics = pd.concat([last_metrics, tmp_df], ignore_index=True)

        test_metrics_path = f"./metrics/{filename_roots[1]}{num}.pkl"
        if num == 1 and "single_cate" in test_metrics_path:
            test_metrics_path = test_metrics_path.replace("single_cate_", "")
        test_metrics_dict = load_metrics(test_metrics_path)

        test_tmp_df = pd.DataFrame(test_metrics_dict, index=["MSE", "r2"]).T
        test_tmp_df = test_tmp_df.reset_index().rename(columns={"index": "cv_num"})
        test_tmp_df["type"] = "test"

        last_metrics = pd.concat([last_metrics, test_tmp_df], ignore_index=True)
        last_metrics = last_metrics.sort_values(["cv_num", "type"], ignore_index=True)
        last_metrics["factor_type"] = factors_columns[num - 1]
        metrics_summary = pd.concat([metrics_summary, last_metrics], ignore_index=True)
    return metrics_summary


def load_model(
    window_size: int, predictors_size: int, checkpoint_path: str, create_model: Callable
):
    """Load model weights for evaluation"""
    model = create_model(window_size, predictors_size)
    model.load_weights(checkpoint_path).expect_partial()  # Ignore optimizer parameters
    model.trainable = False  # Make evaluation faster

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[r2],
    )
    return model


# DISPLAY FEATURE IMPORTANCE
def plot_importance(metrics_path: str, factor: str, num: int | str):
    """Display feature importance"""
    metrics = load_metrics(metrics_path)
    df = pd.DataFrame(metrics, index=["MSE", "R2"])
    df = df.iloc[:1].T
    df = df.sort_values("MSE")
    df = df.reset_index().rename(columns={"index": "Feature"})

    # Plotting
    fig = px.bar(
        df,
        x="MSE",
        y="Feature",
        orientation="h",  # horizontal
        title=f"Feature Importance: {factor} (Model num={num})",
        labels={"MSE": "CV6 MSE with feature permuted", "Feature": "Feature"},
        height=max(50 * len(df), 300),
        width=800,
    )
    # add baseline MSE
    fig.add_vline(
        x=metrics["baseline"][0],
        line=dict(color="orange", dash="dash"),
        annotation_text=f'Baseline OOS MSE={metrics["baseline"][0]:.3f}',
        annotation_position="bottom right",
    )
    return fig


def save_plots_to_html(
    figures: List, filename: str = "./feature_importance_plots.html"
):
    """Saves a list of Matplotlib figures into a single HTML file."""
    html_content = ""
    for fig in figures:
        html_content += fig.to_html(full_html=False)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
