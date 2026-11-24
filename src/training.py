import gc
import os
import time
import random
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from src.datasets import create_dataset
from src.config import SEEDS, MIN_BATCH_SIZE

def create_run_datasets(
    train_df,
    val_df,
    test_df,
    batch_size
):
    """
    Create the training, validation, and test datasets
    required for a single training run.
    """
    # Create the training dataset with shuffling enabled
    train_dataset = create_dataset(train_df, batch_size, shuffle=True)

    # Create the validation dataset without shuffling
    val_dataset = create_dataset(val_df, batch_size)

    # Create the test dataset without shuffling
    test_dataset = create_dataset(test_df, batch_size)

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )

def set_random_seed(seed):
    """
    Set random seeds across all libraries used 
    during model training to improve 
    experiment reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def create_callbacks():
    """
    Create in-model callbacks.
    """
    return [
        # Stop training early if validation loss has not improved for 10 consecutive epochs.
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        # Reduce the learning rate when validation loss plateaus.
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

def train_model(
    model_fn,
    model_kwargs,
    loss,
    train_dataset,
    val_dataset,
    callbacks,
    class_weight
):
    """
    Train the supplied model on the supplied parameters.
    """
    # Create a model using the supplied architecture and configuration parameters.
    model = model_fn(**model_kwargs)

    # Compile the model using AdamW optimisation, 
    # the selected loss function, and accuracy tracking.
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-4,
            weight_decay=1e-4
        ),
        loss=loss,
        metrics=["accuracy"]
    )
    
    # Define the common training configuration.
    fit_kwargs = {
        "x": train_dataset,
        "validation_data": val_dataset,
        "epochs": 1000,
        "callbacks": callbacks,
        "verbose": 0
    }

    # Apply class weighting when enabled to increase the influence of minority emotion classes.
    if class_weight is not None:
        fit_kwargs["class_weight"] = {
            int(0): float(1.0266046844269623),
            int(1): float(9.406618610747051),
            int(2): float(1.0010460615781582),
            int(3): float(0.5684387684387684),
            int(4): float(0.8491274770777877),
            int(5): float(1.293372978330405),
            int(6): float(0.8260394187886635),
        }

    # Train the model using the assembled configuration.
    history = model.fit(**fit_kwargs)

    return model, history

def evaluate_trained_model(
    model,
    test_dataset
):
    """
    Evaluate the Trained Model.
    """
    # Evaluate the model on the test dataset and return all metrics as a dictionary.
    test_results = model.evaluate(
        test_dataset,
        verbose=0,
        return_dict=True
    )

    # Generate prediction probabilities for every sample in the test dataset.
    y_pred_probs = model.predict(test_dataset, verbose=0)

    # Convert probability distributions into predicted class indices.
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Extract the one-hot encoded true labels from the dataset.
    y_true = np.concatenate([y for _, y in test_dataset])

    # Convert one-hot labels into class indices.
    y_true = np.argmax(y_true, axis=1)

    # Calculate balanced accuracy so that each emotion class contributes equally regardless of dataset imbalance.
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return (
        test_results,
        y_true,
        y_pred,
        balanced_acc
    )

def save_model_results(
    model_name,
    results,
    best_history
):
    """
    Aggregate and Save all Results From num_runs Models.
    """
    # Convert all run results into a DataFrame for easier analysis and storage.
    results_df = pd.DataFrame(results)

    # Calculate summary statistics across runs.
    summary_df = pd.DataFrame({
        "Mean": results_df.mean(numeric_only=True),
        "Std Dev": results_df.std(numeric_only=True),
        "Minimum": results_df.min(numeric_only=True),
        "Maximum": results_df.max(numeric_only=True)
    })

    # Save detailed results from every run.
    results_df.to_csv(
        f"results/{model_name}_all_runs.csv",
        index=False
    )

    # Save aggregated statistics for quick comparison.
    summary_df.to_csv(f"results/{model_name}_summary.csv")

    # Save the training history of the best-performing run for later visualisation and analysis.
    pd.DataFrame(best_history).to_csv(
        f"histories/{model_name}_history.csv",
        index=False
    )

    return results_df, summary_df

def create_final_reports(
    model_name,
    best_y_true,
    best_y_pred,
    emotion_labels
):
    """
    Create and Save a Results Report for all run Models.
    """
    # Generate a classification report containing precision, 
    # recall, F1-score, and support for each emotion class.
    report = classification_report(
        best_y_true,
        best_y_pred,
        target_names=emotion_labels,
        output_dict=True
    )

    # Convert the report into a DataFrame so it can be easily saved and analysed later.
    report_df = pd.DataFrame(report).transpose()

    # Save the classification report for the model.
    report_df.to_csv(f"reports/{model_name}_classification_report.csv")

    # Create a row-normalised confusion matrix to show how often each true emotion is classified as each predicted emotion.
    cm = confusion_matrix(
        best_y_true,
        best_y_pred,
        normalize="true"
    )

    # Save the confusion matrix for later visualisation and comparison.
    pd.DataFrame(cm).to_csv(f"confusion_matrices/{model_name}.csv", index=False)

    # Remove temporary objects to reduce memory usage.
    del report
    del report_df

    return cm

def evaluate_model(
    model_fn,
    model_name,
    loss,
    emotion_labels,
    train_df,
    val_df,
    test_df,
    batch_size,
    class_weight=None,
    model_kwargs=None,
    num_runs=10
):
    # Initiate storage for performance metrics from every successful run.
    results = []

    # Track the best-performing run based on validation accuracy.
    best_val_acc = 0
    best_history = None
    best_run = None
    best_y_true = None
    best_y_pred = None

    # Batch size may be reduced dynamically if GPU memory is exhausted.
    current_batch_size = batch_size

    # Path used to save the best model discovered across all runs.
    model_path = f"models/{model_name}_best.keras"

    print(f"\nTraining {model_name}")

    # Repeat training multiple times using different random seeds.
    for run in range(num_runs):
        # Indicates whether the current run completed successfully.
        run_success = False

        # Select a predefined seed for reproducibility.
        seed = SEEDS[run]

        # Retry the same run if memory issues require a smaller batch size.
        while not run_success:
            # Create fresh datasets using the current batch size.
            train_dataset, val_dataset, test_dataset = (
                create_run_datasets(
                    train_df,
                    val_df,
                    test_df,
                    current_batch_size
                )
            )

            # Apply the selected random seed.
            set_random_seed(seed)

            # Create training callbacks.
            callbacks = create_callbacks()

            print(
                f"{model_name}: "
                f"Run {run+1}/{num_runs} "
                f"(batch={current_batch_size})"
            )

            # Monitor RAM usage before training begins.
            process = psutil.Process(os.getpid())

            print(
                f"RAM: "
                f"{process.memory_info().rss / 1024**3:.2f} GB"
            )
            
            # Start timing the training process.
            start_time = time.perf_counter()

            # Create a fresh copy of model parameters to avoid accidental modification.
            current_model_kwargs = (
                model_kwargs.copy()
                if model_kwargs is not None
                else {}
            )

            try:
                # Build, compile and train the model.
                model, history = train_model(
                    model_fn=model_fn,
                    model_kwargs=current_model_kwargs,
                    loss=loss,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    callbacks=callbacks,
                    class_weight=class_weight
                )
            # Handle GPU out-of-memory errors gracefully.
            except tf.errors.ResourceExhaustedError:
                print(
                    f"{model_name} Run {run+1} "
                    f"OOM at batch size "
                    f"{current_batch_size}"
                )
                try:
                    del model
                except:
                    pass
                    
                # Release memory before retrying.
                K.clear_session()
                gc.collect()

                # Reduce batch size and retry if possible.
                if current_batch_size > MIN_BATCH_SIZE:
                    current_batch_size //= 2
                    print(
                        f"Retrying at "
                        f"batch size "
                        f"{current_batch_size}"
                    )
                    continue

                # Stop attempting the run if the minimum batch size fails.
                print(
                    f"{model_name} Run {run+1} "
                    f"failed at minimum "
                    f"batch size"
                )
                break

            # Evaluate the trained model on the test set.
            (test_results, y_true, y_pred, balanced_acc) = evaluate_trained_model(
                model,
                test_dataset
            )

            # Calculate total training duration.
            training_time = (time.perf_counter() - start_time)

            # Mark the run as successful.
            run_success = True

        # Skip metric collection if training failed completely.
        if not run_success:
            continue

        # Store results from the completed run.
        results.append({
            "seed": seed,
            "train_loss": float(history.history["loss"][-1]),
            "train_accuracy": float(history.history["accuracy"][-1]),
            "val_loss": float(min(history.history["val_loss"])),
            "val_accuracy": float(max(history.history["val_accuracy"])),
            "test_loss": float(test_results["loss"]),
            "test_accuracy": float(test_results["accuracy"]),
            "balanced_accuracy": float(balanced_acc),
            "epochs": len(history.history["loss"]),
            "training_time_seconds": float(training_time),
            "batch_size": int(current_batch_size)
        })

        # Extract the highest validation accuracy achieved in this run.
        current_val_acc = max(
            history.history["val_accuracy"]
        )

        # Replace stored best model information if this run performs better.
        if current_val_acc > best_val_acc:

            best_val_acc = current_val_acc
            best_run = run + 1

            # Store complete training history for later analysis.
            best_history = {
                k: [float(x) for x in v]
                for k, v in history.history.items()
            }

            # Store predictions required for the final classification report.
            best_y_true = y_true.copy()
            best_y_pred = y_pred.copy()

            # Save the best-performing model checkpoint.
            model.save(model_path)

        # Explicitly release memory after each run.
        del model
        del history
        del test_results
        del y_true
        del y_pred
        del process

        K.clear_session()
        gc.collect()

    # Exit early if every run failed.
    if len(results) == 0:

        print(
            f"{model_name} failed "
            f"all runs."
        )

        return None

    # Save per-run metrics and summary statistics.
    save_model_results(
        model_name,
        results,
        best_history
    )

    # Generate classification report and confusion matrix using the best run.
    cm = create_final_reports(
        model_name,
        best_y_true,
        best_y_pred,
        emotion_labels
    )

    print(
        f"{model_name} complete "
        f"(best run = {best_run}, "
        f"best val acc = {best_val_acc:.4f})"
    )

    # Final memory cleanup.
    K.clear_session()
    gc.collect()

    return {
        "model_name": model_name,
        "best_run": best_run,
        "best_val_acc": float(best_val_acc),
        "confusion_matrix": cm,
        "model_path": model_path
    }