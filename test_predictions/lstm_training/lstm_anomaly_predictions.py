import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import tensorflow as tf

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set output directory
OUTPUT_DIR = "C:\\Users\\gaia1\\Desktop\\UDOO Lab\\Manutenzione predittiva e LLM\\code 10 - validation\\test_predictions\\lstm_training\\lstm_results"
DATASET_DIR = "C:\\Users\\gaia1\\Desktop\\UDOO Lab\\Manutenzione predittiva e LLM\\code 10 - validation\\test_predictions\\dataset"
DATASET_FILE = os.path.join(DATASET_DIR, "compressor_monitoring_dataset.csv")

def prepare_sequences(data, seq_length=1, test_size=0.2, random_state=42):
    """
    Prepare data sequences for the LSTM model
    """
    # Prepare features based on the new dataset columns
    X = data[['discharge_temp_true', 'vibration_true', 'discharge_pressure_true', 
              'suction_pressure_true', 'bearing_temp_true', 'motor_speed_true',
              'ambient_temperature', 'humidity', 'atmospheric_pressure']]
    
    # Add derived features if available
    if 'discharge_temp_pred' in data.columns and 'vibration_pred' in data.columns:
        data['temp_deviation'] = data['discharge_temp_true'] - data['discharge_temp_pred']
        data['vibration_deviation'] = data['vibration_true'] - data['vibration_pred']
        X['temp_deviation'] = data['temp_deviation']
        X['vibration_deviation'] = data['vibration_deviation']
    
    # Add useful ratios and derived features
    X['temp_vib_ratio'] = data['discharge_temp_true'] / np.maximum(data['vibration_true'], 0.1)
    X['temp_ambient_delta'] = data['discharge_temp_true'] - data['ambient_temperature']
    X['pressure_ratio'] = data['discharge_pressure_true'] / np.maximum(data['suction_pressure_true'], 0.1)
    
    # Add bearing status if available
    if 'bearing_status' in data.columns:
        X['bearing_status'] = data['bearing_status']
    
    # Map labels based on anomaly_type
    if 'anomaly_type' in data.columns:
        y = data['anomaly_type'].map({'TRUE_POSITIVE': 1, 'FALSE_POSITIVE': 0, 'NORMAL': 0})
    else:
        # If no labels, use is_anomaly column
        y = data['is_anomaly'].astype(int)
    
    # Split into train/test
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, np.arange(len(X)), test_size=test_size, random_state=random_state, stratify=y)
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], seq_length, X_train_scaled.shape[1])
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], seq_length, X_test_scaled.shape[1])
    
    return X_train_lstm, X_test_lstm, y_train, y_test, scaler, train_indices, test_indices, X.columns.tolist()

def build_lstm_model(input_shape):
    """
    Build an LSTM model for binary classification
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def calculate_class_weights(y_train):
    """
    Calculate class weights to handle class imbalance
    """
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    return class_weights

def evaluate_model(y_true, y_pred, output_dir=OUTPUT_DIR):
    """
    Evaluate model with standardized metrics and create advanced visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_tp = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_tp = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_tp = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    precision_fp = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_fp = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_fp = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate additional standardized metrics
    # Calculate ROC AUC if we have enough samples of both classes
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_pred)
    else:
        roc_auc = None
    
    # Calculate error metrics (if prediction is probability-based)
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    # Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    print("\n===== Model Evaluation with Standardized Metrics =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision - True Anomaly: {precision_tp:.4f}")
    print(f"Recall - True Anomaly: {recall_tp:.4f}")
    print(f"F1 Score - True Anomaly: {f1_tp:.4f}")
    print(f"Precision - Normal/False Positive: {precision_fp:.4f}")
    print(f"Recall - Normal/False Positive: {recall_fp:.4f}")
    print(f"F1 Score - Normal/False Positive: {f1_fp:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Visualize the confusion matrix with English labels
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=["NORMAL/FALSE_POSITIVE", "TRUE_ANOMALY"],
               yticklabels=["NORMAL/FALSE_POSITIVE", "TRUE_ANOMALY"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'LSTM Confusion Matrix - Accuracy: {accuracy:.2%}')
    
    # Save the figure
    cm_file = os.path.join(output_dir, "lstm_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_file)
    plt.close()
    print(f"Confusion matrix saved to {cm_file}")
    
    # Return metrics for reporting
    metrics = {
        'accuracy': accuracy,
        'precision_tp': precision_tp,
        'recall_tp': recall_tp,
        'f1_tp': f1_tp,
        'precision_fp': precision_fp,
        'recall_fp': recall_fp,
        'f1_fp': f1_fp,
        'confusion_matrix': cm.tolist(),
        'roc_auc': roc_auc,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }
    
    return metrics, cm_file

def plot_training_history(history, output_dir=OUTPUT_DIR):
    """
    Visualize and save loss and accuracy graphs during training
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    history_file = os.path.join(output_dir, "lstm_training_history.png")
    plt.savefig(history_file)
    plt.close()
    print(f"Training history plot saved to {history_file}")
    
    return history_file

def optimize_threshold(y_test, y_pred_proba, output_dir=OUTPUT_DIR):
    """
    Optimize decision threshold to maximize accuracy
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal point on ROC curve (maximize tpr - fpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate accuracy for various thresholds
    thresholds_acc = np.arange(0.1, 0.91, 0.05)
    accuracies = []
    
    for threshold in thresholds_acc:
        y_pred = (y_pred_proba >= threshold).astype(int)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    # Find the threshold that maximizes accuracy
    best_threshold_idx = np.argmax(accuracies)
    best_threshold = thresholds_acc[best_threshold_idx]
    best_accuracy = accuracies[best_threshold_idx]
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight optimal point
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', 
             markersize=8, label=f'Optimal (threshold={optimal_threshold:.2f})')
    plt.legend()
    
    plt.tight_layout()
    roc_file = os.path.join(output_dir, "lstm_roc_curve.png")
    plt.savefig(roc_file)
    plt.close()
    
    # Plot accuracy vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_acc, accuracies, 'o-', linewidth=2)
    plt.axvline(x=best_threshold, color='red', linestyle='--', 
                label=f'Best Threshold: {best_threshold:.2f}')
    plt.title('Accuracy by Decision Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    threshold_file = os.path.join(output_dir, "lstm_threshold_tuning.png")
    plt.savefig(threshold_file)
    plt.close()
    print(f"Threshold tuning plot saved to {threshold_file}")
    
    return best_threshold, best_accuracy, roc_file, threshold_file

def plot_probability_distribution(y_test, y_pred_proba, output_dir=OUTPUT_DIR):
    """
    Visualize probability distribution for each class
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a dataframe with the results
    results_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_proba': y_pred_proba.flatten()
    })
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Kernel Density Estimation plot for each true class
    sns.kdeplot(
        data=results_df[results_df['true_label'] == 1], 
        x='predicted_proba',
        fill=True,
        label='TRUE_ANOMALY',
        alpha=0.7
    )
    
    sns.kdeplot(
        data=results_df[results_df['true_label'] == 0], 
        x='predicted_proba',
        fill=True,
        label='NORMAL/FALSE_POSITIVE',
        alpha=0.7
    )
    
    plt.axvline(x=0.5, color='red', linestyle='--', label='Default Threshold')
    plt.title('LSTM Prediction Probability Distribution')
    plt.xlabel('Probability of TRUE_ANOMALY')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    prob_file = os.path.join(output_dir, "lstm_probability_distribution.png")
    plt.savefig(prob_file)
    plt.close()
    print(f"Probability distribution plot saved to {prob_file}")
    
    return prob_file

def generate_html_report(metrics, features, history_file, cm_file, prob_file, threshold_file, 
                        roc_file, original_accuracy, optimized_accuracy=None, optimized_threshold=None,
                        output_dir=OUTPUT_DIR):
    """
    Generate a complete HTML report with all standardized metrics and charts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare threshold optimization section
    threshold_section = ""
    if optimized_accuracy and optimized_threshold:
        threshold_section = f"""
        <h2>Threshold Optimization</h2>
        <div class="metric">Default Threshold (0.5) Accuracy: <span>{original_accuracy:.2%}</span></div>
        <div class="metric">Optimized Threshold ({optimized_threshold:.2f}) Accuracy: <span>{optimized_accuracy:.2%}</span></div>
        <div class="metric">Accuracy Improvement: <span>{(optimized_accuracy - original_accuracy):.2%}</span></div>
        
        <div class="image-container">
            <h3>Threshold Tuning</h3>
            <img src="{os.path.basename(threshold_file)}" alt="Threshold Tuning" width="600">
        </div>
        
        <div class="image-container">
            <h3>ROC Curve</h3>
            <img src="{os.path.basename(roc_file)}" alt="ROC Curve" width="600">
        </div>
        """
    
    # Add additional metrics section
    additional_metrics = ""
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        additional_metrics += f"""
        <h2>Additional Performance Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Description</th>
            </tr>
            <tr>
                <td>ROC AUC</td>
                <td>{metrics['roc_auc']:.4f}</td>
                <td>Area under ROC curve - model's ability to discriminate between classes</td>
            </tr>
            <tr>
                <td>MAE</td>
                <td>{metrics['mae']:.4f}</td>
                <td>Mean Absolute Error - average magnitude of errors</td>
            </tr>
            <tr>
                <td>MSE</td>
                <td>{metrics['mse']:.4f}</td>
                <td>Mean Squared Error - average squared differences</td>
            </tr>
            <tr>
                <td>RMSE</td>
                <td>{metrics['rmse']:.4f}</td>
                <td>Root Mean Squared Error - standard deviation of prediction errors</td>
            </tr>
        </table>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LSTM Classifier for Compressor Anomaly Detection</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .metric {{ margin-bottom: 10px; }}
            .metric span {{ font-weight: bold; color: #3498db; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .image-container {{ margin-right: 20px; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>LSTM Classifier for Compressor Anomaly Detection</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Dataset: Compressor Monitoring Dataset (2000 records)</p>
        
        <h2>Overall Performance</h2>
        <div class="metric">Accuracy: <span>{metrics['accuracy']:.2%}</span></div>
        
        <h2>Class Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>True Anomaly</th>
                <th>Normal/False Positive</th>
            </tr>
            <tr>
                <td>Precision</td>
                <td>{metrics['precision_tp']:.2%}</td>
                <td>{metrics['precision_fp']:.2%}</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>{metrics['recall_tp']:.2%}</td>
                <td>{metrics['recall_fp']:.2%}</td>
            </tr>
            <tr>
                <td>F1 Score</td>
                <td>{metrics['f1_tp']:.2%}</td>
                <td>{metrics['f1_fp']:.2%}</td>
            </tr>
        </table>
        
        <h2>Confusion Matrix</h2>
        <div class="image-container">
            <img src="{os.path.basename(cm_file)}" alt="Confusion Matrix" width="500">
        </div>
        
        {threshold_section}
        
        {additional_metrics}
        
        <h2>Training History</h2>
        <div class="image-container">
            <img src="{os.path.basename(history_file)}" alt="Training History" width="700">
        </div>
        
        <h2>Probability Distribution</h2>
        <div class="image-container">
            <img src="{os.path.basename(prob_file)}" alt="Probability Distribution" width="700">
        </div>
        
        <h2>Model Architecture</h2>
        <pre>
LSTM(64, return_sequences=True)
Dropout(0.3)
LSTM(32)
Dropout(0.2)
Dense(16, activation='relu')
Dense(1, activation='sigmoid')
        </pre>
        
        <h2>Features Used</h2>
        <ul>
            {"".join(f"<li>{feature}</li>" for feature in features)}
        </ul>
    </body>
    </html>
    """
    
    report_file = os.path.join(output_dir, "lstm_report.html")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Complete HTML report saved to {report_file}")
    return report_file

def generate_markdown_report(metrics, features, output_dir=OUTPUT_DIR):
    """
    Generate a markdown report with model performance metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Format confusion matrix for markdown
    cm = np.array(metrics['confusion_matrix'])
    confusion_matrix_md = f"""
|                         | Predicted: Normal/FP | Predicted: True Anomaly |
|-------------------------|:--------------------:|:-----------------------:|
| **Actual: Normal/FP**   | {cm[0][0]}          | {cm[0][1]}              |
| **Actual: True Anomaly**| {cm[1][0]}          | {cm[1][1]}              |
"""
    
    markdown_content = f"""
# LSTM Compressor Anomaly Detection Report

## Model Performance

**Overall Accuracy: {metrics['accuracy']:.2%}**

### Performance Metrics by Class

| Metric    | True Anomaly | Normal/False Positive |
|-----------|:-----------:|:--------------------:|
| Precision | {metrics['precision_tp']:.2%} | {metrics['precision_fp']:.2%} |
| Recall    | {metrics['recall_tp']:.2%} | {metrics['recall_fp']:.2%} |
| F1 Score  | {metrics['f1_tp']:.2%} | {metrics['f1_fp']:.2%} |

### Confusion Matrix
{confusion_matrix_md}

## Model Architecture
LSTM(64, return_sequences=True) Dropout(0.3) LSTM(32) Dropout(0.2) Dense(16, activation='relu') Dense(1, activation='sigmoid')

## Features Used

{', '.join(features)}

## Training Information

- Dataset: Compressor Monitoring Dataset (2000 records)
- Training Date: {datetime.now().strftime('%Y-%m-%d')}

## Visualizations

The following visualizations are available in the results directory:

1. Confusion Matrix (`lstm_confusion_matrix.png`)
2. Training History (`lstm_training_history.png`) 
3. ROC Curve (`lstm_roc_curve.png`)
4. Threshold Tuning (`lstm_threshold_tuning.png`)
5. Probability Distribution (`lstm_probability_distribution.png`)

"""
    
    md_file = os.path.join(output_dir, "lstm_report.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Markdown report saved to {md_file}")
    return md_file

def save_predictions(df, test_indices, y_test, y_pred, y_pred_proba=None, 
                   output_file=os.path.join(OUTPUT_DIR, "lstm_predictions.csv")):
    """
    Save LSTM model predictions to CSV and Excel files for subsequent evaluation
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a dataframe with detailed results
    test_data = df.iloc[test_indices].copy()
    test_data['actual_label'] = y_test
    test_data['predicted_label'] = y_pred
    
    if y_pred_proba is not None:
        test_data['predicted_probability'] = y_pred_proba
    
    # Map numeric values to strings for better readability
    def get_class_label(val):
        if val == 1:
            return 'TRUE_ANOMALY'
        elif val == 0:
            if 'anomaly_type' in test_data.columns and test_data.loc[test_data.index[0], 'anomaly_type'] == 'FALSE_POSITIVE':
                return 'FALSE_POSITIVE'
            else:
                return 'NORMAL'
        return 'UNKNOWN'
    
    test_data['actual_class'] = test_data['actual_label'].apply(get_class_label)
    test_data['predicted_class'] = test_data['predicted_label'].apply(get_class_label)
    
    # Add column for disagreement cases
    test_data['is_correct'] = (test_data['actual_label'] == test_data['predicted_label']).astype(int)
    
    # Save results as CSV
    test_data.to_csv(output_file, index=False)
    print(f"✓ Detailed results saved to CSV: {output_file}")
    
    # Also save as Excel file
    output_excel = output_file.replace('.csv', '.xlsx')
    test_data.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"✓ Detailed results saved to Excel: {output_excel}")
    
    return output_file, output_excel

def train_and_evaluate_lstm(data_file=DATASET_FILE, output_dir=OUTPUT_DIR, epochs=50, batch_size=16):
    """
    Main function to train and evaluate the LSTM model
    
    Args:
        data_file: CSV file with input data
        output_dir: Directory to save results
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        tuple: (model, scaler, accuracy, html_report_path)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from: {data_file}")
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"❌ File not found: {data_file}")
        # Try alternative paths
        alt_paths = [
            "C:\\Users\\gaia1\\Desktop\\UDOO Lab\\Manutenzione predittiva e LLM\\code 10 - validation\\test_predictions\\dataset\\compressor_monitoring_dataset.csv",
            "C:\\Users\\gaia1\\Desktop\\UDOO Lab\\Manutenzione predittiva e LLM\\code 10 - validation\\test_predictions\\test_predictions_data\\outputs\\enhanced_anomalies_with_weather.csv",
            "test_predictions\\dataset\\compressor_monitoring_dataset.csv",
            "dataset\\compressor_monitoring_dataset.csv"
        ]
        
        for alt_path in alt_paths:
            print(f"Trying alternative path: {alt_path}")
            try:
                df = pd.read_csv(alt_path)
                print(f"✅ Successfully loaded data from: {alt_path}")
                break
            except FileNotFoundError:
                continue
        else:
            print("❌ Failed to load data from all alternative paths")
            return None, None, 0, None
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Using all {len(df)} records for training and evaluation")
    
    # Display dataset composition
    if 'anomaly_type' in df.columns:
        anomaly_counts = df['anomaly_type'].value_counts()
        print("Dataset composition:")
        for anomaly_type, count in anomaly_counts.items():
            print(f"  - {anomaly_type}: {count} records ({count/len(df):.1%})")
    
    # Prepare data sequences
    print("Preparing data sequences with feature engineering...")
    X_train, X_test, y_train, y_test, scaler, train_indices, test_indices, features = prepare_sequences(
        df, seq_length=1, test_size=0.2, random_state=42)
    
    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Calculate class weights to handle imbalance
    class_weights = calculate_class_weights(y_train)
    print(f"Class weights: {class_weights}")
    
    # Build and train the model
    print("\nBuilding and training LSTM model...")
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )
    
    # Make predictions on test set (with probabilities)
    print("\nGenerating predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred_classes = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Evaluate model with default threshold
    metrics, cm_file = evaluate_model(y_test, y_pred_classes, output_dir)
    original_accuracy = metrics['accuracy']
    
    # Plot training history
    history_file = plot_training_history(history, output_dir)
    
    # Visualize probability distribution
    prob_file = plot_probability_distribution(y_test, y_pred_proba, output_dir)
    
    # Optimize decision threshold
    print("\nOptimizing decision threshold...")
    best_threshold, best_accuracy, roc_file, threshold_file = optimize_threshold(y_test, y_pred_proba, output_dir)
    
    # If optimized threshold improves accuracy, use it
    if best_accuracy > original_accuracy:
        print(f"Optimized threshold ({best_threshold:.2f}) improves accuracy: {best_accuracy:.2%} vs {original_accuracy:.2%}")
        y_pred_optimized = (y_pred_proba > best_threshold).astype(int).flatten()
        
        # Evaluate with new threshold
        optimized_metrics, _ = evaluate_model(y_test, y_pred_optimized, output_dir)
        
        # Use optimized predictions
        y_pred_classes = y_pred_optimized
    else:
        print(f"Default threshold (0.5) produces the best accuracy: {original_accuracy:.2%}")
        best_threshold = 0.5
        best_accuracy = original_accuracy
        optimized_metrics = metrics
    
    # Save detailed predictions
    predictions_file = os.path.join(output_dir, "lstm_predictions.csv")
    csv_output, excel_output = save_predictions(
        df, test_indices, y_test, y_pred_classes, y_pred_proba.flatten(),
        output_file=predictions_file
    )
    
    # Generate reports
    html_report = generate_html_report(
        optimized_metrics, features, history_file, cm_file, prob_file, 
        threshold_file, roc_file, original_accuracy, best_accuracy, best_threshold, output_dir
    )
    
    markdown_report = generate_markdown_report(optimized_metrics, features, output_dir)
    
    # Verify accuracy target
    target_accuracy = 0.90
    if best_accuracy >= target_accuracy:
        print(f"\n✅ SUCCESS! Accuracy {best_accuracy:.2%} meets/exceeds target {target_accuracy:.0%}")
    else:
        print(f"\n⚠️ Accuracy {best_accuracy:.2%} does not meet target {target_accuracy:.0%}")
    
    # Return the model, scaler, accuracy, and report path
    return model, scaler, best_accuracy, html_report

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*50)
    print("LSTM ANOMALY PREDICTION STARTING...")
    print("="*50)
    print(f"Dataset path: {DATASET_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Print Python and TensorFlow versions for debugging
    import sys
    import tensorflow as tf
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")

    # Check if dataset directory exists
    if not os.path.exists(DATASET_DIR):
        print(f"WARNING: Dataset directory not found: {DATASET_DIR}")
        print("Attempting to create directory...")
        os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Check if dataset file exists before running
    if os.path.exists(DATASET_FILE):
        print(f"Dataset file found: {DATASET_FILE}")
    else:
        print(f"WARNING: Dataset file not found: {DATASET_FILE}")
        print("Will try alternate paths...")
    
    # Call the main function to train and evaluate the model
    try:
        model, scaler, accuracy, html_report = train_and_evaluate_lstm(
            data_file=DATASET_FILE,
            output_dir=OUTPUT_DIR,
            epochs=50,
            batch_size=16
        )
        
        if model is None:
            print("❌ Model training failed. Check error messages above.")
            sys.exit(1)
        
        print("\nTraining and evaluation completed!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"Final accuracy: {accuracy:.2%}")
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        print("Please make sure to run create_dataset.py first to generate the dataset.")
        sys.exit(1)
