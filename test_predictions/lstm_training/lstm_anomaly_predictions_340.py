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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from datetime import datetime
import tensorflow as tf

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def train_and_evaluate_on_exact_llm_data(data_file, llm_results_file, output_dir="lstm_results"):
    """
    Train and evaluate LSTM model on exactly the same data used by LLM
    """
    print(f"Loading original data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"Full dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Filter only anomalies
    if 'is_anomaly' in df.columns:
        anomalies_df = df[df['is_anomaly'] == 1].copy()
        print(f"Anomalies filtered: {anomalies_df.shape[0]} rows")
    else:
        anomalies_df = df.copy()
    
    # Load LLM results to extract exact indices
    print(f"Loading LLM results from: {llm_results_file}")
    llm_df = pd.read_csv(llm_results_file)
    print(f"LLM results: {llm_df.shape[0]} rows")
    
    # Extract indices used in LLM evaluation
    if 'index' in llm_df.columns:
        llm_indices = llm_df['index'].tolist()
        print(f"Extracted {len(llm_indices)} indices from LLM results")
    else:
        raise ValueError("LLM results file does not contain 'index' column")
    
    # Create a perfect match between anomalies_df and llm_df
    valid_indices = []
    for idx in llm_indices:
        if idx in anomalies_df.index:
            valid_indices.append(idx)
        else:
            print(f"Warning: LLM index {idx} not found in anomalies dataframe")
    
    print(f"Found {len(valid_indices)} matching indices between LLM results and anomalies")
    
    # Create a subset with only the matching data
    matched_df = anomalies_df.loc[valid_indices].copy()
    print(f"Matched dataframe: {matched_df.shape[0]} rows (should be 340)")
    
    # Extract features and target
    X = matched_df[['discharge_pressure_true', 'suction_pressure_true', 'discharge_temp_true', 
                    'vibration_true', 'ambient_temperature', 'humidity']]
    
    # Add derived features
    if 'discharge_temp_pred' in matched_df.columns and 'vibration_pred' in matched_df.columns:
        # Deviations from predicted values
        matched_df['temp_deviation'] = matched_df['discharge_temp_true'] - matched_df['discharge_temp_pred']
        matched_df['vibration_deviation'] = matched_df['vibration_true'] - matched_df['vibration_pred']
        X['temp_deviation'] = matched_df['temp_deviation']
        X['vibration_deviation'] = matched_df['vibration_deviation']
    
    # Calculate useful ratios
    X['temp_vib_ratio'] = matched_df['discharge_temp_true'] / np.maximum(matched_df['vibration_true'], 0.1)
    X['temp_ambient_ratio'] = matched_df['discharge_temp_true'] / np.maximum(matched_df['ambient_temperature'], 1.0)
    
    # Add binary features based on thresholds
    X['is_temp_high'] = (matched_df['discharge_temp_true'] >= 140).astype(float)
    X['is_vib_high'] = (matched_df['vibration_true'] >= 3.8).astype(float)
    X['is_ambient_high'] = (matched_df['ambient_temperature'] >= 30).astype(float)
    
    # Map labels
    if 'anomaly_type' in matched_df.columns:
        y = matched_df['anomaly_type'].map({'TRUE_POSITIVE': 1, 'FALSE_POSITIVE': 0})
    else:
        # If no labels, use is_anomaly column
        y = matched_df['is_anomaly'].astype(int)
    
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Train/test split - using the SAME data as LLM but in a train/test format
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    print(f"Training label distribution: {y_train.value_counts().to_dict()}")
    print(f"Test label distribution: {y_test.value_counts().to_dict()}")
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    
    # Calculate class weights
    class_weights = {}
    class_counts = np.bincount(y_train)
    total = len(y_train)
    for i, count in enumerate(class_counts):
        class_weights[i] = total / (len(class_counts) * count)
    print(f"Class weights: {class_weights}")
    
    # Build the model
    print("Building and training LSTM model...")
    model = Sequential([
        LSTM(64, input_shape=(1, X_train_lstm.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train_lstm, y_train,
        epochs=50,  # Adjust as needed
        batch_size=16,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )
    
    # Make predictions on test set
    print("Generating predictions...")
    y_pred_proba = model.predict(X_test_lstm)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_tp = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall_tp = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_tp = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    precision_fp = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall_fp = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_fp = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print metrics
    print("\n===== LSTM Model Evaluation =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision - True Anomaly: {precision_tp:.4f}")
    print(f"Recall - True Anomaly: {recall_tp:.4f}")
    print(f"F1 Score - True Anomaly: {f1_tp:.4f}")
    print(f"Precision - False Positive: {precision_fp:.4f}")
    print(f"Recall - False Positive: {recall_fp:.4f}")
    print(f"F1 Score - False Positive: {f1_fp:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=["FALSE POSITIVE", "TRUE ANOMALY"],
               yticklabels=["FALSE POSITIVE", "TRUE ANOMALY"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'LSTM Confusion Matrix - Accuracy: {accuracy:.2%}')
    cm_file = os.path.join(output_dir, "lstm_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_file)
    plt.close()
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
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
    
    # Now generate/save predictions on the FULL 340 LLM dataset for direct comparison
    # This is critical for true apples-to-apples comparison with LLM
    print("\n=== Evaluating on full LLM dataset (340 samples) ===")
    
    # Prepare the full dataset (all 340 samples)
    X_full = X.values  # All features
    y_full = y.values  # All labels
    
    # Scale the data
    X_full_scaled = scaler.transform(X_full)
    
    # Reshape for LSTM
    X_full_lstm = X_full_scaled.reshape(X_full_scaled.shape[0], 1, X_full_scaled.shape[1])
    
    # Predict on full dataset
    full_pred_proba = model.predict(X_full_lstm)
    full_pred = (full_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate full metrics for direct comparison with LLM
    full_accuracy = accuracy_score(y_full, full_pred)
    full_precision_tp = precision_score(y_full, full_pred, pos_label=1, zero_division=0)
    full_recall_tp = recall_score(y_full, full_pred, pos_label=1, zero_division=0)
    full_f1_tp = f1_score(y_full, full_pred, pos_label=1, zero_division=0)
    full_precision_fp = precision_score(y_full, full_pred, pos_label=0, zero_division=0)
    full_recall_fp = recall_score(y_full, full_pred, pos_label=0, zero_division=0)
    full_f1_fp = f1_score(y_full, full_pred, pos_label=0, zero_division=0)
    
    # Confusion matrix on full dataset
    full_cm = confusion_matrix(y_full, full_pred)
    
    print(f"Full dataset (340 samples) accuracy: {full_accuracy:.4f}")
    print(f"Precision - True Anomaly: {full_precision_tp:.4f}")
    print(f"Recall - True Anomaly: {full_recall_tp:.4f}")
    print(f"F1 Score - True Anomaly: {full_f1_tp:.4f}")
    
    print("\nFull dataset confusion matrix:")
    print(full_cm)
    
    # Save full dataset evaluation metrics
    plt.figure(figsize=(10, 8))
    sns.heatmap(full_cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=["FALSE POSITIVE", "TRUE ANOMALY"],
               yticklabels=["FALSE POSITIVE", "TRUE ANOMALY"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'LSTM Confusion Matrix (All 340 samples) - Accuracy: {full_accuracy:.2%}')
    full_cm_file = os.path.join(output_dir, "lstm_full_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(full_cm_file)
    plt.close()
    
    # Save full dataset predictions to CSV
    full_results = pd.DataFrame({
        'index': matched_df.index,
        'actual_type': matched_df['anomaly_type'],
        'actual_label': y_full,
        'predicted_label': full_pred,
        'predicted_probability': full_pred_proba.flatten(),
        'discharge_temp': matched_df['discharge_temp_true'],
        'vibration': matched_df['vibration_true'],
        'ambient_temp': matched_df['ambient_temperature'],
        'is_correct': (y_full == full_pred).astype(int)
    })
    
    # Map labels to text for better readability
    full_results['actual_class'] = full_results['actual_label'].map({1: 'TRUE_POSITIVE', 0: 'FALSE_POSITIVE'})
    full_results['predicted_class'] = full_results['predicted_label'].map({1: 'TRUE_POSITIVE', 0: 'FALSE_POSITIVE'})
    
    # Save to CSV
    predictions_file = os.path.join(output_dir, "lstm_predictions_full.csv")
    full_results.to_csv(predictions_file, index=False)
    print(f"✓ Full results saved to: {predictions_file}")
    
    # Create a comprehensive HTML report
    create_html_report(
        full_accuracy, full_precision_tp, full_recall_tp, full_f1_tp,
        full_precision_fp, full_recall_fp, full_f1_fp,
        full_cm_file, history_file, output_dir
    )
    
    return model, full_accuracy

def create_html_report(accuracy, precision_tp, recall_tp, f1_tp, 
                      precision_fp, recall_fp, f1_fp,
                      cm_file, history_file, output_dir):
    """
    Create a comprehensive HTML report with full metrics
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LSTM Classifier for Anomaly Detection (340 samples)</title>
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
        <h1>LSTM Classifier for Anomaly Detection</h1>
        <p>Evaluation on all 340 samples from LLM dataset</p>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Overall Performance</h2>
        <div class="metric">Accuracy: <span>{accuracy:.2%}</span></div>
        
        <h2>Class Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>True Anomaly</th>
                <th>False Positive</th>
            </tr>
            <tr>
                <td>Precision</td>
                <td>{precision_tp:.2%}</td>
                <td>{precision_fp:.2%}</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>{recall_tp:.2%}</td>
                <td>{recall_fp:.2%}</td>
            </tr>
            <tr>
                <td>F1 Score</td>
                <td>{f1_tp:.2%}</td>
                <td>{f1_fp:.2%}</td>
            </tr>
        </table>
        
        <h2>Confusion Matrix</h2>
        <div class="image-container">
            <img src="{os.path.basename(cm_file)}" alt="Confusion Matrix" width="500">
        </div>
        
        <h2>Training History</h2>
        <div class="image-container">
            <img src="{os.path.basename(history_file)}" alt="Training History" width="700">
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
    </body>
    </html>
    """
    
    report_file = os.path.join(output_dir, "lstm_full_report.html")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Complete HTML report saved to: {report_file}")
    return report_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LSTM on exactly the same 340 anomalies as LLM")
    
    # Default paths
    default_data = "C:\\Users\\gaia1\\Desktop\\UDOO Lab\\Manutenzione predittiva e LLM\\code 10 - validation\\test_predictions\\dataset\\compressor_monitoring_dataset.csv"
    default_llm_results = "C:\\Users\\gaia1\\Desktop\\UDOO Lab\\Manutenzione predittiva e LLM\\code 10 - validation\\llm_results\\llm_predictions.csv"
    default_output = "lstm_results_340"
    
    parser.add_argument("--data", type=str, default=default_data, 
                      help="Path to the original data file")
    parser.add_argument("--llm-results", type=str, default=default_llm_results,
                      help="Path to LLM prediction results (with the 340 anomalies)")
    parser.add_argument("--output", type=str, default=default_output,
                      help="Directory to save results")
    
    args = parser.parse_args()
    
    # Train and evaluate on exactly the same data as LLM
    model, accuracy = train_and_evaluate_on_exact_llm_data(
        args.data, 
        args.llm_results, 
        args.output
    )
    
    # Report final accuracy
    target_accuracy = 0.90
    if accuracy >= target_accuracy:
        print(f"\n✅ SUCCESS! Accuracy {accuracy:.2%} meets target {target_accuracy:.0%}")
    else:
        print(f"\n⚠️ Accuracy {accuracy:.2%} does not meet target {target_accuracy:.0%}")