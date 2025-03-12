
# LSTM Compressor Anomaly Detection Report

## Model Performance

**Overall Accuracy: 100.00%**

### Performance Metrics by Class

| Metric    | True Anomaly | Normal/False Positive |
|-----------|:-----------:|:--------------------:|
| Precision | 100.00% | 100.00% |
| Recall    | 100.00% | 100.00% |
| F1 Score  | 100.00% | 100.00% |

### Confusion Matrix

|                         | Predicted: Normal/FP | Predicted: True Anomaly |
|-------------------------|:--------------------:|:-----------------------:|
| **Actual: Normal/FP**   | 300          | 0              |
| **Actual: True Anomaly**| 0          | 100              |


## Model Architecture
LSTM(64, return_sequences=True) Dropout(0.3) LSTM(32) Dropout(0.2) Dense(16, activation='relu') Dense(1, activation='sigmoid')

## Features Used

discharge_temp_true, vibration_true, discharge_pressure_true, suction_pressure_true, bearing_temp_true, motor_speed_true, ambient_temperature, humidity, atmospheric_pressure, temp_deviation, vibration_deviation, temp_vib_ratio, temp_ambient_delta, pressure_ratio, bearing_status

## Training Information

- Dataset: Compressor Monitoring Dataset (2000 records)
- Training Date: 2025-03-10

## Visualizations

The following visualizations are available in the results directory:

1. Confusion Matrix (`lstm_confusion_matrix.png`)
2. Training History (`lstm_training_history.png`) 
3. ROC Curve (`lstm_roc_curve.png`)
4. Threshold Tuning (`lstm_threshold_tuning.png`)
5. Probability Distribution (`lstm_probability_distribution.png`)

