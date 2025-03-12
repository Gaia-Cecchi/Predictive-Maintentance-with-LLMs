import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import sqlite3
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Set up random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define paths
BASE_PATH = 'test_predictions/test_predictions_data/'
tech_info_path = os.path.join(BASE_PATH, 'CSD102_technical_info.txt')
# Modifica per utilizzare il percorso assoluto
output_dir = 'C:\\Users\\gaia1\\Desktop\\UDOO Lab\\Manutenzione predittiva e LLM\\code 10 - validation\\test_predictions\\dataset'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Read technical information
try:
    with open(tech_info_path, 'r') as file:
        tech_info_content = file.read()
    print("Technical information loaded successfully")
except FileNotFoundError:
    print(f"Technical info file not found at: {tech_info_path}")
    # Create default technical specifications based on common CSD 102-8 parameters
    tech_info_content = """
    CSD 102-8 TECHNICAL SPECIFICATIONS:
    
    General:
    - Model: CSD 102-8
    - Motor power: 55 kW
    - Motor speed: 2950 rpm
    - Weight: 950 kg
    
    Operating Parameters:
    - Operating pressure: 7.5 bar (normal), 5.5-8.0 bar (range)
    - Air delivery: 10.2 m³/min
    - Discharge temperature: 70-95°C (normal), 95-115°C (warning), >115°C (critical)
    - Suction pressure: 1.0-1.5 bar (normal)
    - Vibration level: <2.8 mm/s (normal), 2.8-4.5 mm/s (warning), >4.5 mm/s (critical)
    - Bearing temperature: 60-80°C (normal), 80-95°C (warning), >95°C (critical)
    - Motor current: 95 A (nominal)
    
    Environmental Requirements:
    - Ambient temperature: 3-40°C (operating range)
    - Ambient humidity: 0-90% (non-condensing)
    - Maximum installation altitude: 1000m above sea level
    """
    print("Created default technical specs")

# Extract parameter ranges
param_ranges = {
    'discharge_temp_true': {
        'normal': (70, 95),
        'warning': (95, 115),
        'critical': (115, 150)
    },
    'vibration_true': {
        'normal': (0.5, 2.8),
        'warning': (2.8, 4.5),
        'critical': (4.5, 10.0)
    },
    'discharge_pressure_true': {
        'normal': (5.5, 8.0),
        'warning': (8.0, 9.0),
        'critical': (9.0, 10.0)
    },
    'suction_pressure_true': {
        'normal': (1.0, 1.5),
        'warning': (0.8, 1.0),
        'critical': (0.5, 0.8)
    },
    'bearing_temp_true': {  # Bearing temperature parameter
        'normal': (60, 80),
        'warning': (80, 95),
        'critical': (95, 120)
    },
    'motor_speed_true': {  # Motor speed/RPM parameter
        'normal': (2900, 3000),
        'warning': (2800, 2900),
        'critical': (2600, 2800)
    },
    'ambient_temperature': {
        'normal': (15, 30),
        'hot': (30, 40),
        'cold': (3, 15)
    },
    'humidity': {
        'normal': (40, 70),
        'high': (70, 90),
        'low': (20, 40)
    },
    'atmospheric_pressure': {
        'normal': (1000, 1020),
        'high': (1020, 1040),
        'low': (980, 1000)
    }
}

print("Parameter ranges defined for all monitored values")

# Generate synthetic timestamps for 2000 records
start_date = datetime(2024, 1, 1, 0, 0, 0)
timestamps = [start_date + timedelta(hours=i) for i in range(2000)]

# Function to generate normal operating values
def generate_normal_values(param_name, n_samples):
    param_range = param_ranges.get(param_name, {}).get('normal')
    if not param_range:
        param_range = (0, 1)  # Default range if not specified
    
    # Generate values with slight random variations
    mean = (param_range[0] + param_range[1]) / 2
    std = (param_range[1] - param_range[0]) / 6  # Standard deviation to keep ~99% within range
    
    values = np.random.normal(mean, std, n_samples)
    # Clip to ensure values stay within range
    return np.clip(values, param_range[0], param_range[1])

# Function to generate anomalous values
def generate_anomalous_values(param_name, n_samples, severity='warning'):
    param_range = param_ranges.get(param_name, {}).get(severity)
    if not param_range:
        param_range = (1, 2)  # Default range if not specified
    
    # Generate values with higher variability
    mean = (param_range[0] + param_range[1]) / 2
    std = (param_range[1] - param_range[0]) / 4  # Wider standard deviation
    
    values = np.random.normal(mean, std, n_samples)
    # Clip to ensure values stay within range
    return np.clip(values, param_range[0], param_range[1])

# Function to generate predicted values
def generate_predicted_values(true_values, error_mean=0, error_std=0.05):
    relative_errors = np.random.normal(error_mean, error_std, len(true_values))
    predicted_values = true_values * (1 + relative_errors)
    return predicted_values

# Function to generate weather data
def generate_weather_data(timestamps, n_samples):
    # Base values
    base_temp = 20  # Base temperature (°C)
    base_humidity = 50  # Base humidity (%)
    base_pressure = 1013  # Base atmospheric pressure (hPa)
    
    # Annual cycle
    days_since_start = [(t - timestamps[0]).total_seconds() / (24*3600) for t in timestamps]
    annual_cycle = np.sin([2 * np.pi * d / 365 for d in days_since_start])
    
    # Daily cycle
    hours_of_day = [t.hour for t in timestamps]
    daily_cycle = np.sin([2 * np.pi * h / 24 - np.pi/2 for h in hours_of_day])  # Peak at noon
    
    # Generate temperature with seasonal and daily variations
    temperature = base_temp + 10 * annual_cycle + 5 * daily_cycle + np.random.normal(0, 2, n_samples)
    temperature = np.clip(temperature, 3, 40)  # Clip to operating range
    
    # Generate humidity (inverse correlation with temperature)
    humidity = base_humidity - 20 * annual_cycle + np.random.normal(0, 10, n_samples)
    humidity = np.clip(humidity, 20, 90)
    
    # Generate atmospheric pressure with some random variations
    pressure = base_pressure + np.random.normal(0, 10, n_samples)
    pressure = np.clip(pressure, 980, 1040)
    
    return temperature, humidity, pressure

# Generate the dataset
print("Generating dataset with 2000 records (1000 normal, 500 true anomalies, 500 false positives)...")

# Initialize the dataframe with timestamps
df = pd.DataFrame({'timestamp': timestamps})

# Add ambient conditions (weather data)
ambient_temp, humidity, atm_pressure = generate_weather_data(timestamps, 2000)
df['ambient_temperature'] = ambient_temp
df['humidity'] = humidity
df['atmospheric_pressure'] = atm_pressure

# Generate 1000 normal records and 1000 records with anomalies
normal_indices = np.arange(0, 1000)
anomaly_indices = np.arange(1000, 2000)

# Initialize is_anomaly column
df['is_anomaly'] = 0
df.loc[anomaly_indices, 'is_anomaly'] = 1

# Generate parameters for normal records - including all specified monitoring parameters
for param in ['discharge_temp_true', 'vibration_true', 'discharge_pressure_true', 
              'suction_pressure_true', 'bearing_temp_true', 'motor_speed_true']:
    # Normal values for non-anomaly records
    df.loc[normal_indices, param] = generate_normal_values(param, len(normal_indices))
    
    # Generate predicted values
    df[param.replace('_true', '_pred')] = generate_predicted_values(df[param])

# Categorize anomalies into true anomalies and false positives
true_anomalies = anomaly_indices[:500]  # First 500 are true anomalies
false_positives = anomaly_indices[500:]  # Last 500 are false positives

# Initialize anomaly type column
df['anomaly_type'] = 'NORMAL'
df.loc[true_anomalies, 'anomaly_type'] = 'TRUE_POSITIVE'
df.loc[false_positives, 'anomaly_type'] = 'FALSE_POSITIVE'

# Generate parameters for true anomalies
for i, idx in enumerate(true_anomalies):
    # Randomly select 1-3 parameters to be anomalous
    anomalous_params = random.sample(['discharge_temp_true', 'vibration_true', 
                                     'discharge_pressure_true', 'motor_speed_true',
                                     'bearing_temp_true'], 
                                    k=random.randint(1, 3))
    
    # Determine anomaly severity
    severity = random.choice(['warning', 'critical']) if random.random() > 0.3 else 'warning'
    
    # Apply anomalous values
    for param in anomalous_params:
        df.loc[idx, param] = generate_anomalous_values(param, 1, severity)[0]
    
    # Add some explicit bearing failures
    if random.random() > 0.7:
        df.loc[idx, 'bearing_temp_true'] = generate_anomalous_values('bearing_temp_true', 1, 'critical')[0]
        df.loc[idx, 'vibration_true'] = generate_anomalous_values('vibration_true', 1, 'warning')[0]
    
    # Add some explicit motor speed anomalies
    if random.random() > 0.8:
        df.loc[idx, 'motor_speed_true'] = generate_anomalous_values('motor_speed_true', 1, 'warning')[0]

# Generate parameters for false positives
for i, idx in enumerate(false_positives):
    # False positives are more subtle - usually just one parameter is slightly off
    anomalous_param = random.choice(['discharge_temp_true', 'discharge_pressure_true', 'motor_speed_true'])
    
    # Use warning level (not critical) for false positives
    df.loc[idx, anomalous_param] = generate_anomalous_values(anomalous_param, 1, 'warning')[0]
    
    # For false positives, temperature might be high but vibration is normal
    if anomalous_param == 'discharge_temp_true' and random.random() > 0.3:
        # Make temperature higher but keep vibration very normal
        df.loc[idx, 'discharge_temp_true'] = generate_anomalous_values('discharge_temp_true', 1, 'warning')[0]
        df.loc[idx, 'vibration_true'] = generate_normal_values('vibration_true', 1)[0] * 0.7  # Lower than normal vibration
    
    # Some false positives are due to environmental factors
    if random.random() > 0.6:
        df.loc[idx, 'ambient_temperature'] = np.random.uniform(30, 40)  # High ambient temperature
        # Discharge temperature slightly elevated due to ambient, but still within acceptable range
        if 'discharge_temp_true' not in locals():
            df.loc[idx, 'discharge_temp_true'] = np.random.uniform(90, 105)
        
        # Make sure bearing temperature remains normal for false positives
        df.loc[idx, 'bearing_temp_true'] = generate_normal_values('bearing_temp_true', 1)[0]

# Add bearing status based on bearing temperature and vibration
def determine_bearing_status(row):
    bearing_temp = row['bearing_temp_true']
    vibration = row['vibration_true']
    
    # Perfect condition
    if bearing_temp <= 75 and vibration <= 2.0:
        return 3  # Excellent condition
        
    # Good condition
    elif bearing_temp <= 85 and vibration <= 2.8:
        return 2  # Good condition
        
    # Warning condition
    elif bearing_temp <= 95 and vibration <= 4.0:
        return 1  # Warning condition
        
    # Critical condition
    else:
        return 0  # Critical condition - needs replacement

# Apply bearing status calculation
df['bearing_status'] = df.apply(determine_bearing_status, axis=1)

# Calculate essential derived features
print("Calculating essential derived features...")

# Temperature-vibration ratio
df['temp_vib_ratio'] = df['discharge_temp_true'] / np.maximum(df['vibration_true'], 0.1)

# Pressure ratio
df['pressure_ratio'] = df['discharge_pressure_true'] / np.maximum(df['suction_pressure_true'], 0.1)

# Temperature deviation from ambient
df['temp_ambient_delta'] = df['discharge_temp_true'] - df['ambient_temperature']

# Fill any NaN values from calculations
df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

# Create simple anomaly descriptions for LLM
def generate_simple_description(row):
    if row['anomaly_type'] == 'NORMAL':
        bearing_text = {
            3: "Bearings in excellent condition.",
            2: "Bearings in good condition.",
            1: "Bearings functioning adequately.",
            0: "Bearings within specification."
        }
        return f"Normal operating conditions. All parameters within expected ranges. {bearing_text.get(row['bearing_status'], '')}"
    
    elif row['anomaly_type'] == 'TRUE_POSITIVE':
        if row['bearing_status'] <= 1:
            return f"True anomaly with bearing issues. Bearing temperature: {row['bearing_temp_true']:.1f}°C, vibration: {row['vibration_true']:.2f} mm/s. Maintenance required."
        elif row['vibration_true'] > param_ranges['vibration_true']['warning'][0]:
            return f"True anomaly with elevated vibration ({row['vibration_true']:.2f} mm/s). Maintenance required."
        elif row['motor_speed_true'] < param_ranges['motor_speed_true']['warning'][0]:
            return f"True anomaly with low motor speed ({row['motor_speed_true']:.0f} rpm). Check motor and controls."
        elif row['discharge_temp_true'] > param_ranges['discharge_temp_true']['warning'][0]:
            return f"True anomaly with high temperature ({row['discharge_temp_true']:.1f}°C). Maintenance required."
        else:
            return "True anomaly with abnormal operating parameters. Maintenance required."
    
    elif row['anomaly_type'] == 'FALSE_POSITIVE':
        if row['ambient_temperature'] > 30:
            return f"False positive due to high ambient temperature ({row['ambient_temperature']:.1f}°C). Bearings in normal condition. No maintenance needed."
        elif abs(row['motor_speed_true'] - 2950) < 50:
            return f"False positive alert. Motor speed ({row['motor_speed_true']:.0f} rpm) variations within normal operating range."
        else:
            return "False positive alert. Temporary parameter deviation. Bearings functioning correctly. No maintenance needed."
    
    return "Unclassified anomaly."

# Apply the description function
df['anomaly_description'] = df.apply(generate_simple_description, axis=1)

# Create visualizations of the dataset
print("Creating dataset visualizations with standardized metrics...")

# Set style for plots (English locale for numbers)
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')  # Set English locale for numbers

# Figure 1: Distribution of key parameters by anomaly type
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Distribution of Key Parameters by Anomaly Type', fontsize=16)

# Temperature distribution
sns.histplot(data=df, x='discharge_temp_true', hue='anomaly_type', 
             multiple='stack', bins=30, ax=axes[0, 0])
axes[0, 0].set_title('Discharge Temperature Distribution')
axes[0, 0].set_xlabel('Temperature (°C)')
axes[0, 0].set_ylabel('Count')

# Vibration distribution
sns.histplot(data=df, x='vibration_true', hue='anomaly_type', 
             multiple='stack', bins=30, ax=axes[0, 1])
axes[0, 1].set_title('Vibration Level Distribution')
axes[0, 1].set_xlabel('Vibration (mm/s)')
axes[0, 1].set_ylabel('Count')

# Bearing temperature distribution
sns.histplot(data=df, x='bearing_temp_true', hue='anomaly_type', 
             multiple='stack', bins=30, ax=axes[1, 0])
axes[1, 0].set_title('Bearing Temperature Distribution')
axes[1, 0].set_xlabel('Temperature (°C)')
axes[1, 0].set_ylabel('Count')

# Motor speed distribution
sns.histplot(data=df, x='motor_speed_true', hue='anomaly_type', 
             multiple='stack', bins=30, ax=axes[1, 1])
axes[1, 1].set_title('Motor Speed Distribution')
axes[1, 1].set_xlabel('Speed (RPM)')
axes[1, 1].set_ylabel('Count')

# Pressure distributions
sns.histplot(data=df, x='discharge_pressure_true', hue='anomaly_type', 
             multiple='stack', bins=30, ax=axes[2, 0])
axes[2, 0].set_title('Discharge Pressure Distribution')
axes[2, 0].set_xlabel('Pressure (bar)')
axes[2, 0].set_ylabel('Count')

# Bearing status distribution
sns.countplot(data=df, x='bearing_status', hue='anomaly_type', ax=axes[2, 1])
axes[2, 1].set_title('Bearing Status Distribution')
axes[2, 1].set_xlabel('Bearing Status (0:Critical - 3:Excellent)')
axes[2, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'parameter_distributions.png'), dpi=300)
plt.close()

# Figure 2: Scatter plots showing relationships between key parameters
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Parameter Relationships by Anomaly Type', fontsize=16)

# Temperature vs Vibration
sns.scatterplot(data=df, x='discharge_temp_true', y='vibration_true', 
                hue='anomaly_type', alpha=0.7, ax=axes[0, 0])
axes[0, 0].set_title('Discharge Temperature vs Vibration')
axes[0, 0].set_xlabel('Temperature (°C)')
axes[0, 0].set_ylabel('Vibration (mm/s)')
axes[0, 0].axhline(y=param_ranges['vibration_true']['warning'][0], color='orange', linestyle='--')
axes[0, 0].axvline(x=param_ranges['discharge_temp_true']['warning'][0], color='orange', linestyle='--')

# Bearing Temperature vs Vibration
sns.scatterplot(data=df, x='bearing_temp_true', y='vibration_true', 
                hue='anomaly_type', alpha=0.7, ax=axes[0, 1])
axes[0, 1].set_title('Bearing Temperature vs Vibration')
axes[0, 1].set_xlabel('Bearing Temperature (°C)')
axes[0, 1].set_ylabel('Vibration (mm/s)')
axes[0, 1].axhline(y=param_ranges['vibration_true']['warning'][0], color='orange', linestyle='--')
axes[0, 1].axvline(x=param_ranges['bearing_temp_true']['warning'][0], color='orange', linestyle='--')

# Motor Speed vs Vibration
sns.scatterplot(data=df, x='motor_speed_true', y='vibration_true', 
                hue='anomaly_type', alpha=0.7, ax=axes[1, 0])
axes[1, 0].set_title('Motor Speed vs Vibration')
axes[1, 0].set_xlabel('Motor Speed (RPM)')
axes[1, 0].set_ylabel('Vibration (mm/s)')
axes[1, 0].axhline(y=param_ranges['vibration_true']['warning'][0], color='orange', linestyle='--')

# Discharge vs Suction Pressure
sns.scatterplot(data=df, x='suction_pressure_true', y='discharge_pressure_true', 
                hue='anomaly_type', alpha=0.7, ax=axes[1, 1])
axes[1, 1].set_title('Suction vs Discharge Pressure')
axes[1, 1].set_xlabel('Suction Pressure (bar)')
axes[1, 1].set_ylabel('Discharge Pressure (bar)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'parameter_relationships.png'), dpi=300)
plt.close()

# Figure 3: Dataset Composition
plt.figure(figsize=(10, 6))
anomaly_counts = df['anomaly_type'].value_counts()
plt.pie(anomaly_counts, labels=anomaly_counts.index, autopct='%1.1f%%', 
        colors=['lightgreen', 'tomato', 'royalblue'], explode=(0.05, 0.05, 0.05))
plt.title('Dataset Composition by Anomaly Type', fontsize=14)
plt.savefig(os.path.join(output_dir, 'dataset_composition.png'), dpi=300)
plt.close()

# Save outputs in different formats
print("Saving dataset in multiple formats...")

# 1. CSV Format
csv_path = os.path.join(output_dir, 'compressor_monitoring_dataset.csv')
df.to_csv(csv_path, index=False)
print(f"CSV file saved: {csv_path}")

# 2. Excel Format
xlsx_path = os.path.join(output_dir, 'compressor_monitoring_dataset.xlsx')
df.to_excel(xlsx_path, index=False)
print(f"Excel file saved: {xlsx_path}")

# 3. SQLite Format
db_path = os.path.join(output_dir, 'compressor_monitoring_dataset.db')
conn = sqlite3.connect(db_path)
df.to_sql('compressor_monitoring', conn, if_exists='replace', index=False)
conn.close()
print(f"SQLite database saved: {db_path}")

# Print dataset summary
print("\nDataset Summary:")
print(f"Total records: {len(df)}")
print(f"Normal records: {len(df[df['anomaly_type'] == 'NORMAL'])}")
print(f"True anomalies: {len(df[df['anomaly_type'] == 'TRUE_POSITIVE'])}")
print(f"False positives: {len(df[df['anomaly_type'] == 'FALSE_POSITIVE'])}")
print(f"\nParameters included: {df.columns.tolist()}")
print("\nDataset created successfully!")