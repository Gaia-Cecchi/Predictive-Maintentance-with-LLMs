# Anomaly Scenarios for Industrial Compressors

This document contains representative examples of anomalies organized by scenario, distinguishing between true anomalies and false positives.

**Analysis Date:** 2025-03-10 15:58:14
**Created By:** Gaia-Cecchi

## Understanding the Scenarios

Each scenario includes comprehensive examples of both TRUE ANOMALY (requiring maintenance) and FALSE POSITIVE (transitory/environmental) cases to help distinguish between them accurately.

## Critical Mechanical Issues (Primarily TRUE ANOMALY)

### Scenario: Critical Vibration

*Anomalies with critical vibration levels (>4.5 mm/s) indicating mechanical issues*

**Analysis Guidance:**
High vibration levels exceeding 4.5 mm/s indicate mechanical issues such as bearing damage, misalignment, or imbalance. This requires immediate maintenance intervention regardless of other parameters.

Found 3 examples of true anomalies and 0 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 132.4°C (expected: 92.9°C), deviation: 39.5°C (42.5%)
- Discharge Pressure: 8.73 bar (expected: 6.27 bar)
- Vibration: 6.33 mm/s (expected: 1.87 mm/s), deviation: 4.46 mm/s (238.3%)
- Bearing Temperature: 105.4°C
- Motor Speed: 2638 rpm
- Pressure Ratio: 6.73

Environmental conditions:
- Ambient Temperature: 23.5°C
- Temperature-Ambient Delta: 108.9°C
- Humidity: 23%
- Atmospheric Pressure: 1014 hPa
- Atmospheric Pressure Delta: 2.2 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 105.4°C, vibration: 6.33 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 101.6°C (expected: 92.9°C), deviation: 8.7°C (9.4%)
- Discharge Pressure: 8.86 bar (expected: 6.27 bar)
- Vibration: 7.98 mm/s (expected: 1.87 mm/s), deviation: 6.11 mm/s (326.9%)
- Bearing Temperature: 84.8°C
- Motor Speed: 2716 rpm
- Pressure Ratio: 6.83

Environmental conditions:
- Ambient Temperature: 31.9°C
- Temperature-Ambient Delta: 69.8°C
- Humidity: 20%
- Atmospheric Pressure: 1009 hPa
- Atmospheric Pressure Delta: -1.7 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 84.8°C, vibration: 7.98 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 132.3°C (expected: 92.9°C), deviation: 39.4°C (42.4%)
- Discharge Pressure: 9.12 bar (expected: 6.27 bar)
- Vibration: 7.24 mm/s (expected: 1.87 mm/s), deviation: 5.37 mm/s (287.1%)
- Bearing Temperature: 99.5°C
- Motor Speed: 2875 rpm
- Pressure Ratio: 7.03

Environmental conditions:
- Ambient Temperature: 28.3°C
- Temperature-Ambient Delta: 104.0°C
- Humidity: 28%
- Atmospheric Pressure: 1007 hPa
- Atmospheric Pressure Delta: -6.6 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 99.5°C, vibration: 7.24 mm/s. Maintenance required.
```


---

### Scenario: High Vibration

*Anomalies with high vibration (3.5-4.5 mm/s) suggesting developing mechanical problems*

**Analysis Guidance:**
Vibration levels between 3.5-4.5 mm/s indicate developing mechanical problems that require maintenance attention before they worsen to critical levels.

Found 3 examples of true anomalies and 0 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 105.0°C (expected: 92.9°C), deviation: 12.1°C (13.0%)
- Discharge Pressure: 9.29 bar (expected: 6.27 bar)
- Vibration: 3.73 mm/s (expected: 1.87 mm/s), deviation: 1.86 mm/s (99.2%)
- Bearing Temperature: 86.3°C
- Motor Speed: 2865 rpm
- Pressure Ratio: 7.16

Environmental conditions:
- Ambient Temperature: 33.7°C
- Temperature-Ambient Delta: 71.3°C
- Humidity: 20%
- Atmospheric Pressure: 1023 hPa
- Atmospheric Pressure Delta: 9.7 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 86.3°C, vibration: 3.73 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 96.4°C (expected: 92.9°C), deviation: 3.5°C (3.8%)
- Discharge Pressure: 8.42 bar (expected: 6.27 bar)
- Vibration: 3.61 mm/s (expected: 1.87 mm/s), deviation: 1.74 mm/s (92.9%)
- Bearing Temperature: 106.7°C
- Motor Speed: 2653 rpm
- Pressure Ratio: 6.49

Environmental conditions:
- Ambient Temperature: 32.2°C
- Temperature-Ambient Delta: 64.2°C
- Humidity: 34%
- Atmospheric Pressure: 997 hPa
- Atmospheric Pressure Delta: -13.6 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 106.7°C, vibration: 3.61 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 102.4°C (expected: 92.9°C), deviation: 9.5°C (10.3%)
- Discharge Pressure: 9.13 bar (expected: 6.27 bar)
- Vibration: 3.92 mm/s (expected: 1.87 mm/s), deviation: 2.05 mm/s (109.7%)
- Bearing Temperature: 114.3°C
- Motor Speed: 2654 rpm
- Pressure Ratio: 7.04

Environmental conditions:
- Ambient Temperature: 33.5°C
- Temperature-Ambient Delta: 68.9°C
- Humidity: 35%
- Atmospheric Pressure: 999 hPa
- Atmospheric Pressure Delta: -13.7 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 114.3°C, vibration: 3.92 mm/s. Maintenance required.
```


---

### Scenario: Critical Discharge Temperature

*Anomalies with critical discharge temperature (>115°C) indicating overheating*

**Analysis Guidance:**
Discharge temperature above 115°C indicates severe overheating that can damage components and lubricants. This requires immediate maintenance regardless of ambient conditions.

Found 3 examples of true anomalies and 0 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 134.9°C (expected: 92.9°C), deviation: 42.0°C (45.2%)
- Discharge Pressure: 8.41 bar (expected: 6.27 bar)
- Vibration: 3.99 mm/s (expected: 1.87 mm/s), deviation: 2.12 mm/s (113.1%)
- Bearing Temperature: 80.0°C
- Motor Speed: 2866 rpm
- Pressure Ratio: 6.48

Environmental conditions:
- Ambient Temperature: 22.6°C
- Temperature-Ambient Delta: 112.3°C
- Humidity: 45%
- Atmospheric Pressure: 1011 hPa
- Atmospheric Pressure Delta: -1.5 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 80.0°C, vibration: 3.99 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 122.5°C (expected: 92.9°C), deviation: 29.6°C (31.9%)
- Discharge Pressure: 8.27 bar (expected: 6.27 bar)
- Vibration: 4.50 mm/s (expected: 1.87 mm/s), deviation: 2.63 mm/s (140.6%)
- Bearing Temperature: 105.2°C
- Motor Speed: 2674 rpm
- Pressure Ratio: 6.37

Environmental conditions:
- Ambient Temperature: 32.8°C
- Temperature-Ambient Delta: 89.7°C
- Humidity: 48%
- Atmospheric Pressure: 1005 hPa
- Atmospheric Pressure Delta: -5.4 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 105.2°C, vibration: 4.50 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 128.9°C (expected: 92.9°C), deviation: 35.9°C (38.7%)
- Discharge Pressure: 8.35 bar (expected: 6.27 bar)
- Vibration: 3.69 mm/s (expected: 1.87 mm/s), deviation: 1.82 mm/s (97.2%)
- Bearing Temperature: 105.8°C
- Motor Speed: 2830 rpm
- Pressure Ratio: 6.44

Environmental conditions:
- Ambient Temperature: 26.0°C
- Temperature-Ambient Delta: 102.9°C
- Humidity: 28%
- Atmospheric Pressure: 1001 hPa
- Atmospheric Pressure Delta: -9.5 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 105.8°C, vibration: 3.69 mm/s. Maintenance required.
```


---

### Scenario: High Discharge Temperature

*Anomalies with high discharge temperature (105-115°C) not related to ambient conditions*

**Analysis Guidance:**
Discharge temperature between 105-115°C with a high differential from ambient temperature indicates internal heating issues requiring maintenance.

Found 3 examples of true anomalies and 3 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 107.2°C (expected: 92.9°C), deviation: 14.3°C (15.4%)
- Discharge Pressure: 8.56 bar (expected: 6.27 bar)
- Vibration: 4.23 mm/s (expected: 1.87 mm/s), deviation: 2.36 mm/s (126.0%)
- Bearing Temperature: 116.6°C
- Motor Speed: 2840 rpm
- Pressure Ratio: 6.60

Environmental conditions:
- Ambient Temperature: 29.2°C
- Temperature-Ambient Delta: 78.0°C
- Humidity: 28%
- Atmospheric Pressure: 1015 hPa
- Atmospheric Pressure Delta: 4.5 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 116.6°C, vibration: 4.23 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 108.6°C (expected: 92.9°C), deviation: 15.7°C (16.9%)
- Discharge Pressure: 8.42 bar (expected: 6.27 bar)
- Vibration: 3.41 mm/s (expected: 1.87 mm/s), deviation: 1.54 mm/s (82.3%)
- Bearing Temperature: 102.9°C
- Motor Speed: 2823 rpm
- Pressure Ratio: 6.49

Environmental conditions:
- Ambient Temperature: 31.8°C
- Temperature-Ambient Delta: 76.8°C
- Humidity: 38%
- Atmospheric Pressure: 1015 hPa
- Atmospheric Pressure Delta: 4.4 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 102.9°C, vibration: 3.41 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 115.0°C (expected: 92.9°C), deviation: 22.1°C (23.8%)
- Discharge Pressure: 8.21 bar (expected: 6.27 bar)
- Vibration: 3.57 mm/s (expected: 1.87 mm/s), deviation: 1.70 mm/s (91.0%)
- Bearing Temperature: 97.5°C
- Motor Speed: 2846 rpm
- Pressure Ratio: 6.33

Environmental conditions:
- Ambient Temperature: 32.2°C
- Temperature-Ambient Delta: 82.8°C
- Humidity: 27%
- Atmospheric Pressure: 1006 hPa
- Atmospheric Pressure Delta: -7.6 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 97.5°C, vibration: 3.57 mm/s. Maintenance required.
```

#### False Positives

**Example 1:**
Compressor parameters:
- Discharge Temperature: 110.0°C (expected: 92.9°C), deviation: 17.1°C (18.4%)
- Discharge Pressure: 8.32 bar (expected: 6.27 bar)
- Vibration: 1.38 mm/s (expected: 1.87 mm/s), deviation: -0.49 mm/s (-26.0%)
- Bearing Temperature: 74.4°C
- Motor Speed: 2850 rpm
- Pressure Ratio: 6.41

Environmental conditions:
- Ambient Temperature: 32.4°C
- Temperature-Ambient Delta: 77.5°C
- Humidity: 44%
- Atmospheric Pressure: 1000 hPa
- Atmospheric Pressure Delta: -9.9 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (32.4°C). Bearings in normal condition. No maintenance needed.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 108.8°C (expected: 92.9°C), deviation: 15.9°C (17.1%)
- Discharge Pressure: 8.44 bar (expected: 6.27 bar)
- Vibration: 0.85 mm/s (expected: 1.87 mm/s), deviation: -1.02 mm/s (-54.7%)
- Bearing Temperature: 73.7°C
- Motor Speed: 2828 rpm
- Pressure Ratio: 6.51

Environmental conditions:
- Ambient Temperature: 30.6°C
- Temperature-Ambient Delta: 78.2°C
- Humidity: 29%
- Atmospheric Pressure: 1012 hPa
- Atmospheric Pressure Delta: -3.7 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (30.6°C). Bearings in normal condition. No maintenance needed.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 108.0°C (expected: 92.9°C), deviation: 15.1°C (16.3%)
- Discharge Pressure: 8.27 bar (expected: 6.27 bar)
- Vibration: 0.85 mm/s (expected: 1.87 mm/s), deviation: -1.02 mm/s (-54.7%)
- Bearing Temperature: 76.4°C
- Motor Speed: 2828 rpm
- Pressure Ratio: 6.38

Environmental conditions:
- Ambient Temperature: 32.1°C
- Temperature-Ambient Delta: 75.9°C
- Humidity: 36%
- Atmospheric Pressure: 1021 hPa
- Atmospheric Pressure Delta: 5.6 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (32.1°C). Bearings in normal condition. No maintenance needed.
```


---

### Scenario: Bearing Temperature Critical

*Anomalies with bearing temperature above critical threshold (>95°C)*

**Analysis Guidance:**
Bearing temperatures exceeding 95°C indicate bearing damage or lubrication failure. This requires immediate maintenance to prevent catastrophic failure.

Found 3 examples of true anomalies and 0 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 128.9°C (expected: 92.9°C), deviation: 35.9°C (38.7%)
- Discharge Pressure: 8.28 bar (expected: 6.27 bar)
- Vibration: 3.69 mm/s (expected: 1.87 mm/s), deviation: 1.82 mm/s (97.2%)
- Bearing Temperature: 105.8°C
- Motor Speed: 2830 rpm
- Pressure Ratio: 6.38

Environmental conditions:
- Ambient Temperature: 27.2°C
- Temperature-Ambient Delta: 101.6°C
- Humidity: 20%
- Atmospheric Pressure: 1007 hPa
- Atmospheric Pressure Delta: -3.4 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 105.8°C, vibration: 3.69 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 103.2°C (expected: 92.9°C), deviation: 10.3°C (11.1%)
- Discharge Pressure: 8.40 bar (expected: 6.27 bar)
- Vibration: 7.57 mm/s (expected: 1.87 mm/s), deviation: 5.70 mm/s (304.7%)
- Bearing Temperature: 105.9°C
- Motor Speed: 2847 rpm
- Pressure Ratio: 6.47

Environmental conditions:
- Ambient Temperature: 25.2°C
- Temperature-Ambient Delta: 78.0°C
- Humidity: 22%
- Atmospheric Pressure: 1009 hPa
- Atmospheric Pressure Delta: -3.3 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 105.9°C, vibration: 7.57 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 108.8°C (expected: 92.9°C), deviation: 15.9°C (17.1%)
- Discharge Pressure: 9.87 bar (expected: 6.27 bar)
- Vibration: 3.23 mm/s (expected: 1.87 mm/s), deviation: 1.36 mm/s (72.9%)
- Bearing Temperature: 104.4°C
- Motor Speed: 2854 rpm
- Pressure Ratio: 7.61

Environmental conditions:
- Ambient Temperature: 29.0°C
- Temperature-Ambient Delta: 79.9°C
- Humidity: 31%
- Atmospheric Pressure: 1012 hPa
- Atmospheric Pressure Delta: -0.1 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 104.4°C, vibration: 3.23 mm/s. Maintenance required.
```


---

### Scenario: Bearing Temperature High

*Anomalies with elevated bearing temperature (85-95°C)*

**Analysis Guidance:**
Bearing temperatures between 85-95°C indicate developing bearing issues or inadequate lubrication requiring maintenance intervention.

Found 3 examples of true anomalies and 0 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 125.2°C (expected: 92.9°C), deviation: 32.3°C (34.7%)
- Discharge Pressure: 8.42 bar (expected: 6.27 bar)
- Vibration: 3.65 mm/s (expected: 1.87 mm/s), deviation: 1.78 mm/s (95.1%)
- Bearing Temperature: 89.3°C
- Motor Speed: 2860 rpm
- Pressure Ratio: 6.49

Environmental conditions:
- Ambient Temperature: 25.9°C
- Temperature-Ambient Delta: 99.3°C
- Humidity: 28%
- Atmospheric Pressure: 1019 hPa
- Atmospheric Pressure Delta: 8.8 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 89.3°C, vibration: 3.65 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 107.5°C (expected: 92.9°C), deviation: 14.6°C (15.8%)
- Discharge Pressure: 8.63 bar (expected: 6.27 bar)
- Vibration: 3.77 mm/s (expected: 1.87 mm/s), deviation: 1.90 mm/s (101.8%)
- Bearing Temperature: 87.3°C
- Motor Speed: 2827 rpm
- Pressure Ratio: 6.65

Environmental conditions:
- Ambient Temperature: 30.4°C
- Temperature-Ambient Delta: 77.2°C
- Humidity: 32%
- Atmospheric Pressure: 1015 hPa
- Atmospheric Pressure Delta: 1.7 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 87.3°C, vibration: 3.77 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 100.6°C (expected: 92.9°C), deviation: 7.7°C (8.3%)
- Discharge Pressure: 8.36 bar (expected: 6.27 bar)
- Vibration: 4.00 mm/s (expected: 1.87 mm/s), deviation: 2.13 mm/s (113.8%)
- Bearing Temperature: 95.0°C
- Motor Speed: 2863 rpm
- Pressure Ratio: 6.44

Environmental conditions:
- Ambient Temperature: 28.0°C
- Temperature-Ambient Delta: 72.6°C
- Humidity: 34%
- Atmospheric Pressure: 1020 hPa
- Atmospheric Pressure Delta: 5.6 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 95.0°C, vibration: 4.00 mm/s. Maintenance required.
```


---

### Scenario: Abnormal Motor Speed Low

*Anomalies with abnormally low motor speed (<2800 rpm) indicating motor problems*

**Analysis Guidance:**
Motor speeds below 2800 rpm (when nominal is 2950 rpm) indicate motor issues, excessive load, or electrical problems requiring maintenance.

Found 3 examples of true anomalies and 0 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 106.5°C (expected: 92.9°C), deviation: 13.6°C (14.7%)
- Discharge Pressure: 9.40 bar (expected: 6.27 bar)
- Vibration: 2.96 mm/s (expected: 1.87 mm/s), deviation: 1.09 mm/s (58.1%)
- Bearing Temperature: 88.1°C
- Motor Speed: 2636 rpm
- Pressure Ratio: 7.25

Environmental conditions:
- Ambient Temperature: 23.5°C
- Temperature-Ambient Delta: 83.0°C
- Humidity: 33%
- Atmospheric Pressure: 998 hPa
- Atmospheric Pressure Delta: -14.6 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 88.1°C, vibration: 2.96 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 108.9°C (expected: 92.9°C), deviation: 16.0°C (17.2%)
- Discharge Pressure: 9.74 bar (expected: 6.27 bar)
- Vibration: 3.21 mm/s (expected: 1.87 mm/s), deviation: 1.34 mm/s (71.6%)
- Bearing Temperature: 111.9°C
- Motor Speed: 2702 rpm
- Pressure Ratio: 7.51

Environmental conditions:
- Ambient Temperature: 28.8°C
- Temperature-Ambient Delta: 80.0°C
- Humidity: 38%
- Atmospheric Pressure: 1001 hPa
- Atmospheric Pressure Delta: -10.5 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 111.9°C, vibration: 3.21 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 103.5°C (expected: 92.9°C), deviation: 10.6°C (11.4%)
- Discharge Pressure: 8.70 bar (expected: 6.27 bar)
- Vibration: 4.02 mm/s (expected: 1.87 mm/s), deviation: 2.15 mm/s (114.9%)
- Bearing Temperature: 85.3°C
- Motor Speed: 2723 rpm
- Pressure Ratio: 6.70

Environmental conditions:
- Ambient Temperature: 21.4°C
- Temperature-Ambient Delta: 82.1°C
- Humidity: 54%
- Atmospheric Pressure: 1001 hPa
- Atmospheric Pressure Delta: -14.4 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 85.3°C, vibration: 4.02 mm/s. Maintenance required.
```


---

### Scenario: High Pressure Ratio

*Anomalies with elevated pressure ratio (7.0-8.0) suggesting developing issues*

**Analysis Guidance:**
Pressure ratios between 7.0-8.0 indicate developing valve or seal issues that should be addressed during planned maintenance.

Found 3 examples of true anomalies and 0 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 98.1°C (expected: 92.9°C), deviation: 5.2°C (5.6%)
- Discharge Pressure: 9.40 bar (expected: 6.27 bar)
- Vibration: 3.90 mm/s (expected: 1.87 mm/s), deviation: 2.03 mm/s (108.4%)
- Bearing Temperature: 100.0°C
- Motor Speed: 2806 rpm
- Pressure Ratio: 7.24

Environmental conditions:
- Ambient Temperature: 22.5°C
- Temperature-Ambient Delta: 75.5°C
- Humidity: 38%
- Atmospheric Pressure: 1006 hPa
- Atmospheric Pressure Delta: -9.1 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 100.0°C, vibration: 3.90 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 102.4°C (expected: 92.9°C), deviation: 9.5°C (10.3%)
- Discharge Pressure: 9.13 bar (expected: 6.27 bar)
- Vibration: 3.92 mm/s (expected: 1.87 mm/s), deviation: 2.05 mm/s (109.7%)
- Bearing Temperature: 114.3°C
- Motor Speed: 2654 rpm
- Pressure Ratio: 7.04

Environmental conditions:
- Ambient Temperature: 33.5°C
- Temperature-Ambient Delta: 68.9°C
- Humidity: 35%
- Atmospheric Pressure: 999 hPa
- Atmospheric Pressure Delta: -13.7 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 114.3°C, vibration: 3.92 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 108.9°C (expected: 92.9°C), deviation: 16.0°C (17.2%)
- Discharge Pressure: 9.74 bar (expected: 6.27 bar)
- Vibration: 3.31 mm/s (expected: 1.87 mm/s), deviation: 1.44 mm/s (77.1%)
- Bearing Temperature: 111.9°C
- Motor Speed: 2822 rpm
- Pressure Ratio: 7.51

Environmental conditions:
- Ambient Temperature: 35.1°C
- Temperature-Ambient Delta: 73.7°C
- Humidity: 35%
- Atmospheric Pressure: 1018 hPa
- Atmospheric Pressure Delta: 6.7 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 111.9°C, vibration: 3.31 mm/s. Maintenance required.
```


---

### Scenario: Moderate Vibration With High Temperature

*Anomalies with both elevated vibration (3.0-4.5 mm/s) AND high temperature (95-115°C)*

**Analysis Guidance:**
The combination of elevated vibration and high temperature indicates developing mechanical issues that require maintenance before they worsen.

Found 3 examples of true anomalies and 0 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 110.0°C (expected: 92.9°C), deviation: 17.1°C (18.4%)
- Discharge Pressure: 9.41 bar (expected: 6.27 bar)
- Vibration: 3.45 mm/s (expected: 1.87 mm/s), deviation: 1.58 mm/s (84.4%)
- Bearing Temperature: 106.0°C
- Motor Speed: 2853 rpm
- Pressure Ratio: 7.25

Environmental conditions:
- Ambient Temperature: 31.0°C
- Temperature-Ambient Delta: 79.0°C
- Humidity: 38%
- Atmospheric Pressure: 1013 hPa
- Atmospheric Pressure Delta: -1.3 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 106.0°C, vibration: 3.45 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 104.0°C (expected: 92.9°C), deviation: 11.1°C (11.9%)
- Discharge Pressure: 8.40 bar (expected: 6.27 bar)
- Vibration: 3.39 mm/s (expected: 1.87 mm/s), deviation: 1.52 mm/s (81.2%)
- Bearing Temperature: 118.7°C
- Motor Speed: 2830 rpm
- Pressure Ratio: 6.48

Environmental conditions:
- Ambient Temperature: 24.0°C
- Temperature-Ambient Delta: 80.0°C
- Humidity: 33%
- Atmospheric Pressure: 1023 hPa
- Atmospheric Pressure Delta: 7.0 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 118.7°C, vibration: 3.39 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 96.4°C (expected: 92.9°C), deviation: 3.5°C (3.7%)
- Discharge Pressure: 8.05 bar (expected: 6.27 bar)
- Vibration: 4.37 mm/s (expected: 1.87 mm/s), deviation: 2.50 mm/s (133.8%)
- Bearing Temperature: 105.9°C
- Motor Speed: 2819 rpm
- Pressure Ratio: 6.20

Environmental conditions:
- Ambient Temperature: 22.1°C
- Temperature-Ambient Delta: 74.3°C
- Humidity: 28%
- Atmospheric Pressure: 1031 hPa
- Atmospheric Pressure Delta: 18.9 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 105.9°C, vibration: 4.37 mm/s. Maintenance required.
```


---

### Scenario: High Vibration Normal Temperature

*Anomalies with high vibration (>3.5 mm/s) despite normal temperature (70-95°C)*

**Analysis Guidance:**
High vibration with normal temperature typically indicates mechanical issues like bearing wear, misalignment, or loose components requiring maintenance.

Found 3 examples of true anomalies and 0 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 95.0°C (expected: 92.9°C), deviation: 2.1°C (2.3%)
- Discharge Pressure: 8.76 bar (expected: 6.27 bar)
- Vibration: 3.65 mm/s (expected: 1.87 mm/s), deviation: 1.78 mm/s (95.4%)
- Bearing Temperature: 95.0°C
- Motor Speed: 2863 rpm
- Pressure Ratio: 6.75

Environmental conditions:
- Ambient Temperature: 22.1°C
- Temperature-Ambient Delta: 72.9°C
- Humidity: 26%
- Atmospheric Pressure: 1016 hPa
- Atmospheric Pressure Delta: 1.4 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 95.0°C, vibration: 3.65 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 95.0°C (expected: 92.9°C), deviation: 2.1°C (2.3%)
- Discharge Pressure: 9.66 bar (expected: 6.27 bar)
- Vibration: 3.65 mm/s (expected: 1.87 mm/s), deviation: 1.78 mm/s (95.4%)
- Bearing Temperature: 95.0°C
- Motor Speed: 2805 rpm
- Pressure Ratio: 7.45

Environmental conditions:
- Ambient Temperature: 21.0°C
- Temperature-Ambient Delta: 74.0°C
- Humidity: 44%
- Atmospheric Pressure: 1036 hPa
- Atmospheric Pressure Delta: 20.9 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 95.0°C, vibration: 3.65 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 95.0°C (expected: 92.9°C), deviation: 2.1°C (2.3%)
- Discharge Pressure: 9.74 bar (expected: 6.27 bar)
- Vibration: 3.79 mm/s (expected: 1.87 mm/s), deviation: 1.92 mm/s (102.9%)
- Bearing Temperature: 104.0°C
- Motor Speed: 2837 rpm
- Pressure Ratio: 7.51

Environmental conditions:
- Ambient Temperature: 31.8°C
- Temperature-Ambient Delta: 63.2°C
- Humidity: 20%
- Atmospheric Pressure: 1013 hPa
- Atmospheric Pressure Delta: 1.4 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 104.0°C, vibration: 3.79 mm/s. Maintenance required.
```


---

## Environmental and Transient Conditions (Primarily FALSE POSITIVE)

### Scenario: High Ambient Temperature Effect

*Elevated discharge temperature correlated with high ambient temperature (>35°C)*

**Analysis Guidance:**
Elevated discharge temperature that rises proportionally with ambient temperature is typically a normal operating condition in hot environments and not a mechanical issue.

Found 3 examples of true anomalies and 3 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 102.2°C (expected: 92.9°C), deviation: 9.3°C (10.1%)
- Discharge Pressure: 9.22 bar (expected: 6.27 bar)
- Vibration: 3.86 mm/s (expected: 1.87 mm/s), deviation: 1.99 mm/s (106.2%)
- Bearing Temperature: 109.9°C
- Motor Speed: 2873 rpm
- Pressure Ratio: 7.11

Environmental conditions:
- Ambient Temperature: 36.8°C
- Temperature-Ambient Delta: 65.4°C
- Humidity: 26%
- Atmospheric Pressure: 1007 hPa
- Atmospheric Pressure Delta: -3.6 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 109.9°C, vibration: 3.86 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 108.9°C (expected: 92.9°C), deviation: 16.0°C (17.2%)
- Discharge Pressure: 9.74 bar (expected: 6.27 bar)
- Vibration: 3.68 mm/s (expected: 1.87 mm/s), deviation: 1.81 mm/s (96.6%)
- Bearing Temperature: 101.6°C
- Motor Speed: 2846 rpm
- Pressure Ratio: 7.51

Environmental conditions:
- Ambient Temperature: 35.4°C
- Temperature-Ambient Delta: 73.4°C
- Humidity: 37%
- Atmospheric Pressure: 1004 hPa
- Atmospheric Pressure Delta: -7.2 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 101.6°C, vibration: 3.68 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 107.7°C (expected: 92.9°C), deviation: 14.8°C (15.9%)
- Discharge Pressure: 8.24 bar (expected: 6.27 bar)
- Vibration: 3.58 mm/s (expected: 1.87 mm/s), deviation: 1.71 mm/s (91.5%)
- Bearing Temperature: 95.0°C
- Motor Speed: 2713 rpm
- Pressure Ratio: 6.35

Environmental conditions:
- Ambient Temperature: 37.0°C
- Temperature-Ambient Delta: 70.7°C
- Humidity: 34%
- Atmospheric Pressure: 1030 hPa
- Atmospheric Pressure Delta: 17.8 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 95.0°C, vibration: 3.58 mm/s. Maintenance required.
```

#### False Positives

**Example 1:**
Compressor parameters:
- Discharge Temperature: 99.8°C (expected: 92.9°C), deviation: 6.9°C (7.4%)
- Discharge Pressure: 8.14 bar (expected: 6.27 bar)
- Vibration: 1.22 mm/s (expected: 1.87 mm/s), deviation: -0.65 mm/s (-34.7%)
- Bearing Temperature: 71.1°C
- Motor Speed: 2850 rpm
- Pressure Ratio: 6.27

Environmental conditions:
- Ambient Temperature: 36.9°C
- Temperature-Ambient Delta: 62.9°C
- Humidity: 43%
- Atmospheric Pressure: 1017 hPa
- Atmospheric Pressure Delta: 2.3 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (36.9°C). Bearings in normal condition. No maintenance needed.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 95.7°C (expected: 92.9°C), deviation: 2.8°C (3.1%)
- Discharge Pressure: 8.35 bar (expected: 6.27 bar)
- Vibration: 1.26 mm/s (expected: 1.87 mm/s), deviation: -0.61 mm/s (-32.8%)
- Bearing Temperature: 67.6°C
- Motor Speed: 2900 rpm
- Pressure Ratio: 6.44

Environmental conditions:
- Ambient Temperature: 38.1°C
- Temperature-Ambient Delta: 57.7°C
- Humidity: 42%
- Atmospheric Pressure: 1010 hPa
- Atmospheric Pressure Delta: -2.0 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (38.1°C). Bearings in normal condition. No maintenance needed.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 104.9°C (expected: 92.9°C), deviation: 12.0°C (12.9%)
- Discharge Pressure: 8.70 bar (expected: 6.27 bar)
- Vibration: 0.62 mm/s (expected: 1.87 mm/s), deviation: -1.25 mm/s (-66.9%)
- Bearing Temperature: 69.5°C
- Motor Speed: 2868 rpm
- Pressure Ratio: 6.71

Environmental conditions:
- Ambient Temperature: 38.2°C
- Temperature-Ambient Delta: 66.7°C
- Humidity: 21%
- Atmospheric Pressure: 1019 hPa
- Atmospheric Pressure Delta: 7.5 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (38.2°C). Bearings in normal condition. No maintenance needed.
```


---

### Scenario: Transient Temperature Spike

*Short-duration temperature spikes with quick return to normal range*

**Analysis Guidance:**
Brief temperature spikes that quickly normalize are typically due to temporary load changes or environmental factors, not mechanical issues requiring maintenance.

Found 3 examples of true anomalies and 3 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 103.5°C (expected: 92.9°C), deviation: 10.6°C (11.4%)
- Discharge Pressure: 8.79 bar (expected: 6.27 bar)
- Vibration: 4.02 mm/s (expected: 1.87 mm/s), deviation: 2.15 mm/s (114.9%)
- Bearing Temperature: 89.7°C
- Motor Speed: 2864 rpm
- Pressure Ratio: 6.78

Environmental conditions:
- Ambient Temperature: 25.3°C
- Temperature-Ambient Delta: 78.2°C
- Humidity: 33%
- Atmospheric Pressure: 1016 hPa
- Atmospheric Pressure Delta: 1.2 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 89.7°C, vibration: 4.02 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 115.0°C (expected: 92.9°C), deviation: 22.1°C (23.8%)
- Discharge Pressure: 9.26 bar (expected: 6.27 bar)
- Vibration: 6.57 mm/s (expected: 1.87 mm/s), deviation: 4.70 mm/s (251.3%)
- Bearing Temperature: 110.6°C
- Motor Speed: 2707 rpm
- Pressure Ratio: 7.14

Environmental conditions:
- Ambient Temperature: 31.1°C
- Temperature-Ambient Delta: 83.9°C
- Humidity: 32%
- Atmospheric Pressure: 1028 hPa
- Atmospheric Pressure Delta: 14.0 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 110.6°C, vibration: 6.57 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 109.2°C (expected: 92.9°C), deviation: 16.3°C (17.6%)
- Discharge Pressure: 8.68 bar (expected: 6.27 bar)
- Vibration: 4.13 mm/s (expected: 1.87 mm/s), deviation: 2.26 mm/s (120.8%)
- Bearing Temperature: 105.2°C
- Motor Speed: 2833 rpm
- Pressure Ratio: 6.69

Environmental conditions:
- Ambient Temperature: 32.2°C
- Temperature-Ambient Delta: 77.1°C
- Humidity: 45%
- Atmospheric Pressure: 1018 hPa
- Atmospheric Pressure Delta: 2.9 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 105.2°C, vibration: 4.13 mm/s. Maintenance required.
```

#### False Positives

**Example 1:**
Compressor parameters:
- Discharge Temperature: 111.9°C (expected: 92.9°C), deviation: 18.9°C (20.4%)
- Discharge Pressure: 8.57 bar (expected: 6.27 bar)
- Vibration: 1.58 mm/s (expected: 1.87 mm/s), deviation: -0.29 mm/s (-15.7%)
- Bearing Temperature: 64.6°C
- Motor Speed: 2800 rpm
- Pressure Ratio: 6.61

Environmental conditions:
- Ambient Temperature: 29.6°C
- Temperature-Ambient Delta: 82.2°C
- Humidity: 41%
- Atmospheric Pressure: 1032 hPa
- Atmospheric Pressure Delta: 20.1 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive alert. Temporary parameter deviation. Bearings functioning correctly. No maintenance needed.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 101.5°C (expected: 92.9°C), deviation: 8.6°C (9.3%)
- Discharge Pressure: 8.79 bar (expected: 6.27 bar)
- Vibration: 0.92 mm/s (expected: 1.87 mm/s), deviation: -0.95 mm/s (-50.6%)
- Bearing Temperature: 67.7°C
- Motor Speed: 2831 rpm
- Pressure Ratio: 6.78

Environmental conditions:
- Ambient Temperature: 35.1°C
- Temperature-Ambient Delta: 66.4°C
- Humidity: 35%
- Atmospheric Pressure: 996 hPa
- Atmospheric Pressure Delta: -18.5 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (35.1°C). Bearings in normal condition. No maintenance needed.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 100.6°C (expected: 92.9°C), deviation: 7.7°C (8.3%)
- Discharge Pressure: 8.05 bar (expected: 6.27 bar)
- Vibration: 0.99 mm/s (expected: 1.87 mm/s), deviation: -0.88 mm/s (-47.1%)
- Bearing Temperature: 75.8°C
- Motor Speed: 2857 rpm
- Pressure Ratio: 6.20

Environmental conditions:
- Ambient Temperature: 33.1°C
- Temperature-Ambient Delta: 67.4°C
- Humidity: 26%
- Atmospheric Pressure: 1004 hPa
- Atmospheric Pressure Delta: -9.2 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (33.1°C). Bearings in normal condition. No maintenance needed.
```


---

### Scenario: Atmospheric Pressure Fluctuation

*Suction or discharge pressure variations correlated with atmospheric pressure changes*

**Analysis Guidance:**
Pressure variations that correlate with atmospheric pressure changes are natural adaptations of the system to environmental conditions, not mechanical issues.

Found 3 examples of true anomalies and 3 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 107.5°C (expected: 92.9°C), deviation: 14.6°C (15.8%)
- Discharge Pressure: 8.63 bar (expected: 6.27 bar)
- Vibration: 3.65 mm/s (expected: 1.87 mm/s), deviation: 1.78 mm/s (95.2%)
- Bearing Temperature: 106.7°C
- Motor Speed: 2681 rpm
- Pressure Ratio: 6.65

Environmental conditions:
- Ambient Temperature: 29.2°C
- Temperature-Ambient Delta: 78.3°C
- Humidity: 20%
- Atmospheric Pressure: 1033 hPa
- Atmospheric Pressure Delta: 20.6 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 106.7°C, vibration: 3.65 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 147.4°C (expected: 92.9°C), deviation: 54.5°C (58.6%)
- Discharge Pressure: 8.91 bar (expected: 6.27 bar)
- Vibration: 5.24 mm/s (expected: 1.87 mm/s), deviation: 3.37 mm/s (180.1%)
- Bearing Temperature: 112.8°C
- Motor Speed: 2880 rpm
- Pressure Ratio: 6.87

Environmental conditions:
- Ambient Temperature: 26.1°C
- Temperature-Ambient Delta: 121.3°C
- Humidity: 27%
- Atmospheric Pressure: 1007 hPa
- Atmospheric Pressure Delta: -5.8 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 112.8°C, vibration: 5.24 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 100.0°C (expected: 92.9°C), deviation: 7.1°C (7.6%)
- Discharge Pressure: 8.49 bar (expected: 6.27 bar)
- Vibration: 3.85 mm/s (expected: 1.87 mm/s), deviation: 1.98 mm/s (106.1%)
- Bearing Temperature: 110.3°C
- Motor Speed: 2840 rpm
- Pressure Ratio: 6.55

Environmental conditions:
- Ambient Temperature: 30.1°C
- Temperature-Ambient Delta: 69.9°C
- Humidity: 46%
- Atmospheric Pressure: 998 hPa
- Atmospheric Pressure Delta: -15.3 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 110.3°C, vibration: 3.85 mm/s. Maintenance required.
```

#### False Positives

**Example 1:**
Compressor parameters:
- Discharge Temperature: 100.8°C (expected: 92.9°C), deviation: 7.9°C (8.5%)
- Discharge Pressure: 8.52 bar (expected: 6.27 bar)
- Vibration: 1.13 mm/s (expected: 1.87 mm/s), deviation: -0.74 mm/s (-39.7%)
- Bearing Temperature: 69.6°C
- Motor Speed: 2880 rpm
- Pressure Ratio: 6.57

Environmental conditions:
- Ambient Temperature: 38.3°C
- Temperature-Ambient Delta: 62.5°C
- Humidity: 43%
- Atmospheric Pressure: 1007 hPa
- Atmospheric Pressure Delta: -8.5 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (38.3°C). Bearings in normal condition. No maintenance needed.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 97.1°C (expected: 92.9°C), deviation: 4.2°C (4.5%)
- Discharge Pressure: 8.44 bar (expected: 6.27 bar)
- Vibration: 0.98 mm/s (expected: 1.87 mm/s), deviation: -0.89 mm/s (-47.6%)
- Bearing Temperature: 65.0°C
- Motor Speed: 2812 rpm
- Pressure Ratio: 6.50

Environmental conditions:
- Ambient Temperature: 24.3°C
- Temperature-Ambient Delta: 72.9°C
- Humidity: 39%
- Atmospheric Pressure: 1021 hPa
- Atmospheric Pressure Delta: 5.8 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive alert. Temporary parameter deviation. Bearings functioning correctly. No maintenance needed.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 96.1°C (expected: 92.9°C), deviation: 3.1°C (3.4%)
- Discharge Pressure: 8.66 bar (expected: 6.27 bar)
- Vibration: 0.94 mm/s (expected: 1.87 mm/s), deviation: -0.93 mm/s (-49.7%)
- Bearing Temperature: 67.8°C
- Motor Speed: 2851 rpm
- Pressure Ratio: 6.67

Environmental conditions:
- Ambient Temperature: 36.2°C
- Temperature-Ambient Delta: 59.8°C
- Humidity: 36%
- Atmospheric Pressure: 1000 hPa
- Atmospheric Pressure Delta: -11.8 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (36.2°C). Bearings in normal condition. No maintenance needed.
```


---

### Scenario: Normal Startup Conditions

*Parameter variations typical during compressor startup or load changes*

**Analysis Guidance:**
Parameter variations during startup or load changes are normal transient conditions and not indicative of mechanical issues requiring maintenance.

Found 3 examples of true anomalies and 3 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 97.2°C (expected: 92.9°C), deviation: 4.3°C (4.6%)
- Discharge Pressure: 9.49 bar (expected: 6.27 bar)
- Vibration: 3.69 mm/s (expected: 1.87 mm/s), deviation: 1.82 mm/s (97.1%)
- Bearing Temperature: 85.4°C
- Motor Speed: 2816 rpm
- Pressure Ratio: 7.32

Environmental conditions:
- Ambient Temperature: 26.3°C
- Temperature-Ambient Delta: 70.9°C
- Humidity: 20%
- Atmospheric Pressure: 1018 hPa
- Atmospheric Pressure Delta: 2.3 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 85.4°C, vibration: 3.69 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 96.7°C (expected: 92.9°C), deviation: 3.8°C (4.1%)
- Discharge Pressure: 9.40 bar (expected: 6.27 bar)
- Vibration: 3.75 mm/s (expected: 1.87 mm/s), deviation: 1.88 mm/s (100.5%)
- Bearing Temperature: 107.6°C
- Motor Speed: 2843 rpm
- Pressure Ratio: 7.24

Environmental conditions:
- Ambient Temperature: 27.7°C
- Temperature-Ambient Delta: 69.0°C
- Humidity: 36%
- Atmospheric Pressure: 1022 hPa
- Atmospheric Pressure Delta: 9.1 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 107.6°C, vibration: 3.75 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 102.0°C (expected: 92.9°C), deviation: 9.1°C (9.8%)
- Discharge Pressure: 8.43 bar (expected: 6.27 bar)
- Vibration: 3.27 mm/s (expected: 1.87 mm/s), deviation: 1.40 mm/s (74.7%)
- Bearing Temperature: 113.5°C
- Motor Speed: 2845 rpm
- Pressure Ratio: 6.50

Environmental conditions:
- Ambient Temperature: 31.5°C
- Temperature-Ambient Delta: 70.6°C
- Humidity: 44%
- Atmospheric Pressure: 1021 hPa
- Atmospheric Pressure Delta: 6.1 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 113.5°C, vibration: 3.27 mm/s. Maintenance required.
```

#### False Positives

**Example 1:**
Compressor parameters:
- Discharge Temperature: 93.7°C (expected: 92.9°C), deviation: 0.8°C (0.8%)
- Discharge Pressure: 8.34 bar (expected: 6.27 bar)
- Vibration: 1.44 mm/s (expected: 1.87 mm/s), deviation: -0.43 mm/s (-23.1%)
- Bearing Temperature: 68.5°C
- Motor Speed: 2848 rpm
- Pressure Ratio: 6.43

Environmental conditions:
- Ambient Temperature: 27.3°C
- Temperature-Ambient Delta: 66.4°C
- Humidity: 20%
- Atmospheric Pressure: 1003 hPa
- Atmospheric Pressure Delta: -9.7 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive alert. Temporary parameter deviation. Bearings functioning correctly. No maintenance needed.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 94.7°C (expected: 92.9°C), deviation: 1.8°C (1.9%)
- Discharge Pressure: 8.46 bar (expected: 6.27 bar)
- Vibration: 1.20 mm/s (expected: 1.87 mm/s), deviation: -0.67 mm/s (-35.8%)
- Bearing Temperature: 70.6°C
- Motor Speed: 2843 rpm
- Pressure Ratio: 6.52

Environmental conditions:
- Ambient Temperature: 24.9°C
- Temperature-Ambient Delta: 69.7°C
- Humidity: 44%
- Atmospheric Pressure: 1025 hPa
- Atmospheric Pressure Delta: 12.6 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive alert. Temporary parameter deviation. Bearings functioning correctly. No maintenance needed.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 99.8°C (expected: 92.9°C), deviation: 6.9°C (7.4%)
- Discharge Pressure: 8.18 bar (expected: 6.27 bar)
- Vibration: 1.64 mm/s (expected: 1.87 mm/s), deviation: -0.23 mm/s (-12.1%)
- Bearing Temperature: 71.4°C
- Motor Speed: 2840 rpm
- Pressure Ratio: 6.30

Environmental conditions:
- Ambient Temperature: 31.9°C
- Temperature-Ambient Delta: 67.9°C
- Humidity: 35%
- Atmospheric Pressure: 1005 hPa
- Atmospheric Pressure Delta: -5.6 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (31.9°C). Bearings in normal condition. No maintenance needed.
```


---

### Scenario: Moderate Humidity Moderate Temperature

*Moderate discharge temperature increases during moderate humidity (60-85%)*

**Analysis Guidance:**
Moderate temperature increases during periods of moderate humidity are typically due to reduced cooling efficiency rather than mechanical issues.

Found 0 examples of true anomalies and 2 false positives.

#### False Positives

**Example 1:**
Compressor parameters:
- Discharge Temperature: 91.2°C (expected: 92.9°C), deviation: -1.8°C (-1.9%)
- Discharge Pressure: 8.82 bar (expected: 6.27 bar)
- Vibration: 1.29 mm/s (expected: 1.87 mm/s), deviation: -0.58 mm/s (-31.2%)
- Bearing Temperature: 73.0°C
- Motor Speed: 2855 rpm
- Pressure Ratio: 6.80

Environmental conditions:
- Ambient Temperature: 30.8°C
- Temperature-Ambient Delta: 60.3°C
- Humidity: 62%
- Atmospheric Pressure: 1033 hPa
- Atmospheric Pressure Delta: 17.3 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (30.8°C). Bearings in normal condition. No maintenance needed.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 93.2°C (expected: 92.9°C), deviation: 0.3°C (0.3%)
- Discharge Pressure: 8.36 bar (expected: 6.27 bar)
- Vibration: 1.66 mm/s (expected: 1.87 mm/s), deviation: -0.21 mm/s (-11.2%)
- Bearing Temperature: 64.4°C
- Motor Speed: 2878 rpm
- Pressure Ratio: 6.44

Environmental conditions:
- Ambient Temperature: 34.8°C
- Temperature-Ambient Delta: 58.4°C
- Humidity: 64%
- Atmospheric Pressure: 1018 hPa
- Atmospheric Pressure Delta: 6.6 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (34.8°C). Bearings in normal condition. No maintenance needed.
```


---

## Boundary Cases Requiring Careful Analysis

### Scenario: Boundary Discharge Temperature

*Borderline high discharge temperature (95-105°C) requiring contextual analysis*

**Analysis Guidance:**
Temperatures in the 95-105°C range with normal vibration require contextual analysis. If correlated with ambient temperature or transient, it's likely a FALSE_POSITIVE; if persistent or unrelated to ambient conditions, it may be a TRUE_POSITIVE.

Found 3 examples of true anomalies and 3 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 102.3°C (expected: 92.9°C), deviation: 9.4°C (10.1%)
- Discharge Pressure: 8.69 bar (expected: 6.27 bar)
- Vibration: 2.80 mm/s (expected: 1.87 mm/s), deviation: 0.93 mm/s (49.7%)
- Bearing Temperature: 87.5°C
- Motor Speed: 2725 rpm
- Pressure Ratio: 6.70

Environmental conditions:
- Ambient Temperature: 27.1°C
- Temperature-Ambient Delta: 75.3°C
- Humidity: 27%
- Atmospheric Pressure: 1020 hPa
- Atmospheric Pressure Delta: 6.3 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 87.5°C, vibration: 2.80 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 104.2°C (expected: 92.9°C), deviation: 11.3°C (12.2%)
- Discharge Pressure: 8.56 bar (expected: 6.27 bar)
- Vibration: 1.11 mm/s (expected: 1.87 mm/s), deviation: -0.76 mm/s (-40.6%)
- Bearing Temperature: 90.9°C
- Motor Speed: 2835 rpm
- Pressure Ratio: 6.60

Environmental conditions:
- Ambient Temperature: 35.2°C
- Temperature-Ambient Delta: 69.0°C
- Humidity: 31%
- Atmospheric Pressure: 991 hPa
- Atmospheric Pressure Delta: -20.4 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 90.9°C, vibration: 1.11 mm/s. Maintenance required.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 101.3°C (expected: 92.9°C), deviation: 8.3°C (9.0%)
- Discharge Pressure: 8.07 bar (expected: 6.27 bar)
- Vibration: 1.11 mm/s (expected: 1.87 mm/s), deviation: -0.76 mm/s (-40.6%)
- Bearing Temperature: 83.5°C
- Motor Speed: 2844 rpm
- Pressure Ratio: 6.22

Environmental conditions:
- Ambient Temperature: 34.1°C
- Temperature-Ambient Delta: 67.2°C
- Humidity: 52%
- Atmospheric Pressure: 1000 hPa
- Atmospheric Pressure Delta: -11.2 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 83.5°C, vibration: 1.11 mm/s. Maintenance required.
```

#### False Positives

**Example 1:**
Compressor parameters:
- Discharge Temperature: 96.1°C (expected: 92.9°C), deviation: 3.2°C (3.5%)
- Discharge Pressure: 8.19 bar (expected: 6.27 bar)
- Vibration: 1.47 mm/s (expected: 1.87 mm/s), deviation: -0.40 mm/s (-21.4%)
- Bearing Temperature: 76.4°C
- Motor Speed: 2890 rpm
- Pressure Ratio: 6.32

Environmental conditions:
- Ambient Temperature: 23.5°C
- Temperature-Ambient Delta: 72.6°C
- Humidity: 22%
- Atmospheric Pressure: 1025 hPa
- Atmospheric Pressure Delta: 11.4 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive alert. Temporary parameter deviation. Bearings functioning correctly. No maintenance needed.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 103.5°C (expected: 92.9°C), deviation: 10.6°C (11.4%)
- Discharge Pressure: 9.00 bar (expected: 6.27 bar)
- Vibration: 1.18 mm/s (expected: 1.87 mm/s), deviation: -0.69 mm/s (-37.1%)
- Bearing Temperature: 71.8°C
- Motor Speed: 2831 rpm
- Pressure Ratio: 6.94

Environmental conditions:
- Ambient Temperature: 32.5°C
- Temperature-Ambient Delta: 71.0°C
- Humidity: 20%
- Atmospheric Pressure: 1006 hPa
- Atmospheric Pressure Delta: -3.7 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (32.5°C). Bearings in normal condition. No maintenance needed.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 95.2°C (expected: 92.9°C), deviation: 2.3°C (2.5%)
- Discharge Pressure: 8.35 bar (expected: 6.27 bar)
- Vibration: 1.38 mm/s (expected: 1.87 mm/s), deviation: -0.49 mm/s (-26.0%)
- Bearing Temperature: 70.3°C
- Motor Speed: 2850 rpm
- Pressure Ratio: 6.44

Environmental conditions:
- Ambient Temperature: 31.8°C
- Temperature-Ambient Delta: 63.4°C
- Humidity: 34%
- Atmospheric Pressure: 1008 hPa
- Atmospheric Pressure Delta: -2.2 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (31.8°C). Bearings in normal condition. No maintenance needed.
```


---

### Scenario: Boundary Vibration Levels

*Borderline vibration levels (2.8-3.5 mm/s) requiring contextual analysis*

**Analysis Guidance:**
Vibration in the 2.8-3.5 mm/s range with normal temperature requires contextual analysis. If persistent or increasing over time, it's likely a TRUE_POSITIVE; if transient or correlated with specific operating conditions, it may be a FALSE_POSITIVE.

Found 2 examples of true anomalies and 0 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 95.0°C (expected: 92.9°C), deviation: 2.1°C (2.3%)
- Discharge Pressure: 8.04 bar (expected: 6.27 bar)
- Vibration: 3.46 mm/s (expected: 1.87 mm/s), deviation: 1.59 mm/s (85.0%)
- Bearing Temperature: 114.8°C
- Motor Speed: 2819 rpm
- Pressure Ratio: 6.20

Environmental conditions:
- Ambient Temperature: 24.3°C
- Temperature-Ambient Delta: 70.7°C
- Humidity: 25%
- Atmospheric Pressure: 1001 hPa
- Atmospheric Pressure Delta: -9.7 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 114.8°C, vibration: 3.46 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 95.0°C (expected: 92.9°C), deviation: 2.1°C (2.3%)
- Discharge Pressure: 8.04 bar (expected: 6.27 bar)
- Vibration: 3.46 mm/s (expected: 1.87 mm/s), deviation: 1.59 mm/s (85.0%)
- Bearing Temperature: 84.8°C
- Motor Speed: 2866 rpm
- Pressure Ratio: 6.20

Environmental conditions:
- Ambient Temperature: 23.2°C
- Temperature-Ambient Delta: 71.8°C
- Humidity: 36%
- Atmospheric Pressure: 1015 hPa
- Atmospheric Pressure Delta: 4.5 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 84.8°C, vibration: 3.46 mm/s. Maintenance required.
```


---

### Scenario: Borderline Pressure Variations

*Discharge pressure slightly outside normal range (7.5-8.0 bar) requiring analysis*

**Analysis Guidance:**
Pressures in the 7.5-8.0 bar range require contextual analysis. If correlated with load changes or atmospheric pressure variations, it's likely a FALSE_POSITIVE; if persistent or accompanied by other anomalies, it may be a TRUE_POSITIVE.

Found 2 examples of true anomalies and 3 false positives.

#### True Anomalies

**Example 1:**
Compressor parameters:
- Discharge Temperature: 105.0°C (expected: 92.9°C), deviation: 12.1°C (13.0%)
- Discharge Pressure: 8.00 bar (expected: 6.27 bar)
- Vibration: 3.75 mm/s (expected: 1.87 mm/s), deviation: 1.88 mm/s (100.4%)
- Bearing Temperature: 104.3°C
- Motor Speed: 2875 rpm
- Pressure Ratio: 6.17

Environmental conditions:
- Ambient Temperature: 28.8°C
- Temperature-Ambient Delta: 76.2°C
- Humidity: 48%
- Atmospheric Pressure: 1029 hPa
- Atmospheric Pressure Delta: 15.2 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 104.3°C, vibration: 3.75 mm/s. Maintenance required.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 109.2°C (expected: 92.9°C), deviation: 16.3°C (17.6%)
- Discharge Pressure: 8.00 bar (expected: 6.27 bar)
- Vibration: 4.13 mm/s (expected: 1.87 mm/s), deviation: 2.26 mm/s (120.8%)
- Bearing Temperature: 105.2°C
- Motor Speed: 2854 rpm
- Pressure Ratio: 6.17

Environmental conditions:
- Ambient Temperature: 28.4°C
- Temperature-Ambient Delta: 80.9°C
- Humidity: 25%
- Atmospheric Pressure: 1020 hPa
- Atmospheric Pressure Delta: 5.4 hPa
- Anomaly Duration: 1 hours

Technical analysis:
True anomaly with bearing issues. Bearing temperature: 105.2°C, vibration: 4.13 mm/s. Maintenance required.
```

#### False Positives

**Example 1:**
Compressor parameters:
- Discharge Temperature: 92.2°C (expected: 92.9°C), deviation: -0.7°C (-0.7%)
- Discharge Pressure: 8.00 bar (expected: 6.27 bar)
- Vibration: 1.68 mm/s (expected: 1.87 mm/s), deviation: -0.19 mm/s (-10.2%)
- Bearing Temperature: 70.7°C
- Motor Speed: 2879 rpm
- Pressure Ratio: 6.17

Environmental conditions:
- Ambient Temperature: 33.2°C
- Temperature-Ambient Delta: 59.0°C
- Humidity: 36%
- Atmospheric Pressure: 1005 hPa
- Atmospheric Pressure Delta: -8.5 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive due to high ambient temperature (33.2°C). Bearings in normal condition. No maintenance needed.
```

**Example 2:**
Compressor parameters:
- Discharge Temperature: 102.1°C (expected: 92.9°C), deviation: 9.2°C (9.9%)
- Discharge Pressure: 8.00 bar (expected: 6.27 bar)
- Vibration: 1.00 mm/s (expected: 1.87 mm/s), deviation: -0.87 mm/s (-46.5%)
- Bearing Temperature: 73.1°C
- Motor Speed: 2879 rpm
- Pressure Ratio: 6.17

Environmental conditions:
- Ambient Temperature: 24.8°C
- Temperature-Ambient Delta: 77.3°C
- Humidity: 28%
- Atmospheric Pressure: 1034 hPa
- Atmospheric Pressure Delta: 20.1 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive alert. Temporary parameter deviation. Bearings functioning correctly. No maintenance needed.
```

**Example 3:**
Compressor parameters:
- Discharge Temperature: 92.2°C (expected: 92.9°C), deviation: -0.7°C (-0.7%)
- Discharge Pressure: 8.00 bar (expected: 6.27 bar)
- Vibration: 1.68 mm/s (expected: 1.87 mm/s), deviation: -0.19 mm/s (-10.2%)
- Bearing Temperature: 70.7°C
- Motor Speed: 2850 rpm
- Pressure Ratio: 6.17

Environmental conditions:
- Ambient Temperature: 26.4°C
- Temperature-Ambient Delta: 65.8°C
- Humidity: 39%
- Atmospheric Pressure: 1016 hPa
- Atmospheric Pressure Delta: 2.3 hPa
- Anomaly Duration: 1 hours

Technical analysis:
False positive alert. Temporary parameter deviation. Bearings functioning correctly. No maintenance needed.
```


---

