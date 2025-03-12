# Guide to Classifying Anomalies in Industrial Compressors

## 1. Anomaly Classification Framework

In industrial compressor monitoring, it is essential to distinguish between **true anomalies** that require maintenance interventions and **false positives** that are temporary variations in operating parameters caused by external or transient factors.

### Definition of True Anomaly (TRUE_POSITIVE)

A true anomaly is a deviation from standard operating parameters that:
- Indicates a deterioration in compressor performance
- Can lead to failures if not addressed
- Requires planned or immediate maintenance intervention
- Persists over time regardless of environmental conditions

### Definition of False Positive (FALSE_POSITIVE)

A false positive is a deviation from standard operating parameters that:
- Is correlated with external factors such as environmental conditions
- Is temporary and tends to normalize without intervention
- Does not indicate actual deterioration in compressor performance
- Does not require maintenance interventions

## 2. Critical Parameters and Thresholds

### Compressor CSD 102-8 Parameters

| Parameter | Normal Range | Warning Threshold | Critical Threshold |
|-----------|---------------|---------------------|---------------|
| Discharge Temperature | 70-95°C | 95-115°C | >115°C |
| Vibration | 0.5-2.8 mm/s | 2.8-4.5 mm/s | >4.5 mm/s |
| Bearing Temperature | 60-80°C | 80-95°C | >95°C |
| Discharge Pressure | 5.5-8.0 bar | 8.0-9.0 bar | >9.0 bar |
| Motor Speed | 2900-3000 rpm | 2800-2900 rpm | <2800 rpm |

### Influence of Environmental Parameters

| Environmental Parameter | Influence on Compressor Parameters |
|---------------------|----------------------------------------|
| High ambient temperature (>30°C) | Can increase discharge temperature by 5-15°C |
| High humidity (>85%) | Can reduce cooling efficiency |
| Atmospheric pressure variations | Can affect suction and discharge pressures |

## 3. Rules for Anomaly Classification

### Critical Override Rules (Highest Priority)

These rules take absolute precedence in classification decisions:

#### Critical Parameters Rule (TRUE ANOMALY)

If ANY of these critical thresholds are exceeded, the case MUST be classified as TRUE ANOMALY regardless of other factors:
- Discharge temperature > 115°C
- Vibration > 4.5 mm/s
- Bearing temperature > 95°C

#### Environmental Correlation Rule (FALSE POSITIVE)

If ANY of these conditions are met, the case should be classified as FALSE POSITIVE unless critical thresholds are exceeded:
- Ambient temperature > 32°C AND discharge temperature < 110°C
- Humidity > 80% AND vibration < 3.8 mm/s

#### Pattern-Specific Rule (TRUE ANOMALY)

If ALL of these specific conditions are met simultaneously, the case should be classified as TRUE ANOMALY despite environmental correlations:
- Discharge temperature between 105°C and 108°C
- Vibration between 3.1 mm/s and 3.8 mm/s
- Ambient temperature between 30°C and 34°C

*Note: This specific pattern indicates an early-stage bearing wear that shows a characteristic thermal-vibration signature even when ambient temperature is elevated.*

#### Normal Parameters Rule (FALSE POSITIVE)

If ALL parameters are within normal operating ranges, the case MUST be classified as FALSE POSITIVE:
- Discharge temperature < 95°C
- Vibration < 2.8 mm/s

### Standard Classification Rules (Secondary Priority)

Apply these rules only if no override rule matches:

#### Indicators of True Anomaly (requires maintenance)

1. **Vibration above critical threshold**: Vibration > 4.5 mm/s indicates mechanical problems
2. **Critical discharge temperature**: Temperature > 115°C not correlated with ambient temperature
3. **Abnormal bearing temperature**: Bearing temperature > 95°C indicates bearing wear/damage
4. **Abnormal discharge pressure**: Pressure > 9.0 bar indicates valve or control problems
5. **Combination of moderate anomalies**: Multiple parameters simultaneously out of range

#### Indicators of False Positive (does not require maintenance)

1. **Correlation with ambient temperature**: Increase in discharge temperature proportional to increase in ambient temperature
2. **Anomalies during high humidity**: Parameters that return to normal when humidity decreases
3. **Correlation with barometric variations**: Pressure variations corresponding to atmospheric changes
4. **Transient anomalies**: Deviations that normalize within a few hours
5. **Multiple small deviations**: Parameters slightly out of range but not in critical zone

## 4. Examples of True Anomalies (TRUE_POSITIVE)

### Example 1

**Compressor parameters:**
- Discharge Temperature: 96.4°C (expected: 92.9°C)
- Discharge Pressure: 8.60 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 3.05 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 103.3°C
- Motor Speed: 2872 rpm

**Environmental conditions:**
- Ambient Temperature: 30.7°C
- Humidity: 20%
- Atmospheric Pressure: 1014 hPa

**Analysis:**
True anomaly with bearing issues. Bearing temperature: 103.3°C, vibration: 3.05 mm/s. Maintenance required.

---

### Example 2

**Compressor parameters:**
- Discharge Temperature: 101.4°C (expected: 92.9°C)
- Discharge Pressure: 9.02 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 3.81 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 89.7°C
- Motor Speed: 2713 rpm

**Environmental conditions:**
- Ambient Temperature: 25.7°C
- Humidity: 52%
- Atmospheric Pressure: 997 hPa

**Analysis:**
True anomaly with bearing issues. Bearing temperature: 89.7°C, vibration: 3.81 mm/s. Maintenance required.

---

### Example 3

**Compressor parameters:**
- Discharge Temperature: 141.4°C (expected: 92.9°C)
- Discharge Pressure: 9.39 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 3.36 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 93.6°C
- Motor Speed: 2831 rpm

**Environmental conditions:**
- Ambient Temperature: 31.3°C
- Humidity: 20%
- Atmospheric Pressure: 1017 hPa

**Analysis:**
True anomaly with bearing issues. Bearing temperature: 93.6°C, vibration: 3.36 mm/s. Maintenance required.

---

### Example 4

**Compressor parameters:**
- Discharge Temperature: 132.4°C (expected: 92.9°C)
- Discharge Pressure: 8.73 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 6.33 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 105.4°C
- Motor Speed: 2701 rpm

**Environmental conditions:**
- Ambient Temperature: 25.9°C
- Humidity: 20%
- Atmospheric Pressure: 1013 hPa

**Analysis:**
True anomaly with bearing issues. Bearing temperature: 105.4°C, vibration: 6.33 mm/s. Maintenance required.

---

### Example 5

**Compressor parameters:**
- Discharge Temperature: 103.2°C (expected: 92.9°C)
- Discharge Pressure: 9.66 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 8.01 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 89.8°C
- Motor Speed: 2666 rpm

**Environmental conditions:**
- Ambient Temperature: 24.3°C
- Humidity: 28%
- Atmospheric Pressure: 1017 hPa

**Analysis:**
True anomaly with bearing issues. Bearing temperature: 89.8°C, vibration: 8.01 mm/s. Maintenance required.

---

### Example 6

**Compressor parameters:**
- Discharge Temperature: 111.4°C (expected: 92.9°C)
- Discharge Pressure: 8.88 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 3.65 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 94.9°C
- Motor Speed: 2833 rpm

**Environmental conditions:**
- Ambient Temperature: 23.4°C
- Humidity: 53%
- Atmospheric Pressure: 999 hPa

**Analysis:**
True anomaly with bearing issues. Bearing temperature: 94.9°C, vibration: 3.65 mm/s. Maintenance required.

---

### Example 7

**Compressor parameters:**
- Discharge Temperature: 141.4°C (expected: 92.9°C)
- Discharge Pressure: 9.22 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 3.51 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 109.9°C
- Motor Speed: 2862 rpm

**Environmental conditions:**
- Ambient Temperature: 34.0°C
- Humidity: 24%
- Atmospheric Pressure: 991 hPa

**Analysis:**
True anomaly with bearing issues. Bearing temperature: 109.9°C, vibration: 3.51 mm/s. Maintenance required.

---

### Example 8

**Compressor parameters:**
- Discharge Temperature: 139.9°C (expected: 92.9°C)
- Discharge Pressure: 8.26 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 3.83 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 108.2°C
- Motor Speed: 2855 rpm

**Environmental conditions:**
- Ambient Temperature: 28.0°C
- Humidity: 34%
- Atmospheric Pressure: 1020 hPa

**Analysis:**
True anomaly with bearing issues. Bearing temperature: 108.2°C, vibration: 3.83 mm/s. Maintenance required.

---

### Example 9

**Compressor parameters:**
- Discharge Temperature: 110.0°C (expected: 92.9°C)
- Discharge Pressure: 9.41 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 3.45 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 106.0°C
- Motor Speed: 2853 rpm

**Environmental conditions:**
- Ambient Temperature: 31.0°C
- Humidity: 38%
- Atmospheric Pressure: 1013 hPa

**Analysis:**
True anomaly with bearing issues. Bearing temperature: 106.0°C, vibration: 3.45 mm/s. Maintenance required.

---

### Example 10

**Compressor parameters:**
- Discharge Temperature: 100.0°C (expected: 92.9°C)
- Discharge Pressure: 8.28 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 3.11 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 90.7°C
- Motor Speed: 2893 rpm

**Environmental conditions:**
- Ambient Temperature: 33.6°C
- Humidity: 28%
- Atmospheric Pressure: 1009 hPa

**Analysis:**
True anomaly with bearing issues. Bearing temperature: 90.7°C, vibration: 3.11 mm/s. Maintenance required.

---

## 5. Examples of False Positives (FALSE_POSITIVE)

### Example 1

**Compressor parameters:**
- Discharge Temperature: 106.0°C (expected: 92.9°C)
- Discharge Pressure: 8.56 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 1.22 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 69.5°C
- Motor Speed: 2848 rpm

**Environmental conditions:**
- Ambient Temperature: 36.2°C
- Humidity: 23%
- Atmospheric Pressure: 1025 hPa

**Analysis:**
False positive due to high ambient temperature (36.2°C). Bearings in normal condition. No maintenance needed.

---

### Example 2

**Compressor parameters:**
- Discharge Temperature: 104.6°C (expected: 92.9°C)
- Discharge Pressure: 8.26 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 0.92 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 73.8°C
- Motor Speed: 2841 rpm

**Environmental conditions:**
- Ambient Temperature: 31.2°C
- Humidity: 34%
- Atmospheric Pressure: 1009 hPa

**Analysis:**
False positive due to high ambient temperature (31.2°C). Bearings in normal condition. No maintenance needed.

---

### Example 3

**Compressor parameters:**
- Discharge Temperature: 100.7°C (expected: 92.9°C)
- Discharge Pressure: 8.31 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 1.45 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 67.4°C
- Motor Speed: 2863 rpm

**Environmental conditions:**
- Ambient Temperature: 37.2°C
- Humidity: 23%
- Atmospheric Pressure: 1026 hPa

**Analysis:**
False positive due to high ambient temperature (37.2°C). Bearings in normal condition. No maintenance needed.

---

### Example 4

**Compressor parameters:**
- Discharge Temperature: 95.3°C (expected: 92.9°C)
- Discharge Pressure: 8.90 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 1.54 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 67.6°C
- Motor Speed: 2847 rpm

**Environmental conditions:**
- Ambient Temperature: 23.9°C
- Humidity: 35%
- Atmospheric Pressure: 1005 hPa

**Analysis:**
False positive alert. Temporary parameter deviation. Bearings functioning correctly. No maintenance needed.

---

### Example 5

**Compressor parameters:**
- Discharge Temperature: 101.2°C (expected: 92.9°C)
- Discharge Pressure: 8.97 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 0.90 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 71.6°C
- Motor Speed: 2861 rpm

**Environmental conditions:**
- Ambient Temperature: 25.7°C
- Humidity: 40%
- Atmospheric Pressure: 1017 hPa

**Analysis:**
False positive alert. Temporary parameter deviation. Bearings functioning correctly. No maintenance needed.

---

### Example 6

**Compressor parameters:**
- Discharge Temperature: 96.3°C (expected: 92.9°C)
- Discharge Pressure: 8.91 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 1.07 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 64.9°C
- Motor Speed: 2861 rpm

**Environmental conditions:**
- Ambient Temperature: 38.8°C
- Humidity: 27%
- Atmospheric Pressure: 1018 hPa

**Analysis:**
False positive due to high ambient temperature (38.8°C). Bearings in normal condition. No maintenance needed.

---

### Example 7

**Compressor parameters:**
- Discharge Temperature: 96.0°C (expected: 92.9°C)
- Discharge Pressure: 8.12 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 1.45 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 71.1°C
- Motor Speed: 2823 rpm

**Environmental conditions:**
- Ambient Temperature: 28.0°C
- Humidity: 33%
- Atmospheric Pressure: 1017 hPa

**Analysis:**
False positive alert. Temporary parameter deviation. Bearings functioning correctly. No maintenance needed.

---

### Example 8

**Compressor parameters:**
- Discharge Temperature: 102.3°C (expected: 92.9°C)
- Discharge Pressure: 8.57 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 1.11 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 70.9°C
- Motor Speed: 2847 rpm

**Environmental conditions:**
- Ambient Temperature: 32.9°C
- Humidity: 51%
- Atmospheric Pressure: 1030 hPa

**Analysis:**
False positive due to high ambient temperature (32.9°C). Bearings in normal condition. No maintenance needed.

---

### Example 9

**Compressor parameters:**
- Discharge Temperature: 94.3°C (expected: 92.9°C)
- Discharge Pressure: 8.94 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 0.92 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 73.7°C
- Motor Speed: 2864 rpm

**Environmental conditions:**
- Ambient Temperature: 38.4°C
- Humidity: 34%
- Atmospheric Pressure: 1016 hPa

**Analysis:**
False positive due to high ambient temperature (38.4°C). Bearings in normal condition. No maintenance needed.

---

### Example 10

**Compressor parameters:**
- Discharge Temperature: 100.0°C (expected: 92.9°C)
- Discharge Pressure: 8.62 bar (expected: 6.27 bar)
- Suction Pressure: 1.30 bar (expected: 1.32 bar)
- Vibration: 0.85 mm/s (expected: 1.87 mm/s)
- Bearing Temperature: 73.1°C
- Motor Speed: 2828 rpm

**Environmental conditions:**
- Ambient Temperature: 26.3°C
- Humidity: 35%
- Atmospheric Pressure: 1014 hPa

**Analysis:**
False positive alert. Temporary parameter deviation. Bearings functioning correctly. No maintenance needed.

---

## 6. Decision Framework for Classification

To correctly classify an anomaly, follow this decision process in strict order of priority:

1. **Apply Critical Override Rules First**
   - Is any critical parameter exceeded? (Temperature > 115°C OR Vibration > 4.5 mm/s OR Bearing Temperature > 95°C) → TRUE ANOMALY
   - Is the specific pattern present? (Temperature 105-108°C AND Vibration 3.1-3.8 mm/s AND Ambient 30-34°C) → TRUE ANOMALY
   - Are all parameters within normal range? (Temperature < 95°C AND Vibration < 2.8 mm/s) → FALSE POSITIVE
   - Is there clear environmental correlation? (Ambient > 32°C AND Discharge < 110°C) OR (Humidity > 80% AND Vibration < 3.8 mm/s) → FALSE POSITIVE

2. **If No Override Rule Applies, Use Standard Classification Rules**
   - Does vibration exceed 4.5 mm/s? → True anomaly
   - Does discharge temperature exceed 115°C? → True anomaly
   - Does bearing temperature exceed 95°C? → True anomaly
   - Does discharge pressure exceed 9.0 bar? → True anomaly

3. **Analyze Environmental Correlations**
   - Does the anomaly coincide with ambient temperature >30°C? → Possible false positive
   - Does the anomaly coincide with humidity >85%? → Possible false positive
   - Does the anomaly coincide with atmospheric pressure variations? → Possible false positive

4. **Verify Persistence**
   - Does the anomaly persist for more than 12 hours? → Likely true anomaly
   - Is the anomaly transient (few hours)? → Likely false positive

5. **Evaluate Combined Severity**
   - Are multiple parameters simultaneously approaching critical thresholds? → True anomaly
   - Only isolated small deviations? → Likely false positive

