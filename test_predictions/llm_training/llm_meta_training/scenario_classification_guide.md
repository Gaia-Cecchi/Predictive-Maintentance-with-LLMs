# Quick Reference Guide for Anomaly Classification

This guide provides quick reference rules for classifying compressor anomalies.

**Analysis Date:** 2025-03-10 15:58:14
**Created By:** Gaia-Cecchi

## TRUE ANOMALY Indicators (Requires Maintenance)

Parameters that strongly indicate mechanical issues requiring intervention:

| Parameter | Threshold | Interpretation |
|-----------|-----------|---------------|
| Vibration | > 4.5 mm/s | Critical - Indicates bearing damage or severe mechanical issues |
| Vibration | 3.5-4.5 mm/s | High - Likely indicates developing mechanical problems |
| Discharge Temperature | > 115°C | Critical - Indicates severe overheating and lubrication issues |
| Bearing Temperature | > 95°C | Critical - Indicates bearing damage or lubrication failure |
| Pressure Ratio | > 8.0 | Critical - Indicates valve issues or internal leakage |
| Discharge Pressure | < 5.5 bar | Low - Indicates leaks or valve issues |
| Motor Speed | < 2800 rpm | Low - Indicates motor or control system problems |
| Motor Speed | > 3100 rpm | High - Indicates control system malfunction |

## FALSE POSITIVE Indicators (No Maintenance Required)

Conditions that typically indicate environmental or transient effects:

| Parameter Combination | Interpretation |
|----------------------|----------------|
| High discharge temperature (95-115°C) WITH high ambient temperature (>35°C) | Environmental effect - Temperature difference remains normal |
| Parameter deviations DURING high humidity (>85%) | Environmental effect on cooling efficiency |
| Brief temperature spikes that normalize quickly (<2 hours) | Transient condition due to load changes |
| Pressure variations correlated with atmospheric pressure changes | Normal adaptation to environmental conditions |
| Multiple small deviations within warning ranges | Normal operational variations |
| Parameter fluctuations during startup or load changes | Expected transient conditions |

## Decision Tree for Ambiguous Cases

When parameters fall in uncertain ranges, use this hierarchical assessment:

1. **Check Critical Thresholds**
   - If ANY parameter exceeds critical limits → TRUE ANOMALY
   - If vibration > 4.5 mm/s OR discharge temp > 115°C OR bearing temp > 95°C → TRUE ANOMALY

2. **Check Environmental Correlation**
   - If discharge temp rise proportional to ambient temp → FALSE POSITIVE
   - If pressure variations match atmospheric pressure changes → FALSE POSITIVE

3. **Check Persistence**
   - If anomaly persists >3 hours without environmental correlation → TRUE ANOMALY
   - If anomaly appears and resolves quickly (<2 hours) → FALSE POSITIVE

4. **Check Multiple Parameters**
   - If multiple parameters show moderate deviations simultaneously → TRUE ANOMALY
   - If only one parameter deviates moderately with others normal → Likely FALSE POSITIVE

## Specific CSD 102-8 Failure Modes

| Failure Mode | Observable Parameter Pattern | Classification |
|--------------|------------------------------|----------------|
| Bearing wear | Gradual vibration increase, possible bearing temp rise | TRUE ANOMALY |
| Valve leakage | Low discharge pressure, elevated discharge temp | TRUE ANOMALY |
| Oil separator clogging | High pressure differential, high discharge temp | TRUE ANOMALY |
| Intake filter clogging | Low suction pressure, high pressure ratio | TRUE ANOMALY |
| Cooling system issue | High discharge temp without high ambient temp | TRUE ANOMALY |
| Transient load spike | Brief parameter excursion with quick normalization | FALSE POSITIVE |
| Seasonal temperature effect | Discharge temp proportional to ambient | FALSE POSITIVE |

## Rules Derived from Training Data Analysis

### TRUE ANOMALY Rules:
- Vibration > 10.0 mm/s is critical and indicates mechanical issues
- Vibration > 3.7 mm/s is commonly associated with TRUE anomalies

### FALSE POSITIVE Rules:
- High humidity (>32.1%) combined with minor parameter deviations often indicates a FALSE POSITIVE


Analysis by: Gaia-Cecchi
Date: 2025-03-10 15:58:14
