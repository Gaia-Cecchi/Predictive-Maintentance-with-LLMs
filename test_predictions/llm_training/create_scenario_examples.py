from datetime import datetime
import pandas as pd
import numpy as np
import os
import json
import re
import ast
import random

"""
#######################################################################################
# GENERATORE DI SCENARI PER ADDESTRARE LLM NELLA PREVISIONE DI ANOMALIE
#
# Questo script crea un repository di scenari di anomalie organizzati e documentati
# per addestrare i Large Language Models (LLM) a riconoscere pattern specifici nelle 
# anomalie di compressori industriali. Lo scopo è fornire una knowledge base
# strutturata per il prompt engineering avanzato.
#
# CONCETTO FONDAMENTALE: La tecnica di scenario-based learning implementata qui
# permette ai LLM di acquisire conoscenza specializzata non solo attraverso esempi
# isolati, ma tramite categorie concettuali (scenari) che inquadrano il ragionamento
# esperto in schemi riconoscibili.
#######################################################################################
"""

def load_jsonl_data(file_path):
    """Load data from JSONL file"""
    """
    #######################################################################################
    # FUNZIONE DI CARICAMENTO JSONL
    #
    # Questa funzione carica dati da file JSONL, un formato comune per dataset di LLM.
    # JSONL (JSON Lines) è preferito rispetto a JSON standard perché:
    # 1. Permette di processare il file una riga alla volta (streaming)
    # 2. Supporta dataset di grandi dimensioni che non entrerebbero in memoria
    # 3. Facilita l'aggiunta incrementale di nuovi esempi
    #
    # LIMITAZIONE NOTA: L'implementazione qui carica l'intero file in memoria, 
    # contraddicendo parzialmente i vantaggi di JSONL. In un'implementazione ottimale,
    # useremmo un approccio di streaming, ma è stato sacrificato per semplicità del codice.
    #######################################################################################
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        print(f"Loaded {len(data)} records from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def extract_patterns_from_jsonl(jsonl_files):
    """Extract classification patterns and expert reasoning from JSONL files"""
    """
    #######################################################################################
    # ESTRAZIONE DI PATTERN DA DATI DI TRAINING LLM
    #
    # Questa funzione analizza i dati di addestramento LLM esistenti per estrarre pattern
    # statistici e regole latenti. È una forma di meta-learning: imparare dagli esempi
    # già utilizzati per l'addestramento dei modelli.
    #
    # MOTIVAZIONE TECNICA: Questa funzione implementa un approccio di "bootstrapping"
    # che sfrutta conoscenza già codificata nei dataset di fine-tuning per migliorare
    # la qualità dei nuovi scenari. Questo crea un ciclo virtuoso dove ogni iterazione
    # di training migliora la successiva.
    #
    # LIMITAZIONE CRITICA: Il pattern matching implementato è estremamente semplicistico,
    # basato su regex per estrarre valori numerici. Un approccio più sofisticato
    # richiederebbe NLP avanzato o semantic parsing. Questa scelta pragmatica riflette
    # un trade-off tra complessità implementativa e risultati accettabili.
    #######################################################################################
    """
    all_patterns = {
        "TRUE_POSITIVE": {},
        "FALSE_POSITIVE": {}
    }
    
    parameters = [
        "discharge_temp", "vibration", "discharge_pressure", "suction_pressure",
        "bearing_temp", "motor_speed", "ambient_temperature", "humidity"
    ]
    
    total_examples = 0
    
    for file_path in jsonl_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        data = load_jsonl_data(file_path)
        
        for item in data:
            if "messages" not in item:
                continue
                
            # Extract the relevant data from the conversation
            anomaly_data = None
            classification = None
            reasoning = None
            
            for msg in item["messages"]:
                if msg["role"] == "user":
                    # Extract parameters from user message
                    anomaly_data = msg["content"]
                elif msg["role"] == "assistant":
                    # Extract classification from assistant response
                    response = msg["content"].upper()
                    if "TRUE ANOMALY" in response:
                        classification = "TRUE_POSITIVE"
                        reasoning = msg["content"]
                    elif "FALSE POSITIVE" in response:
                        classification = "FALSE_POSITIVE"
                        reasoning = msg["content"]
            
            if anomaly_data and classification:
                total_examples += 1
                
                # Extract parameter values using regex
                for param in parameters:
                    # Use raw string for regex pattern with proper escaping
                    pattern = r"{}: ?([0-9.]+)".format(param.replace('_', ' '))
                    match = re.search(pattern, anomaly_data, re.IGNORECASE)
                    
                    if match:
                        value = float(match.group(1))
                        
                        # Initialize param stats if not exists
                        if param not in all_patterns[classification]:
                            all_patterns[classification][param] = []
                            
                        all_patterns[classification][param].append(value)
    
    print(f"Extracted patterns from {total_examples} examples")
    
    # Generate statistics
    stats = {}
    for classification, params in all_patterns.items():
        stats[classification] = {}
        for param, values in params.items():
            if values:
                stats[classification][param] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "median": sorted(values)[len(values) // 2],
                    "count": len(values)
                }
    
    return all_patterns, stats

def get_jsonl_classification_rules(stats):
    """Generate classification rules based on JSONL analysis"""
    """
    #######################################################################################
    # DERIVAZIONE AUTOMATICA DI REGOLE DI CLASSIFICAZIONE
    #
    # Questa funzione trasforma statistiche grezze in regole linguistiche comprensibili
    # per l'essere umano e utili per i LLM. È un tentativo di distillare la conoscenza
    # implicita nei dati in conoscenza esplicita nelle regole.
    #
    # NOTA ALGORITMICA: L'implementazione utilizza principalmente statistiche semplici
    # (mediana e massimi) per derivare soglie. Questo approccio è deliberatamente
    # conservativo per evitare overfitting su pattern spuri nei dati.
    #
    # CRITICITÀ FILOSOFICA: Stiamo essenzialmente "inventando" regole dall'analisi
    # statistica, senza validazione esperta. Questa è una forma di "data snooping"
    # che può portare a regole plausibili ma tecnicamente non valide. Il compromesso
    # è accettabile solo perché queste regole verranno ulteriormente filtrate dalla
    # capacità di reasoning del LLM.
    #######################################################################################
    """
    rules = {
        "TRUE_POSITIVE": [],
        "FALSE_POSITIVE": []
    }
    
    # Generate rules for TRUE_POSITIVE
    if "TRUE_POSITIVE" in stats:
        tp = stats["TRUE_POSITIVE"]
        
        if "vibration" in tp:
            if tp["vibration"]["max"] > 4.0:
                rules["TRUE_POSITIVE"].append(f"Vibration > {tp['vibration']['max']:.1f} mm/s is critical and indicates mechanical issues")
            rules["TRUE_POSITIVE"].append(f"Vibration > {tp['vibration']['median']:.1f} mm/s is commonly associated with TRUE anomalies")
        
        if "discharge_temp" in tp:
            rules["TRUE_POSITIVE"].append(f"Discharge temperature > {tp['discharge_temp']['median']:.1f}°C that is not correlated with ambient temperature may indicate a TRUE anomaly")
        
        if "bearing_temp" in tp and tp["bearing_temp"]["max"] > 85:
            rules["TRUE_POSITIVE"].append(f"Bearing temperature > {tp['bearing_temp']['median']:.1f}°C is a common indicator of real mechanical issues")
    
    # Generate rules for FALSE_POSITIVE
    if "FALSE_POSITIVE" in stats:
        fp = stats["FALSE_POSITIVE"]
        
        if "ambient_temperature" in fp and "discharge_temp" in fp:
            rules["FALSE_POSITIVE"].append(f"When ambient temperature is high (>{fp['ambient_temperature']['median']:.1f}°C) and discharge temperature is proportionally elevated, it's often a FALSE POSITIVE")
        
        if "humidity" in fp:
            rules["FALSE_POSITIVE"].append(f"High humidity (>{fp['humidity']['median']:.1f}%) combined with minor parameter deviations often indicates a FALSE POSITIVE")
    
    return rules

def create_scenario_examples(input_csv, output_dir, jsonl_dir=None):
    """
    Group anomalies into representative scenarios for LLM training
    with enhanced patterns from JSONL training data
    """
    """
    #######################################################################################
    # FUNZIONE PRINCIPALE: CREAZIONE DI SCENARI DI ANOMALIE
    #
    # Questa è la funzione centrale che organizza gli esempi di anomalie in scenari
    # coerenti per l'addestramento dei LLM. Ogni scenario rappresenta un pattern
    # tipico di comportamento del compressore, sia per anomalie reali che per falsi allarmi.
    #
    # APPROCCIO METODOLOGICO: La funzione utilizza una combinazione di:
    # 1. Conoscenza esperta hard-coded (regole definite manualmente dagli esperti)
    # 2. Analisi data-driven (pattern estratti automaticamente dai dati di training)
    # 3. Categorizzazione gerarchica (organizzazione in gruppi concettuali)
    #
    # Questa ibridazione di metodi knowledge-based e data-driven è una scelta deliberata
    # per bilanciare la precisione della conoscenza umana con la scalabilità
    # dell'analisi automatica.
    #######################################################################################
    """
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_user = "Gaia-Cecchi"
    
    print(f"Loading data from: {input_csv}")
    print(f"Analysis timestamp: {current_date}")
    print(f"Analysis by: {current_user}")
    
    df = pd.read_csv(input_csv)
    
    # Process JSONL files if provided
    jsonl_patterns = {}
    jsonl_stats = {}
    jsonl_rules = {}
    
    if jsonl_dir and os.path.exists(jsonl_dir):
        print(f"Analyzing JSONL files from: {jsonl_dir}")
        jsonl_files = [
            os.path.join(jsonl_dir, "llm_training_data.jsonl"),
            os.path.join(jsonl_dir, "llm_validation_data.jsonl"),
            os.path.join(jsonl_dir, "llm_test_data.jsonl")
        ]
        
        jsonl_patterns, jsonl_stats = extract_patterns_from_jsonl(jsonl_files)
        jsonl_rules = get_jsonl_classification_rules(jsonl_stats)
    
    # Filter anomalies based on available columns
    if 'anomaly_type' in df.columns:
        anomalies_df = df[(df['anomaly_type'] == 'TRUE_POSITIVE') | 
                          (df['anomaly_type'] == 'FALSE_POSITIVE')].copy()
        print(f"Filtered {len(anomalies_df)} anomalies based on anomaly_type")
    elif 'is_anomaly' in df.columns:
        anomalies_df = df[df['is_anomaly'] == 1].copy()
        print(f"Filtered {len(anomalies_df)} anomalies based on is_anomaly")
    else:
        print("Neither anomaly_type nor is_anomaly column found. Using all data.")
        anomalies_df = df.copy()
    
    """
    #######################################################################################
    # FEATURE ENGINEERING PER ANALISI AVANZATA
    #
    # Questa sezione aggiunge metriche derivate che catturano relazioni complesse
    # tra parametri e sono fondamentali per il corretto reasoning sui dati.
    #
    # Le feature derivate implementate qui codificano conoscenza di dominio specifica
    # sui compressori industriali. Ad esempio:
    # - temp_ambient_delta: differenza tra temperatura di scarico e ambiente, critica per
    #   distinguere surriscaldamenti meccanici da quelli ambientali
    # - pressure_ratio: rapporto tra pressione di scarico e aspirazione, indicativo di
    #   efficienza del compressore e possibili perdite
    #
    # NOTA TECNICA: Alcune di queste feature potrebbero sembrare ridondanti, ma sono
    # deliberatamente incluse perché i LLM non sono bravi a fare calcoli matematici
    # in runtime. È preferibile fornire direttamente i valori precalcolati.
    #######################################################################################
    """
    # Calculate derived columns
    print("Calculating derived metrics...")
    
    # Calculate pressure delta if needed
    if 'atm_pressure_delta' not in anomalies_df.columns and 'atmospheric_pressure' in anomalies_df.columns:
        anomalies_df['date'] = pd.to_datetime(anomalies_df['timestamp']).dt.date if 'timestamp' in anomalies_df.columns else pd.Timestamp.now().date()
        daily_mean = anomalies_df.groupby('date')['atmospheric_pressure'].transform('mean')
        anomalies_df['atm_pressure_delta'] = anomalies_df['atmospheric_pressure'] - daily_mean
    
    # Calculate temperature delta
    anomalies_df['temp_ambient_delta'] = anomalies_df['discharge_temp_true'] - anomalies_df['ambient_temperature']
    
    # Calculate pressure ratio
    anomalies_df['pressure_ratio'] = anomalies_df['discharge_pressure_true'] / np.maximum(anomalies_df['suction_pressure_true'], 0.1)
    
    # Calculate prediction deviations where available
    if 'discharge_temp_pred' in anomalies_df.columns:
        anomalies_df['temp_deviation'] = anomalies_df['discharge_temp_true'] - anomalies_df['discharge_temp_pred']
        anomalies_df['temp_deviation_pct'] = (anomalies_df['temp_deviation'] / np.maximum(anomalies_df['discharge_temp_pred'], 0.1)) * 100
    
    if 'vibration_pred' in anomalies_df.columns:
        anomalies_df['vibration_deviation'] = anomalies_df['vibration_true'] - anomalies_df['vibration_pred']
        anomalies_df['vibration_deviation_pct'] = (anomalies_df['vibration_deviation'] / np.maximum(anomalies_df['vibration_pred'], 0.1)) * 100
    
    # Ensure we have anomaly_duration column
    if 'anomaly_duration' not in anomalies_df.columns:
        anomalies_df['anomaly_duration'] = 1
    
    """
    #######################################################################################
    # DEFINIZIONE DEGLI SCENARI DI ANOMALIA
    #
    # Questa è la sezione più critica del codice, dove viene codificata la conoscenza
    # di dominio specialistica sui pattern di anomalia nei compressori industriali.
    #
    # STRUTTURA DEGLI SCENARI:
    # Ogni scenario è definito da:
    # 1. description: descrizione testuale del pattern per la comprensione umana
    # 2. condition: funzione lambda che implementa la logica di matching su dataframe
    # 3. priority: classificazione principale (TRUE_POSITIVE, FALSE_POSITIVE, BORDERLINE)
    # 4. analysis: spiegazione tecnica dettagliata utilizzata per il meta-training
    #
    # SCELTA IMPLEMENTATIVA CONTROVERSA: Gli scenari sono hard-coded nel codice invece
    # di essere definiti in un file di configurazione esterno. Questo compromette la 
    # manutenibilità ma garantisce un controllo preciso e la possibilità di usare
    # funzioni lambda per la logica di matching complessa.
    #
    # HONESTY NOTE: Questa sezione è eccessivamente verbosa e difficile da mantenere.
    # In un'implementazione produttiva sarebbe preferibile un sistema di definizione 
    # degli scenari esterno al codice, possibilmente in formato YAML o JSON con un DSL
    # per le condizioni logiche.
    #######################################################################################
    """
    # Define comprehensive scenario dictionary
    scenarios = {
        # === TRUE ANOMALY SCENARIOS (MECHANICAL ISSUES) ===
        "critical_vibration": {
            "description": "Anomalies with critical vibration levels (>4.5 mm/s) indicating mechanical issues",
            "condition": lambda df: df['vibration_true'] > 4.5,
            "priority": "TRUE_POSITIVE",
            "analysis": "High vibration levels exceeding 4.5 mm/s indicate mechanical issues such as bearing damage, misalignment, or imbalance. This requires immediate maintenance intervention regardless of other parameters."
        },
        "high_vibration": {
            "description": "Anomalies with high vibration (3.5-4.5 mm/s) suggesting developing mechanical problems",
            "condition": lambda df: (df['vibration_true'] > 3.5) & (df['vibration_true'] <= 4.5),
            "priority": "TRUE_POSITIVE",
            "analysis": "Vibration levels between 3.5-4.5 mm/s indicate developing mechanical problems that require maintenance attention before they worsen to critical levels."
        },
        "critical_discharge_temperature": {
            "description": "Anomalies with critical discharge temperature (>115°C) indicating overheating",
            "condition": lambda df: df['discharge_temp_true'] > 115,
            "priority": "TRUE_POSITIVE",
            "analysis": "Discharge temperature above 115°C indicates severe overheating that can damage components and lubricants. This requires immediate maintenance regardless of ambient conditions."
        },
        "high_discharge_temperature": {
            "description": "Anomalies with high discharge temperature (105-115°C) not related to ambient conditions",
            "condition": lambda df: (df['discharge_temp_true'] > 105) & 
                                   (df['discharge_temp_true'] <= 115) & 
                                   (df['temp_ambient_delta'] > 75),
            "priority": "TRUE_POSITIVE",
            "analysis": "Discharge temperature between 105-115°C with a high differential from ambient temperature indicates internal heating issues requiring maintenance."
        },
        "bearing_temperature_critical": {
            "description": "Anomalies with bearing temperature above critical threshold (>95°C)",
            "condition": lambda df: 'bearing_temp_true' in df.columns and df['bearing_temp_true'] > 95,
            "priority": "TRUE_POSITIVE",
            "analysis": "Bearing temperatures exceeding 95°C indicate bearing damage or lubrication failure. This requires immediate maintenance to prevent catastrophic failure."
        },
        "bearing_temperature_high": {
            "description": "Anomalies with elevated bearing temperature (85-95°C)",
            "condition": lambda df: 'bearing_temp_true' in df.columns and 
                                    (df['bearing_temp_true'] > 85) & (df['bearing_temp_true'] <= 95),
            "priority": "TRUE_POSITIVE",
            "analysis": "Bearing temperatures between 85-95°C indicate developing bearing issues or inadequate lubrication requiring maintenance intervention."
        },
        "abnormal_motor_speed_low": {
            "description": "Anomalies with abnormally low motor speed (<2800 rpm) indicating motor problems",
            "condition": lambda df: 'motor_speed_true' in df.columns and df['motor_speed_true'] < 2800,
            "priority": "TRUE_POSITIVE",
            "analysis": "Motor speeds below 2800 rpm (when nominal is 2950 rpm) indicate motor issues, excessive load, or electrical problems requiring maintenance."
        },
        "abnormal_motor_speed_high": {
            "description": "Anomalies with abnormally high motor speed (>3100 rpm) indicating control issues",
            "condition": lambda df: 'motor_speed_true' in df.columns and df['motor_speed_true'] > 3100,
            "priority": "TRUE_POSITIVE",
            "analysis": "Motor speeds above 3100 rpm (when nominal is 2950 rpm) indicate control system malfunctions or sensor issues requiring maintenance."
        },
        "excessive_pressure_ratio": {
            "description": "Anomalies with dangerously high pressure ratio (>8.0) indicating valve or seal issues",
            "condition": lambda df: df['pressure_ratio'] > 8.0,
            "priority": "TRUE_POSITIVE",
            "analysis": "Pressure ratios above 8.0 indicate valve problems, internal leakage, or seal failures requiring maintenance intervention."
        },
        "high_pressure_ratio": {
            "description": "Anomalies with elevated pressure ratio (7.0-8.0) suggesting developing issues",
            "condition": lambda df: (df['pressure_ratio'] > 7.0) & (df['pressure_ratio'] <= 8.0),
            "priority": "TRUE_POSITIVE",
            "analysis": "Pressure ratios between 7.0-8.0 indicate developing valve or seal issues that should be addressed during planned maintenance."
        },
        "insufficient_discharge_pressure": {
            "description": "Anomalies with insufficient discharge pressure (<5.5 bar) indicating leaks or valve issues",
            "condition": lambda df: df['discharge_pressure_true'] < 5.5,
            "priority": "TRUE_POSITIVE",
            "analysis": "Discharge pressure below 5.5 bar indicates leaks, faulty valves, or intake restrictions requiring maintenance."
        },

        # === COMBINED MECHANICAL ISSUE SCENARIOS ===
        "moderate_vibration_with_high_temperature": {
            "description": "Anomalies with both elevated vibration (3.0-4.5 mm/s) AND high temperature (95-115°C)",
            "condition": lambda df: (df['vibration_true'] >= 3.0) & 
                                   (df['vibration_true'] <= 4.5) & 
                                   (df['discharge_temp_true'] > 95) & 
                                   (df['discharge_temp_true'] <= 115),
            "priority": "TRUE_POSITIVE",
            "analysis": "The combination of elevated vibration and high temperature indicates developing mechanical issues that require maintenance before they worsen."
        },
        "persistent_moderate_anomalies": {
            "description": "Anomalies with multiple moderate parameter deviations persisting over time",
            "condition": lambda df: ((df['vibration_true'] > 2.8) | 
                                    (df['discharge_temp_true'] > 95) | 
                                    (df['discharge_pressure_true'] > 7.5)) &
                                    (df['anomaly_duration'] >= 3),
            "priority": "TRUE_POSITIVE",
            "analysis": "Multiple parameters showing moderate deviations over an extended period indicate genuine mechanical issues requiring maintenance."
        },
        "high_vibration_normal_temperature": {
            "description": "Anomalies with high vibration (>3.5 mm/s) despite normal temperature (70-95°C)",
            "condition": lambda df: (df['vibration_true'] > 3.5) & 
                                   (df['discharge_temp_true'] >= 70) & 
                                   (df['discharge_temp_true'] <= 95),
            "priority": "TRUE_POSITIVE",
            "analysis": "High vibration with normal temperature typically indicates mechanical issues like bearing wear, misalignment, or loose components requiring maintenance."
        },

        # === FALSE POSITIVE SCENARIOS (ENVIRONMENTAL OR TRANSIENT) ===
        "high_ambient_temperature_effect": {
            "description": "Elevated discharge temperature correlated with high ambient temperature (>35°C)",
            "condition": lambda df: (df['ambient_temperature'] > 35) & 
                                   (df['discharge_temp_true'] > 95) & 
                                   (df['discharge_temp_true'] <= 115) &
                                   (df['temp_ambient_delta'] < 75),
            "priority": "FALSE_POSITIVE",
            "analysis": "Elevated discharge temperature that rises proportionally with ambient temperature is typically a normal operating condition in hot environments and not a mechanical issue."
        },
        "high_humidity_effect": {
            "description": "Parameter variations during very high humidity conditions (>85%)",
            "condition": lambda df: (df['humidity'] > 85) & 
                                   ((df['discharge_temp_true'] > 95) | 
                                    (df['vibration_true'] > 2.8)),
            "priority": "FALSE_POSITIVE",
            "analysis": "Parameter variations during high humidity conditions are often due to reduced cooling efficiency in humid environments rather than mechanical issues."
        },
        "transient_temperature_spike": {
            "description": "Short-duration temperature spikes with quick return to normal range",
            "condition": lambda df: (df['discharge_temp_true'] > 95) & 
                                    (df['discharge_temp_true'] <= 115) &
                                    (df['anomaly_duration'] <= 2),
            "priority": "FALSE_POSITIVE",
            "analysis": "Brief temperature spikes that quickly normalize are typically due to temporary load changes or environmental factors, not mechanical issues requiring maintenance."
        },
        "atmospheric_pressure_fluctuation": {
            "description": "Suction or discharge pressure variations correlated with atmospheric pressure changes",
            "condition": lambda df: ('atm_pressure_delta' in df.columns) & 
                                   (df['atm_pressure_delta'].abs() > 5) & 
                                   ((df['discharge_pressure_true'] > 7.5) | 
                                    (df['discharge_pressure_true'] < 6.0)),
            "priority": "FALSE_POSITIVE",
            "analysis": "Pressure variations that correlate with atmospheric pressure changes are natural adaptations of the system to environmental conditions, not mechanical issues."
        },
        "minor_parameter_deviations": {
            "description": "Small deviations across multiple parameters within acceptable ranges",
            "condition": lambda df: ((df['discharge_temp_true'] > 85) & (df['discharge_temp_true'] <= 95)) &
                                    ((df['vibration_true'] > 2.0) & (df['vibration_true'] <= 2.8)) &
                                    ((df['discharge_pressure_true'] > 7.0) & (df['discharge_pressure_true'] <= 7.5)),
            "priority": "FALSE_POSITIVE",
            "analysis": "Small simultaneous deviations that remain within acceptable ranges are part of normal operational variations and don't indicate mechanical problems."
        },
        "normal_startup_conditions": {
            "description": "Parameter variations typical during compressor startup or load changes",
            "condition": lambda df: ('motor_speed_true' in df.columns) & 
                                   (((df['motor_speed_true'] < 2900) & (df['motor_speed_true'] >= 2800)) | 
                                    ((df['motor_speed_true'] > 3000) & (df['motor_speed_true'] <= 3100))) &
                                   ((df['discharge_temp_true'] > 90) & (df['discharge_temp_true'] <= 105)),
            "priority": "FALSE_POSITIVE",
            "analysis": "Parameter variations during startup or load changes are normal transient conditions and not indicative of mechanical issues requiring maintenance."
        },
        "moderate_humidity_moderate_temperature": {
            "description": "Moderate discharge temperature increases during moderate humidity (60-85%)",
            "condition": lambda df: (df['humidity'] > 60) & (df['humidity'] <= 85) &
                                   (df['discharge_temp_true'] > 90) & (df['discharge_temp_true'] <= 100),
            "priority": "FALSE_POSITIVE",
            "analysis": "Moderate temperature increases during periods of moderate humidity are typically due to reduced cooling efficiency rather than mechanical issues."
        },

        # === BOUNDARY CASE SCENARIOS (REQUIRE CAREFUL ANALYSIS) ===
        "boundary_discharge_temperature": {
            "description": "Borderline high discharge temperature (95-105°C) requiring contextual analysis",
            "condition": lambda df: (df['discharge_temp_true'] > 95) & 
                                   (df['discharge_temp_true'] <= 105) & 
                                   (df['vibration_true'] <= 3.0),
            "priority": "BORDERLINE",
            "analysis": "Temperatures in the 95-105°C range with normal vibration require contextual analysis. If correlated with ambient temperature or transient, it's likely a FALSE_POSITIVE; if persistent or unrelated to ambient conditions, it may be a TRUE_POSITIVE."
        },
        "boundary_vibration_levels": {
            "description": "Borderline vibration levels (2.8-3.5 mm/s) requiring contextual analysis",
            "condition": lambda df: (df['vibration_true'] > 2.8) & 
                                   (df['vibration_true'] <= 3.5) & 
                                   (df['discharge_temp_true'] <= 95),
            "priority": "BORDERLINE",
            "analysis": "Vibration in the 2.8-3.5 mm/s range with normal temperature requires contextual analysis. If persistent or increasing over time, it's likely a TRUE_POSITIVE; if transient or correlated with specific operating conditions, it may be a FALSE_POSITIVE."
        },
        "borderline_pressure_variations": {
            "description": "Discharge pressure slightly outside normal range (7.5-8.0 bar) requiring analysis",
            "condition": lambda df: (df['discharge_pressure_true'] > 7.5) & 
                                   (df['discharge_pressure_true'] <= 8.0),
            "priority": "BORDERLINE",
            "analysis": "Pressures in the 7.5-8.0 bar range require contextual analysis. If correlated with load changes or atmospheric pressure variations, it's likely a FALSE_POSITIVE; if persistent or accompanied by other anomalies, it may be a TRUE_POSITIVE."
        }
    }
    
    # Add scenarios based on JSONL analysis if available
    if jsonl_rules:
        print("Adding scenarios based on JSONL analysis...")
        
        """
        #######################################################################################
        # INTEGRAZIONE DI SCENARI DA ANALISI JSONL
        #
        # Questa sezione estende gli scenari predefiniti con quelli derivati da analisi
        # automatica dei dati di training. Rappresenta un approccio "ibrido" che combina
        # conoscenza esperta (scenari predefiniti) con conoscenza derivata dai dati.
        #
        # PROBLEMATICA METODOLOGICA: Gli scenari derivati automaticamente hanno un problema
        # fondamentale - la condizione lambda è impostata sempre a "True". Questo è un
        # compromesso tecnico che riflette la difficoltà di convertire regole testuali
        # in funzioni di filtro programmatiche senza un parser complesso. La conseguenza
        # è che questi scenari non possono essere utilizzati per filtrare automaticamente
        # gli esempi, ma solo come documentazione di pattern ricorrenti.
        #
        # NOTA CRITICA: Questa funzionalità è un esempio di intent ambiziosa ma
        # implementazione parziale. In un sistema di produzione, sarebbe necessario un
        # parser che converta le regole testuali in funzioni lambda effettive.
        #######################################################################################
        """
        
        # Add TRUE_POSITIVE scenarios from JSONL
        for i, rule in enumerate(jsonl_rules.get("TRUE_POSITIVE", []), 1):
            scenario_name = f"jsonl_tp_scenario_{i}"
            scenarios[scenario_name] = {
                "description": rule,
                "condition": lambda df: True,  # Will manually filter later
                "priority": "TRUE_POSITIVE",
                "analysis": rule
            }
        
        # Add FALSE_POSITIVE scenarios from JSONL
        for i, rule in enumerate(jsonl_rules.get("FALSE_POSITIVE", []), 1):
            scenario_name = f"jsonl_fp_scenario_{i}"
            scenarios[scenario_name] = {
                "description": rule,
                "condition": lambda df: True,  # Will manually filter later
                "priority": "FALSE_POSITIVE",
                "analysis": rule
            }
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract examples for each scenario
    scenario_examples = {}
    
    for scenario_name, scenario_info in scenarios.items():
        # Filter anomalies for this scenario
        try:
            """
            #######################################################################################
            # ESTRAZIONE DI ESEMPI RAPPRESENTATIVI PER OGNI SCENARIO
            #
            # Questa sezione applica le funzioni di condizione degli scenari per filtrare il dataset
            # e trovare esempi rappresentativi. Per ciascuno scenario, vengono selezionati fino a 
            # 3 esempi di TRUE_POSITIVE e 3 di FALSE_POSITIVE.
            #
            # MOTIVAZIONE DELLA LIMITAZIONE: La scelta di limitare a 3 esempi per tipo è un 
            # compromesso tra:
            # 1. Fornire sufficienti esempi per comprendere il pattern
            # 2. Evitare di sovraccaricare il documento markdown finale
            # 3. Ridurre la ridondanza, dato che molti scenari potrebbero avere esempi simili
            #
            # SEED FISSO: Anche qui utilizziamo random_state=42 per garantire riproducibilità,
            # essenziale in un contesto di addestramento di modelli.
            #######################################################################################
            """
            # Apply the condition function to the dataframe
            filtered_df = anomalies_df[scenario_info["condition"](anomalies_df)]
            
            # Count TRUE_POSITIVE and FALSE_POSITIVE examples
            true_examples = []
            false_examples = []
            
            if 'anomaly_type' in filtered_df.columns:
                tp_df = filtered_df[filtered_df['anomaly_type'] == 'TRUE_POSITIVE']
                fp_df = filtered_df[filtered_df['anomaly_type'] == 'FALSE_POSITIVE']
                
                # Sample up to 3 examples of each type if available
                if len(tp_df) > 0:
                    true_examples = tp_df.sample(min(3, len(tp_df)), random_state=42).to_dict('records')
                if len(fp_df) > 0:
                    false_examples = fp_df.sample(min(3, len(fp_df)), random_state=42).to_dict('records')
            
            scenario_examples[scenario_name] = {
                "description": scenario_info["description"],
                "priority": scenario_info["priority"],
                "analysis": scenario_info["analysis"],
                "count_tp": len(true_examples),
                "count_fp": len(false_examples),
                "tp_examples": true_examples,
                "fp_examples": false_examples
            }
            
            print(f"Scenario '{scenario_name}': {len(filtered_df)} matches, {len(true_examples)} TRUE_POSITIVE, {len(false_examples)} FALSE_POSITIVE")
            
        except Exception as e:
            print(f"Error processing scenario '{scenario_name}': {e}")
    
    # Generate a markdown file with examples organized by scenario
    output_md = os.path.join(output_dir, "anomaly_scenarios.md")
    
    """
    #######################################################################################
    # GENERAZIONE DEL DOCUMENTO MARKDOWN DEGLI SCENARI
    #
    # Questa sezione genera un documento markdown strutturato che organizza gli scenari
    # e i relativi esempi in modo gerarchico. La struttura scelta non è casuale, ma 
    # riflette un'organizzazione cognitiva che facilita l'apprendimento del LLM:
    #
    # 1. Categorie principali (TRUE_POSITIVE, FALSE_POSITIVE, BORDERLINE)
    # 2. Scenari specifici all'interno di ogni categoria
    # 3. Esempi concreti per ogni scenario con parametri tecnici dettagliati
    # 4. Analisi tecnica esplicita che spiega il ragionamento per ogni esempio
    #
    # NOTA TECNICA: La codifica markdown è stata progettata specificamente per rendere
    # la struttura gerarchica chiara non solo visivamente ma anche semanticamente,
    # utilizzando opportunamente intestazioni di livello h2, h3, h4 per distinguere
    # categorie, scenari ed esempi.
    #######################################################################################
    """
    
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Anomaly Scenarios for Industrial Compressors\n\n")
        f.write("This document contains representative examples of anomalies organized by scenario, distinguishing between true anomalies and false positives.\n\n")
        f.write(f"**Analysis Date:** {current_date}\n")
        f.write(f"**Created By:** {current_user}\n\n")
        f.write("## Understanding the Scenarios\n\n")
        f.write("Each scenario includes comprehensive examples of both TRUE ANOMALY (requiring maintenance) and FALSE POSITIVE (transitory/environmental) cases to help distinguish between them accurately.\n\n")
        
        # Group scenarios by priority for better organization
        priority_groups = {
            "TRUE_POSITIVE": "Critical Mechanical Issues (Primarily TRUE ANOMALY)",
            "FALSE_POSITIVE": "Environmental and Transient Conditions (Primarily FALSE POSITIVE)",
            "BORDERLINE": "Boundary Cases Requiring Careful Analysis",
            "BALANCED": "General Scenarios"
        }
                
        # Write each scenario group to the markdown file
        for priority, heading in priority_groups.items():
            # Get scenarios with this priority
            matching_scenarios = {k: v for k, v in scenario_examples.items() 
                                if v["priority"] == priority and (v["count_tp"] > 0 or v["count_fp"] > 0)}
            
            if not matching_scenarios:
                continue
                
            f.write(f"## {heading}\n\n")
            
            # Write each scenario in this priority group
            for scenario_name, scenario_data in matching_scenarios.items():
                f.write(f"### Scenario: {scenario_name.replace('_', ' ').title()}\n\n")
                f.write(f"*{scenario_data['description']}*\n\n")
                
                # Include the analytical guidance for this scenario
                f.write("**Analysis Guidance:**\n")
                f.write(f"{scenario_data['analysis']}\n\n")
                
                f.write(f"Found {scenario_data['count_tp']} examples of true anomalies and {scenario_data['count_fp']} false positives.\n\n")
                
                """
                #######################################################################################
                # FORMATTAZIONE DEGLI ESEMPI SPECIFICI
                #
                # Questa sezione formatta dettagliatamente i singoli esempi di anomalie con
                # una struttura che rispecchia il modo in cui il LLM interagisce con i dati.
                # La scelta di includere sia i parametri tecnici che le condizioni ambientali
                # è fondamentale per permettere al modello di apprendere le correlazioni
                # corrette tra ambiente e anomalie.
                #
                # SCELTA TECNICA: La notazione ``` per delimitare gli esempi è stata scelta 
                # specificamente perché nei markdowns è usata per delimitare blocchi di codice.
                # Questo aiuta il LLM a distinguere nettamente tra testo descrittivo e valori
                # tecnici parametrici, facilitando l'estrazione di informazioni strutturate.
                #######################################################################################
                """
                
                if scenario_data['count_tp'] > 0:
                    f.write("#### True Anomalies\n\n")
                    
                    for i, example in enumerate(scenario_data['tp_examples']):
                        f.write(f"**Example {i+1}:**\n")
                        f.write(f"Compressor parameters:\n")
                        f.write(f"- Discharge Temperature: {example['discharge_temp_true']:.1f}°C")
                        if 'discharge_temp_pred' in example:
                            f.write(f" (expected: {example['discharge_temp_pred']:.1f}°C)")
                            if 'temp_deviation' in example:
                                f.write(f", deviation: {example['temp_deviation']:.1f}°C ({example['temp_deviation_pct']:.1f}%)")
                        f.write("\n")
                        
                        f.write(f"- Discharge Pressure: {example['discharge_pressure_true']:.2f} bar")
                        if 'discharge_pressure_pred' in example:
                            f.write(f" (expected: {example['discharge_pressure_pred']:.2f} bar)")
                        f.write("\n")
                        
                        f.write(f"- Vibration: {example['vibration_true']:.2f} mm/s")
                        if 'vibration_pred' in example:
                            f.write(f" (expected: {example['vibration_pred']:.2f} mm/s)")
                            if 'vibration_deviation' in example:
                                f.write(f", deviation: {example['vibration_deviation']:.2f} mm/s ({example['vibration_deviation_pct']:.1f}%)")
                        f.write("\n")
                        
                        if 'bearing_temp_true' in example:
                            f.write(f"- Bearing Temperature: {example['bearing_temp_true']:.1f}°C\n")
                        
                        if 'motor_speed_true' in example:
                            f.write(f"- Motor Speed: {example['motor_speed_true']:.0f} rpm\n")
                        
                        f.write(f"- Pressure Ratio: {example['pressure_ratio']:.2f}\n") if 'pressure_ratio' in example else None
                        
                        f.write(f"\nEnvironmental conditions:\n")
                        f.write(f"- Ambient Temperature: {example['ambient_temperature']:.1f}°C\n")
                        f.write(f"- Temperature-Ambient Delta: {example['temp_ambient_delta']:.1f}°C\n") if 'temp_ambient_delta' in example else None
                        f.write(f"- Humidity: {example['humidity']:.0f}%\n")
                        
                        if 'atmospheric_pressure' in example:
                            f.write(f"- Atmospheric Pressure: {example['atmospheric_pressure']:.0f} hPa\n")
                            if 'atm_pressure_delta' in example:
                                f.write(f"- Atmospheric Pressure Delta: {example['atm_pressure_delta']:.1f} hPa\n")
                        
                        f.write(f"- Anomaly Duration: {example['anomaly_duration']} hours\n") if 'anomaly_duration' in example else None
                        
                        f.write(f"\nTechnical analysis:\n")
                        if 'anomaly_description' in example and not pd.isna(example['anomaly_description']):
                            f.write(f"{example['anomaly_description']}\n")
                        else:
                            f.write("This is a TRUE ANOMALY requiring maintenance. Key indicators show mechanical issues that require intervention rather than environmental or transient effects.\n")
                        f.write("```\n\n")
                
                if scenario_data['count_fp'] > 0:
                    f.write("#### False Positives\n\n")
                    
                    for i, example in enumerate(scenario_data['fp_examples']):
                        f.write(f"**Example {i+1}:**\n")
                        f.write(f"Compressor parameters:\n")
                        f.write(f"- Discharge Temperature: {example['discharge_temp_true']:.1f}°C")
                        if 'discharge_temp_pred' in example:
                            f.write(f" (expected: {example['discharge_temp_pred']:.1f}°C)")
                            if 'temp_deviation' in example:
                                f.write(f", deviation: {example['temp_deviation']:.1f}°C ({example['temp_deviation_pct']:.1f}%)")
                        f.write("\n")
                        
                        f.write(f"- Discharge Pressure: {example['discharge_pressure_true']:.2f} bar")
                        if 'discharge_pressure_pred' in example:
                            f.write(f" (expected: {example['discharge_pressure_pred']:.2f} bar)")
                        f.write("\n")
                        
                        f.write(f"- Vibration: {example['vibration_true']:.2f} mm/s")
                        if 'vibration_pred' in example:
                            f.write(f" (expected: {example['vibration_pred']:.2f} mm/s)")
                            if 'vibration_deviation' in example:
                                f.write(f", deviation: {example['vibration_deviation']:.2f} mm/s ({example['vibration_deviation_pct']:.1f}%)")
                        f.write("\n")
                        
                        if 'bearing_temp_true' in example:
                            f.write(f"- Bearing Temperature: {example['bearing_temp_true']:.1f}°C\n")
                        
                        if 'motor_speed_true' in example:
                            f.write(f"- Motor Speed: {example['motor_speed_true']:.0f} rpm\n")
                        
                        f.write(f"- Pressure Ratio: {example['pressure_ratio']:.2f}\n") if 'pressure_ratio' in example else None
                        
                        f.write(f"\nEnvironmental conditions:\n")
                        f.write(f"- Ambient Temperature: {example['ambient_temperature']:.1f}°C\n")
                        f.write(f"- Temperature-Ambient Delta: {example['temp_ambient_delta']:.1f}°C\n") if 'temp_ambient_delta' in example else None
                        f.write(f"- Humidity: {example['humidity']:.0f}%\n")
                        
                        if 'atmospheric_pressure' in example:
                            f.write(f"- Atmospheric Pressure: {example['atmospheric_pressure']:.0f} hPa\n")
                            if 'atm_pressure_delta' in example:
                                f.write(f"- Atmospheric Pressure Delta: {example['atm_pressure_delta']:.1f} hPa\n")
                        
                        f.write(f"- Anomaly Duration: {example['anomaly_duration']} hours\n") if 'anomaly_duration' in example else None
                        
                        f.write(f"\nTechnical analysis:\n")
                        if 'anomaly_description' in example and not pd.isna(example['anomaly_description']):
                            f.write(f"{example['anomaly_description']}\n")
                        else:
                            f.write("This is a FALSE POSITIVE that does not require maintenance. The parameter deviations are caused by environmental factors or transient conditions rather than actual mechanical issues.\n")
                        f.write("```\n\n")
                
                f.write("\n---\n\n")
    
    print(f"Enhanced scenario examples saved to: {output_md}")
    
    """
    #######################################################################################
    # GENERAZIONE DELLA GUIDA DI CLASSIFICAZIONE
    #
    # Oltre al documento principale degli scenari, viene generata una guida di riferimento
    # rapido che distilla le regole di classificazione in un formato più sintetico e
    # immediatamente applicabile. Questa è una scelta deliberata per servire due scopi:
    #
    # 1. Documento scenari: per apprendimento profondo da esempi reali (case-based reasoning)
    # 2. Guida di classificazione: per reference rapido di regole esplicite (rule-based reasoning)
    #
    # NOTA IMPLEMENTATIVA: La struttura tabulare della guida è progettata per facilitare
    # l'estrazione rapida di soglie numeriche e regole decisionali. L'uso di tabelle 
    # markdown è particolarmente efficace con i LLM perché organizza visivamente
    # l'informazione in modo che sia facile da elaborare e riutilizzare.
    #######################################################################################
    """
    
    # Create a classification guide based on scenarios and JSONL analysis
    guide_md = os.path.join(output_dir, "scenario_classification_guide.md")
    
    with open(guide_md, 'w', encoding='utf-8') as f:
        f.write("# Quick Reference Guide for Anomaly Classification\n\n")
        f.write("This guide provides quick reference rules for classifying compressor anomalies.\n\n")
        f.write(f"**Analysis Date:** {current_date}\n")
        f.write(f"**Created By:** {current_user}\n\n")
        
        # TRUE ANOMALY indicators
        f.write("## TRUE ANOMALY Indicators (Requires Maintenance)\n\n")
        f.write("Parameters that strongly indicate mechanical issues requiring intervention:\n\n")
        f.write("| Parameter | Threshold | Interpretation |\n")
        f.write("|-----------|-----------|---------------|\n")
        f.write("| Vibration | > 4.5 mm/s | Critical - Indicates bearing damage or severe mechanical issues |\n")
        f.write("| Vibration | 3.5-4.5 mm/s | High - Likely indicates developing mechanical problems |\n")
        f.write("| Discharge Temperature | > 115°C | Critical - Indicates severe overheating and lubrication issues |\n")
        f.write("| Bearing Temperature | > 95°C | Critical - Indicates bearing damage or lubrication failure |\n")
        f.write("| Pressure Ratio | > 8.0 | Critical - Indicates valve issues or internal leakage |\n")
        f.write("| Discharge Pressure | < 5.5 bar | Low - Indicates leaks or valve issues |\n")
        f.write("| Motor Speed | < 2800 rpm | Low - Indicates motor or control system problems |\n")
        f.write("| Motor Speed | > 3100 rpm | High - Indicates control system malfunction |\n\n")
        
        # FALSE POSITIVE indicators
        f.write("## FALSE POSITIVE Indicators (No Maintenance Required)\n\n")
        f.write("Conditions that typically indicate environmental or transient effects:\n\n")
        f.write("| Parameter Combination | Interpretation |\n")
        f.write("|----------------------|----------------|\n")
        f.write("| High discharge temperature (95-115°C) WITH high ambient temperature (>35°C) | Environmental effect - Temperature difference remains normal |\n")
        f.write("| Parameter deviations DURING high humidity (>85%) | Environmental effect on cooling efficiency |\n")
        f.write("| Brief temperature spikes that normalize quickly (<2 hours) | Transient condition due to load changes |\n")
        f.write("| Pressure variations correlated with atmospheric pressure changes | Normal adaptation to environmental conditions |\n")
        f.write("| Multiple small deviations within warning ranges | Normal operational variations |\n")
        f.write("| Parameter fluctuations during startup or load changes | Expected transient conditions |\n\n")
        
        # Decision tree
        f.write("## Decision Tree for Ambiguous Cases\n\n")
        f.write("When parameters fall in uncertain ranges, use this hierarchical assessment:\n\n")
        f.write("1. **Check Critical Thresholds**\n")
        f.write("   - If ANY parameter exceeds critical limits → TRUE ANOMALY\n")
        f.write("   - If vibration > 4.5 mm/s OR discharge temp > 115°C OR bearing temp > 95°C → TRUE ANOMALY\n\n")
        f.write("2. **Check Environmental Correlation**\n")
        f.write("   - If discharge temp rise proportional to ambient temp → FALSE POSITIVE\n")
        f.write("   - If pressure variations match atmospheric pressure changes → FALSE POSITIVE\n\n")
        f.write("3. **Check Persistence**\n")
        f.write("   - If anomaly persists >3 hours without environmental correlation → TRUE ANOMALY\n")
        f.write("   - If anomaly appears and resolves quickly (<2 hours) → FALSE POSITIVE\n\n")
        f.write("4. **Check Multiple Parameters**\n")
        f.write("   - If multiple parameters show moderate deviations simultaneously → TRUE ANOMALY\n")
        f.write("   - If only one parameter deviates moderately with others normal → Likely FALSE POSITIVE\n\n")
        
        # Add specific failure modes
        f.write("## Specific CSD 102-8 Failure Modes\n\n")
        f.write("| Failure Mode | Observable Parameter Pattern | Classification |\n")
        f.write("|--------------|------------------------------|----------------|\n")
        f.write("| Bearing wear | Gradual vibration increase, possible bearing temp rise | TRUE ANOMALY |\n")
        f.write("| Valve leakage | Low discharge pressure, elevated discharge temp | TRUE ANOMALY |\n")
        f.write("| Oil separator clogging | High pressure differential, high discharge temp | TRUE ANOMALY |\n")
        f.write("| Intake filter clogging | Low suction pressure, high pressure ratio | TRUE ANOMALY |\n")
        f.write("| Cooling system issue | High discharge temp without high ambient temp | TRUE ANOMALY |\n")
        f.write("| Transient load spike | Brief parameter excursion with quick normalization | FALSE POSITIVE |\n")
        f.write("| Seasonal temperature effect | Discharge temp proportional to ambient | FALSE POSITIVE |\n\n")
        
        # Add JSONL-derived rules if available
        if jsonl_rules and (jsonl_rules.get("TRUE_POSITIVE") or jsonl_rules.get("FALSE_POSITIVE")):
            f.write("## Rules Derived from Training Data Analysis\n\n")
            
            if jsonl_rules.get("TRUE_POSITIVE"):
                f.write("### TRUE ANOMALY Rules:\n")
                for rule in jsonl_rules["TRUE_POSITIVE"]:
                    f.write(f"- {rule}\n")
                f.write("\n")
            
            if jsonl_rules.get("FALSE_POSITIVE"):
                f.write("### FALSE POSITIVE Rules:\n")
                for rule in jsonl_rules["FALSE_POSITIVE"]:
                    f.write(f"- {rule}\n")
                f.write("\n")
        
        f.write(f"\nAnalysis by: {current_user}\n")
        f.write(f"Date: {current_date}\n")
    
    print(f"Classification guide saved to: {guide_md}")
    
    return scenario_examples

if __name__ == "__main__":
    """
    #######################################################################################
    # PUNTO DI INGRESSO PRINCIPALE E CONFIGURAZIONE
    #
    # Questa sezione configura i percorsi di file e avvia l'esecuzione dello script.
    # I percorsi hardcoded sono una semplificazione che rende lo script meno portabile
    # ma più semplice da eseguire in un ambiente di sviluppo conosciuto.
    #
    # LIMITAZIONE DI PORTABILITÀ: L'uso di percorsi Windows assoluti con doppio backslash
    # rende lo script difficile da utilizzare su altri sistemi operativi o configurazioni.
    # In un'implementazione robusta, questi percorsi dovrebbero essere configurabili
    # tramite parametri CLI o file di configurazione.
    #
    # NOTA: L'uso esplicito di exit(1) per gli errori di caricamento del file garantisce
    # che il processo termini con un codice di errore che può essere rilevato da
    # eventuali script o sistemi che chiamano questo modulo.
    #######################################################################################
    """
    # Dataset paths
    input_file = "C:\\Users\\gaia1\\Desktop\\UDOO Lab\\Manutenzione predittiva e LLM\\code 10 - validation\\test_predictions\\dataset\\compressor_monitoring_dataset.csv"
    jsonl_dir = "C:\\Users\\gaia1\\Desktop\\UDOO Lab\\Manutenzione predittiva e LLM\\code 10 - validation\\test_predictions\\llm_training\\llm_dataset"
    
    # Save scenarios to the llm_training directory
    output_dir = "C:\\Users\\gaia1\\Desktop\\UDOO Lab\\Manutenzione predittiva e LLM\\code 10 - validation\\test_predictions\\llm_training\\llm_scenarios"
    
    # Verify that the input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("Absolute path:", os.path.abspath(input_file))
        exit(1)
    
    # Create enhanced scenarios incorporating JSONL data
    create_scenario_examples(input_file, output_dir, jsonl_dir)
    print("Scenario generation complete!")