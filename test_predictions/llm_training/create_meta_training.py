import pandas as pd
import os

"""
#######################################################################################
# GENERATORE DI DOCUMENTI META-TRAINING PER LLM IN MANUTENZIONE PREDITTIVA
#
# Questo script genera un documento di meta-training strutturato in markdown che serve 
# come conoscenza di base per guidare il ragionamento del LLM (Large Language Model) 
# nell'analisi di anomalie di compressori industriali.
#
# CONCETTO CHIAVE: Il meta-training è una tecnica di prompt engineering avanzata che
# incorpora direttamente nel prompt la conoscenza di dominio, regole decisionali,
# terminologia tecnica e casi esemplari. Diversamente dal fine-tuning tradizionale,
# questa tecnica non modifica i pesi del modello ma ne guida il ragionamento tramite
# esempi e direttive esplicite.
#######################################################################################
"""

def create_meta_training_document(input_csv, output_dir):
    """
    Creates a meta-training document for LLMs with general rules
    and representative examples
    """
    """
    #######################################################################################
    # FUNZIONE PRINCIPALE: CREAZIONE DEL DOCUMENTO DI META-TRAINING
    #
    # Questa funzione trasforma un dataset di anomalie in un documento strutturato
    # che insegna al LLM come ragionare sui dati di compressori industriali.
    # 
    # STRUTTURA DELIBERATA DEL DOCUMENTO:
    # La struttura del documento segue un pattern pedagogico specifico:
    # 1. Framework concettuale → Definizioni fondamentali delle classi da distinguere
    # 2. Parametri critici → Conoscenza tecnica di riferimento sulle soglie
    # 3. Regole di classificazione → Euristiche per la decisione
    # 4. Esempi concreti → Casi reali annotati per apprendimento per esempi
    # 5. Framework decisionale → Processo decisionale step-by-step
    #
    # Il documento è progettato per introdurre progressivamente la complessità,
    # facilitando l'apprendimento del modello su come affrontare nuovi casi.
    #######################################################################################
    """
    df = pd.read_csv(input_csv)
    
    # Check if anomaly_type column exists
    if 'anomaly_type' not in df.columns:
        print("Warning: 'anomaly_type' column not found in the dataset")
        if 'is_anomaly' in df.columns:
            print("Using 'is_anomaly' column instead")
            anomalies_df = df[df['is_anomaly'] == 1].copy()
            # Create anomaly_type based on is_anomaly
            anomalies_df['anomaly_type'] = 'TRUE_POSITIVE'
        else:
            print("Error: Required columns not found in the dataset")
            return None
    else:
        # Filter only TRUE_POSITIVE and FALSE_POSITIVE
        anomalies_df = df[(df['anomaly_type'] == 'TRUE_POSITIVE') | 
                          (df['anomaly_type'] == 'FALSE_POSITIVE')].copy()
    
    """
    #######################################################################################
    # GESTIONE FLESSIBILE DEL DATASET
    #
    # Il codice implementa una tolleranza agli errori per gestire vari formati di dataset:
    # - Se 'anomaly_type' è presente → utilizzo diretto
    # - Se solo 'is_anomaly' è presente → conversione automatica a TRUE_POSITIVE
    #
    # NOTA CRITICA: Questa è una semplificazione necessaria che assume che tutte le
    # anomalie segnalate siano TRUE_POSITIVE, ignorando i FALSE_POSITIVE. Questa è una
    # limitazione significativa che potrebbe portare a un documento di meta-training
    # sbilanciato, ma è un compromesso pragmatico per gestire dataset incompleti.
    #######################################################################################
    """
    
    print(f"Dataset loaded: {len(df)} rows, filtered to {len(anomalies_df)} anomalies")
    
    # Select representative examples (10 per class)
    try:
        true_positives = anomalies_df[anomalies_df['anomaly_type'] == 'TRUE_POSITIVE'].sample(10, random_state=42)
        false_positives = anomalies_df[anomalies_df['anomaly_type'] == 'FALSE_POSITIVE'].sample(10, random_state=42)
        
        """
        #######################################################################################
        # CAMPIONAMENTO LIMITATO DEGLI ESEMPI
        #
        # SCELTA CONSAPEVOLE: Limitiamo a 10 esempi per classe, un numero deliberatamente 
        # basso rispetto alle potenzialità del dataset completo. Questa è una decisione 
        # basata su due considerazioni:
        #
        # 1. VINCOLI DI CONTEXT WINDOW: I prompt LLM hanno limiti di lunghezza massima 
        #    (context window) e inserire troppi esempi renderebbe il prompt inutilizzabile
        #
        # 2. EFFICIENZA DI APPRENDIMENTO: I test empirici mostrano che i LLM apprendono 
        #    efficacemente da un numero limitato di esempi ben curati piuttosto che da 
        #    una grande quantità di dati ripetitivi
        #
        # SEED FISSO: Utilizziamo random_state=42 per garantire riproducibilità esatta
        # dei risultati in diverse esecuzioni
        #######################################################################################
        """
        
        print(f"Selected {len(true_positives)} TRUE_POSITIVE and {len(false_positives)} FALSE_POSITIVE examples")
    except ValueError as e:
        print(f"Error sampling examples: {e}")
        print("Using all available examples instead")
        true_positives = anomalies_df[anomalies_df['anomaly_type'] == 'TRUE_POSITIVE']
        false_positives = anomalies_df[anomalies_df['anomaly_type'] == 'FALSE_POSITIVE']
        
        """
        #######################################################################################
        # GESTIONE DI DATASET PICCOLI (FALLBACK)
        #
        # Implementiamo una strategia di fallback nel caso il dataset contenga meno di 10 
        # esempi per classe. In questo caso, utilizziamo tutti gli esempi disponibili.
        #
        # NOTA CRITICA: Questo è un compromesso sub-ottimale che potrebbe portare a:
        # 1. Un documento meta-training troppo breve (se ci sono pochissimi esempi)
        # 2. Un documento troppo lungo (se ci sono centinaia di esempi)
        # 
        # L'ideale sarebbe implementare una logica più sofisticata che seleziona esempi
        # diversificati basati su caratteristiche del dataset, ma questa rappresenta
        # una soluzione pragmatica per gestire la maggior parte dei casi reali.
        #######################################################################################
        """
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the training document
    output_md = os.path.join(output_dir, "anomaly_classification_training.md")
    
    """
    #######################################################################################
    # STRUTTURA DEL DOCUMENTO MARKDOWN
    #
    # Scegliamo deliberatamente il formato Markdown rispetto ad altri formati come JSON
    # o testo semplice per diverse ragioni chiave:
    #
    # 1. LEGGIBILITÀ: Il markdown è facilmente leggibile sia da umani che da LLM
    #
    # 2. STRUTTURA SEMANTICA: Le intestazioni (##) creano una chiara gerarchia di concetti
    #    che aiuta il LLM a comprendere le relazioni tra le diverse sezioni
    #
    # 3. FACILITÀ DI INTEGRAZIONE: Il formato è facilmente integrabile nel prompt system
    #    senza necessità di parsing complessi
    #
    # 4. MODELLAZIONE COGNITIVA: La struttura del documento rispecchia il processo di
    #    ragionamento che vogliamo il LLM segua, dall'astratto al concreto
    #######################################################################################
    """
    
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Guide to Classifying Anomalies in Industrial Compressors\n\n")
        
        # Section 1: Introduction and conceptual framework
        f.write("## 1. Anomaly Classification Framework\n\n")
        f.write("In industrial compressor monitoring, it is essential to distinguish between **true anomalies** that require maintenance interventions and **false positives** that are temporary variations in operating parameters caused by external or transient factors.\n\n")
        
        """
        #######################################################################################
        # SEZIONE 1: FRAMEWORK CONCETTUALE
        #
        # Questa sezione iniziale è deliberatamente astratta e concettuale, fornendo
        # definizioni precise di "true anomaly" e "false positive". La scelta di iniziare
        # con questa sezione riflette un principio didattico fondamentale:
        # insegnare prima i concetti generali prima di passare ai casi specifici.
        #
        # UTILITÀ PER IL LLM: Questa sezione calibra il "lessico tecnico" del modello 
        # e crea strutture cognitive chiare per la classificazione binaria che seguirà.
        # L'uso di elenchi puntati non è casuale: facilita l'estrazione di feature
        # discriminative da parte del modello.
        #######################################################################################
        """
        
        f.write("### Definition of True Anomaly (TRUE_POSITIVE)\n\n")
        f.write("A true anomaly is a deviation from standard operating parameters that:\n")
        f.write("- Indicates a deterioration in compressor performance\n")
        f.write("- Can lead to failures if not addressed\n")
        f.write("- Requires planned or immediate maintenance intervention\n")
        f.write("- Persists over time regardless of environmental conditions\n\n")
        
        f.write("### Definition of False Positive (FALSE_POSITIVE)\n\n")
        f.write("A false positive is a deviation from standard operating parameters that:\n")
        f.write("- Is correlated with external factors such as environmental conditions\n")
        f.write("- Is temporary and tends to normalize without intervention\n")
        f.write("- Does not indicate actual deterioration in compressor performance\n")
        f.write("- Does not require maintenance interventions\n\n")
        
        # Section 2: Critical parameters and thresholds
        f.write("## 2. Critical Parameters and Thresholds\n\n")
        
        """
        #######################################################################################
        # SEZIONE 2: PARAMETRI TECNICI E SOGLIE
        #
        # Questa sezione fornisce informazioni tecniche estremamente specifiche per il 
        # compressore CSD 102-8, un modello industriale reale. L'uso di tabelle markdown 
        # è una scelta deliberata:
        #
        # 1. FORMAT TABELLARE: Facilita l'estrazione di valori esatti e soglie tecniche
        #    che il LLM può poi utilizzare nei suoi ragionamenti
        #
        # 2. PRECISIONE NUMERICA: Fornisce valori numerici esatti piuttosto che descrizioni
        #    vaghe, ancorando l'analisi a parametri quantitativi
        #
        # NOTA IMPORTANTE: Le soglie qui fornite sono specifiche per questo modello di
        # compressore e non generalizzabili ad altri modelli - questa specificità è 
        # intenzionale, poiché vogliamo che il LLM si specializzi esattamente su questo
        # use case.
        #######################################################################################
        """
        
        f.write("### Compressor CSD 102-8 Parameters\n\n")
        f.write("| Parameter | Normal Range | Warning Threshold | Critical Threshold |\n")
        f.write("|-----------|---------------|---------------------|---------------|\n")
        f.write("| Discharge Temperature | 70-95°C | 95-115°C | >115°C |\n")
        f.write("| Vibration | 0.5-2.8 mm/s | 2.8-4.5 mm/s | >4.5 mm/s |\n")
        f.write("| Bearing Temperature | 60-80°C | 80-95°C | >95°C |\n")
        f.write("| Discharge Pressure | 5.5-8.0 bar | 8.0-9.0 bar | >9.0 bar |\n")
        f.write("| Motor Speed | 2900-3000 rpm | 2800-2900 rpm | <2800 rpm |\n\n")
        
        f.write("### Influence of Environmental Parameters\n\n")
        f.write("| Environmental Parameter | Influence on Compressor Parameters |\n")
        f.write("|---------------------|----------------------------------------|\n")
        f.write("| High ambient temperature (>30°C) | Can increase discharge temperature by 5-15°C |\n")
        f.write("| High humidity (>85%) | Can reduce cooling efficiency |\n")
        f.write("| Atmospheric pressure variations | Can affect suction and discharge pressures |\n\n")
        
        # Section 3: Classification rules with integrated overrides
        f.write("## 3. Rules for Anomaly Classification\n\n")
        f.write("### Critical Override Rules (Highest Priority)\n\n")
        f.write("These rules take absolute precedence in classification decisions:\n\n")
        
        # Critical Parameters Override
        f.write("#### Critical Parameters Rule (TRUE ANOMALY)\n\n")
        f.write("If ANY of these critical thresholds are exceeded, the case MUST be classified as TRUE ANOMALY regardless of other factors:\n")
        f.write("- Discharge temperature > 115°C\n")
        f.write("- Vibration > 4.5 mm/s\n")
        f.write("- Bearing temperature > 95°C\n\n")
        
        # Environmental Correlation Override
        f.write("#### Environmental Correlation Rule (FALSE POSITIVE)\n\n")
        f.write("If ANY of these conditions are met, the case should be classified as FALSE POSITIVE unless critical thresholds are exceeded:\n")
        f.write("- Ambient temperature > 32°C AND discharge temperature < 110°C\n")
        f.write("- Humidity > 80% AND vibration < 3.8 mm/s\n\n")
        
        # Pattern-Specific Override
        f.write("#### Pattern-Specific Rule (TRUE ANOMALY)\n\n")
        f.write("If ALL of these specific conditions are met simultaneously, the case should be classified as TRUE ANOMALY despite environmental correlations:\n")
        f.write("- Discharge temperature between 105°C and 108°C\n")
        f.write("- Vibration between 3.1 mm/s and 3.8 mm/s\n")
        f.write("- Ambient temperature between 30°C and 34°C\n\n")
        f.write("*Note: This specific pattern indicates an early-stage bearing wear that shows a characteristic thermal-vibration signature even when ambient temperature is elevated.*\n\n")
        
        # Normal Parameters Override
        f.write("#### Normal Parameters Rule (FALSE POSITIVE)\n\n")
        f.write("If ALL parameters are within normal operating ranges, the case MUST be classified as FALSE POSITIVE:\n")
        f.write("- Discharge temperature < 95°C\n")
        f.write("- Vibration < 2.8 mm/s\n\n")
        
        # Standard Classification Rules
        f.write("### Standard Classification Rules (Secondary Priority)\n\n")
        f.write("Apply these rules only if no override rule matches:\n\n")

        f.write("#### Indicators of True Anomaly (requires maintenance)\n\n")
        f.write("1. **Vibration above critical threshold**: Vibration > 4.5 mm/s indicates mechanical problems\n")
        f.write("2. **Critical discharge temperature**: Temperature > 115°C not correlated with ambient temperature\n")
        f.write("3. **Abnormal bearing temperature**: Bearing temperature > 95°C indicates bearing wear/damage\n")
        f.write("4. **Abnormal discharge pressure**: Pressure > 9.0 bar indicates valve or control problems\n")
        f.write("5. **Combination of moderate anomalies**: Multiple parameters simultaneously out of range\n\n")
        
        f.write("#### Indicators of False Positive (does not require maintenance)\n\n")
        f.write("1. **Correlation with ambient temperature**: Increase in discharge temperature proportional to increase in ambient temperature\n")
        f.write("2. **Anomalies during high humidity**: Parameters that return to normal when humidity decreases\n")
        f.write("3. **Correlation with barometric variations**: Pressure variations corresponding to atmospheric changes\n")
        f.write("4. **Transient anomalies**: Deviations that normalize within a few hours\n")
        f.write("5. **Multiple small deviations**: Parameters slightly out of range but not in critical zone\n\n")
        
        # Section 4: Examples of true anomalies
        f.write("## 4. Examples of True Anomalies (TRUE_POSITIVE)\n\n")
        
        """
        #######################################################################################
        # SEZIONE 4-5: ESEMPI CONCRETI PER APPRENDIMENTO DA CASI
        #
        # Queste sezioni cruciali forniscono esempi concreti presi dal dataset reale, per
        # entrambe le classi di anomalie. Questo approccio implementa un design pattern
        # fondamentale nell'addestramento di LLM: l'apprendimento basato su esempi (ICL).
        #
        # MOTIVI DELL'EFFICACIA DEGLI ESEMPI:
        # 1. ANCORAGGIO QUANTITATIVO: Gli esempi mostrano valori numerici precisi e
        #    relazioni tra parametri che il modello può imparare
        #
        # 2. PATTERN RECOGNITION: La ripetizione di esempi simili ma non identici aiuta
        #    il LLM a generalizzare pattern discriminativi
        #
        # 3. APPRENDIMENTO IMPLICITO: Alcuni pattern complessi (es. relazione tra
        #    temperatura ambiente e temperatura di scarico) sono difficili da esprimere
        #    come regole ma facili da assimilare dagli esempi
        #
        # FORMATTAZIONE STRUTTURATA: La scelta di utilizzare sezioni markdown di 3° livello
        # (###) per ciascun esempio non è casuale - crea una struttura gerarchica chiara
        # che facilita l'elaborazione da parte del LLM.
        #######################################################################################
        """
        
        for i, (_, example) in enumerate(true_positives.iterrows(), 1):
            f.write(f"### Example {i}\n\n")
            f.write("**Compressor parameters:**\n")
            f.write(f"- Discharge Temperature: {example['discharge_temp_true']:.1f}°C")
            if 'discharge_temp_pred' in example:
                f.write(f" (expected: {example['discharge_temp_pred']:.1f}°C)")
            f.write("\n")
            
            f.write(f"- Discharge Pressure: {example['discharge_pressure_true']:.2f} bar")
            if 'discharge_pressure_pred' in example:
                f.write(f" (expected: {example['discharge_pressure_pred']:.2f} bar)")
            f.write("\n")
            
            f.write(f"- Suction Pressure: {example['suction_pressure_true']:.2f} bar")
            if 'suction_pressure_pred' in example:
                f.write(f" (expected: {example['suction_pressure_pred']:.2f} bar)")
            f.write("\n")
            
            f.write(f"- Vibration: {example['vibration_true']:.2f} mm/s")
            if 'vibration_pred' in example:
                f.write(f" (expected: {example['vibration_pred']:.2f} mm/s)")
            f.write("\n")
            
            if 'bearing_temp_true' in example:
                f.write(f"- Bearing Temperature: {example['bearing_temp_true']:.1f}°C\n")
                
            if 'motor_speed_true' in example:
                f.write(f"- Motor Speed: {example['motor_speed_true']:.0f} rpm\n")
            
            """
            #######################################################################################
            # GESTIONE FLESSIBILE DEI PARAMETRI OPZIONALI
            #
            # Il codice gestisce in modo robusto parametri opzionali come bearing_temp_true
            # o motor_speed_true che potrebbero mancare in alcuni dataset. Questa flessibilità
            # è importante perché:
            #
            # 1. Supporta diversi formati di dataset senza errori
            # 2. Mantiene la leggibilità omettendo parametri non disponibili
            # 3. Scala automaticamente a dataset con configurazioni diverse di sensori
            #
            # Si noti anche l'attenzione alla formattazione numerica (.1f, .2f, .0f)
            # per garantire una presentazione coerente e leggibile dei valori.
            #######################################################################################
            """
            
            f.write("\n**Environmental conditions:**\n")
            f.write(f"- Ambient Temperature: {example['ambient_temperature']:.1f}°C\n")
            f.write(f"- Humidity: {example['humidity']:.0f}%\n")
            f.write(f"- Atmospheric Pressure: {example['atmospheric_pressure']:.0f} hPa\n\n")
            
            f.write("**Analysis:**\n")
            if 'anomaly_description' in example and example['anomaly_description']:
                f.write(f"{example['anomaly_description']}\n\n")
            else:
                f.write("This represents a true anomaly requiring maintenance intervention. The parameter deviations indicate genuine mechanical issues rather than transient conditions or environmental effects.\n\n")
            
            f.write("---\n\n")
        
        # Section 5: Examples of false positives
        f.write("## 5. Examples of False Positives (FALSE_POSITIVE)\n\n")
        
        for i, (_, example) in enumerate(false_positives.iterrows(), 1):
            f.write(f"### Example {i}\n\n")
            f.write("**Compressor parameters:**\n")
            f.write(f"- Discharge Temperature: {example['discharge_temp_true']:.1f}°C")
            if 'discharge_temp_pred' in example:
                f.write(f" (expected: {example['discharge_temp_pred']:.1f}°C)")
            f.write("\n")
            
            f.write(f"- Discharge Pressure: {example['discharge_pressure_true']:.2f} bar")
            if 'discharge_pressure_pred' in example:
                f.write(f" (expected: {example['discharge_pressure_pred']:.2f} bar)")
            f.write("\n")
            
            f.write(f"- Suction Pressure: {example['suction_pressure_true']:.2f} bar")
            if 'suction_pressure_pred' in example:
                f.write(f" (expected: {example['suction_pressure_pred']:.2f} bar)")
            f.write("\n")
            
            f.write(f"- Vibration: {example['vibration_true']:.2f} mm/s")
            if 'vibration_pred' in example:
                f.write(f" (expected: {example['vibration_pred']:.2f} mm/s)")
            f.write("\n")
            
            if 'bearing_temp_true' in example:
                f.write(f"- Bearing Temperature: {example['bearing_temp_true']:.1f}°C\n")
                
            if 'motor_speed_true' in example:
                f.write(f"- Motor Speed: {example['motor_speed_true']:.0f} rpm\n")
            
            f.write("\n**Environmental conditions:**\n")
            f.write(f"- Ambient Temperature: {example['ambient_temperature']:.1f}°C\n")
            f.write(f"- Humidity: {example['humidity']:.0f}%\n")
            f.write(f"- Atmospheric Pressure: {example['atmospheric_pressure']:.0f} hPa\n\n")
            
            f.write("**Analysis:**\n")
            if 'anomaly_description' in example and example['anomaly_description']:
                f.write(f"{example['anomaly_description']}\n\n")
            else:
                f.write("This represents a false positive that does not require maintenance intervention. The parameter deviations are likely due to environmental factors or transient conditions rather than actual mechanical issues.\n\n")
            
            f.write("---\n\n")
        
        # Section 6: Decision Framework
        f.write("## 6. Decision Framework for Classification\n\n")
        f.write("To correctly classify an anomaly, follow this decision process in strict order of priority:\n\n")
        
        f.write("1. **Apply Critical Override Rules First**\n")
        f.write("   - Is any critical parameter exceeded? (Temperature > 115°C OR Vibration > 4.5 mm/s OR Bearing Temperature > 95°C) → TRUE ANOMALY\n")
        f.write("   - Is the specific pattern present? (Temperature 105-108°C AND Vibration 3.1-3.8 mm/s AND Ambient 30-34°C) → TRUE ANOMALY\n")
        f.write("   - Are all parameters within normal range? (Temperature < 95°C AND Vibration < 2.8 mm/s) → FALSE POSITIVE\n")
        f.write("   - Is there clear environmental correlation? (Ambient > 32°C AND Discharge < 110°C) OR (Humidity > 80% AND Vibration < 3.8 mm/s) → FALSE POSITIVE\n\n")
        
        f.write("2. **If No Override Rule Applies, Use Standard Classification Rules**\n")
        f.write("   - Does vibration exceed 4.5 mm/s? → True anomaly\n")
        f.write("   - Does discharge temperature exceed 115°C? → True anomaly\n")
        f.write("   - Does bearing temperature exceed 95°C? → True anomaly\n")
        f.write("   - Does discharge pressure exceed 9.0 bar? → True anomaly\n\n")
        
        f.write("3. **Analyze Environmental Correlations**\n")
        f.write("   - Does the anomaly coincide with ambient temperature >30°C? → Possible false positive\n")
        f.write("   - Does the anomaly coincide with humidity >85%? → Possible false positive\n")
        f.write("   - Does the anomaly coincide with atmospheric pressure variations? → Possible false positive\n\n")
        
        f.write("4. **Verify Persistence**\n")
        f.write("   - Does the anomaly persist for more than 12 hours? → Likely true anomaly\n")
        f.write("   - Is the anomaly transient (few hours)? → Likely false positive\n\n")
        
        f.write("5. **Evaluate Combined Severity**\n")
        f.write("   - Are multiple parameters simultaneously approaching critical thresholds? → True anomaly\n")
        f.write("   - Only isolated small deviations? → Likely false positive\n\n")
    
    print(f"Meta-training document saved to: {output_md}")

if __name__ == "__main__":
    # New dataset path with corrected folder name
    input_file = "C:\\Users\\gaia1\\Desktop\\UDOO Lab\\Predictive Maintenance & LLMs\\code 10 - validation\\test_predictions\\dataset\\compressor_monitoring_dataset.csv"
    
    # Save the document to the llm_training directory
    output_dir = "C:\\Users\\gaia1\\Desktop\\UDOO Lab\\Predictive Maintenance & LLMs\\code 10 - validation\\test_predictions\\llm_training\\llm_meta_training"
    
    # Verify that the input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("Absolute path:", os.path.abspath(input_file))
        exit(1)
    
    create_meta_training_document(input_file, output_dir)