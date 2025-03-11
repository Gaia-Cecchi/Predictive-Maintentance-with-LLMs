import streamlit as st
import pandas as pd
import numpy as np  # Aggiungiamo l'import di numpy
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
from llm_interface import CompressorAssistant
import sqlite3
from pathlib import Path
from prediction_service import PredictionService
import logging
from plotly.subplots import make_subplots

# Import the KnowledgeChatAssistant
from llm_chat import KnowledgeChatAssistant
from styles import get_chat_styles
import asyncio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurazione della pagina
st.set_page_config(
    page_title="Compressor Monitoring Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inizializzazione della sessione
if 'assistant' not in st.session_state:
    st.session_state.assistant = CompressorAssistant()

if 'chat_assistant' not in st.session_state:
    st.session_state.chat_assistant = KnowledgeChatAssistant(base_path=Path("/teamspace/studios/this_studio"))
    
if 'base_path' not in st.session_state:
    st.session_state.base_path = Path("/teamspace/studios/this_studio")
    st.session_state.db_path = st.session_state.base_path / 'processed_data' / 'compressor_data.db'

# Modifica della data di default
if 'simulated_datetime' not in st.session_state:
    st.session_state.simulated_datetime = datetime(2024, 6, 4, 12, 0)

# Initialize chat messages if not already in session state
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# Sincronizza il timestamp degli assistenti con quello di Streamlit
st.session_state.assistant.set_current_time(st.session_state.simulated_datetime)
st.session_state.chat_assistant.set_current_time(st.session_state.simulated_datetime)

@st.cache_data(ttl=3600)
def get_compressors():
    """Get list of available compressors"""
    with sqlite3.connect(st.session_state.db_path) as conn:
        query = "SELECT DISTINCT compressor_id FROM compressor_measurements"
        df = pd.read_sql_query(query, conn)
        return df['compressor_id'].tolist()

@st.cache_data(ttl=300)
def get_anomalies(compressor_id, start_date=None, end_date=None, _sim_time=None):
    """Get anomalies for a specific compressor"""
    with sqlite3.connect(st.session_state.db_path) as conn:
        query = """
            SELECT *
            FROM anomalies
            WHERE compressor_id = ?
        """
        params = [compressor_id]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        # Converti esplicitamente la colonna timestamp in datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

@st.cache_data(ttl=300)
def get_measurements(compressor_id, start_date=None, end_date=None, _sim_time=None):
    """Get measurements for a specific compressor"""
    with sqlite3.connect(st.session_state.db_path) as conn:
        query = """
            SELECT *
            FROM compressor_measurements
            WHERE compressor_id = ?
        """
        params = [compressor_id]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp DESC"
        
        return pd.read_sql_query(query, conn, params=params)

@st.cache_data(ttl=3600)
def get_weather_data(date):
    """Get weather data for a specific date"""
    with sqlite3.connect(st.session_state.db_path) as conn:
        query = """
            SELECT *
            FROM weather_data
            WHERE date = DATE(?)
        """
        return pd.read_sql_query(query, conn, params=[date])

@st.cache_data(ttl=300)
def calculate_kpis(compressor_id, timeframe='7d'):
    """Calculate KPIs for a specific compressor"""
    end_date = st.session_state.simulated_datetime
    if timeframe == '7d':
        start_date = end_date - timedelta(days=7)
    elif timeframe == '30d':
        start_date = end_date - timedelta(days=30)
    else:
        start_date = end_date - timedelta(days=1)

    # Ottieni tutte le anomalie nel periodo
    anomalies_df = get_anomalies(compressor_id, start_date, end_date)
    # Filtra solo le vere anomalie
    true_anomalies = anomalies_df[anomalies_df['is_anomaly'] == True]
    
    # Ottieni tutte le misurazioni nel periodo
    measurements = get_measurements(compressor_id, start_date, end_date)
    
    # Calcola il numero totale di misurazioni uniche (per timestamp)
    total_timestamps = len(measurements['timestamp'].unique()) if not measurements.empty else 0
    
    # Calcola il numero di timestamp con anomalie
    anomaly_timestamps = len(true_anomalies['timestamp'].unique()) if not true_anomalies.empty else 0
    
    return {
        'anomaly_count': anomaly_timestamps,  # Ora conta solo i timestamp con vere anomalie
        'avg_cosphi': measurements['cosphi'].mean() if not measurements.empty else 0,
        'reliability': 1 - (anomaly_timestamps / total_timestamps) if total_timestamps > 0 else 1,
        'total_runtime': len(measurements) * 5 / 60  # Assuming 5-minute intervals
    }

@st.cache_data(ttl=300)
def get_historical_values(start_date, end_date):
    """Get all values between two dates"""
    try:
        with sqlite3.connect(st.session_state.db_path) as conn:
            # Query anomalies with explicit hour selection
            anomalies_query = """
                SELECT 
                    timestamp,
                    compressor_id,
                    parameter,
                    is_anomaly,
                    true_value,
                    predicted_value,
                    error_value
                FROM anomalies
                WHERE DATE(timestamp) BETWEEN DATE(?) AND DATE(?)
                AND strftime('%M', timestamp) = '00'  -- Solo ore esatte
                ORDER BY timestamp ASC
            """
            
            anomalies_df = pd.read_sql_query(
                anomalies_query, 
                conn, 
                params=[start_date, end_date],
                parse_dates=['timestamp']
            )
            
            # Query weather data for the date range
            weather_query = """
                SELECT * FROM weather_data
                WHERE date BETWEEN DATE(?) AND DATE(?)
                ORDER BY date ASC
            """
            weather_df = pd.read_sql_query(
                weather_query,
                conn,
                params=[start_date, end_date]
            )
            
            # Process weather data
            if not weather_df.empty:
                weather_data = []
                for _, row in weather_df.iterrows():
                    weather_dict = {}
                    for key, value in row.items():
                        if pd.isna(value):
                            weather_dict[key] = None if key == 'phenomena' else 0.0
                        elif isinstance(value, pd.Timestamp):
                            weather_dict[key] = value.strftime('%Y-%m-%d')
                        else:
                            weather_dict[key] = float(value) if isinstance(value, (int, float)) else str(value)
                    weather_data.append(weather_dict)
            else:
                weather_data = None
            
            return {
                'anomalies': anomalies_df.to_dict('records') if not anomalies_df.empty else None,
                'weather': weather_data
            }
            
    except Exception as e:
        st.error(f"Error retrieving historical data: {str(e)}")
        st.exception(e)
        return None

@st.cache_data(ttl=300)
def get_historical_maintenance(start_date, end_date):
    """Get maintenance records between dates"""
    try:
        with sqlite3.connect(st.session_state.db_path) as conn:
            query = """
                SELECT 
                    date,
                    intervention_number,
                    intervention_type,
                    operating_hours,
                    activities,
                    anomalies,
                    recommendations,
                    compressor_id
                FROM maintenance
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date],
                parse_dates=['date']
            )
            
            return df
            
    except Exception as e:
        st.error(f"Errore nel caricamento delle manutenzioni: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_historical_failures(start_date, end_date):
    """Get failure records between dates from the database"""
    try:
        with sqlite3.connect(st.session_state.db_path) as conn:
            # Updated query to match actual schema
            query = """
                SELECT 
                    date,
                    failure_type,
                    frequency,
                    cause,
                    solution,
                    additional_info,
                    feedback,
                    compressor_id
                FROM failures
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date],
                parse_dates=['date']
            )
            
            if df.empty:
                st.info(f"Nessun guasto trovato tra {start_date} e {end_date}")
            else:
                st.success(f"Trovati {len(df)} guasti nel periodo selezionato")
                
            return df
            
    except Exception as e:
        st.error(f"Errore nel caricamento dei guasti: {str(e)}")
        st.exception(e)  # Show full traceback
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_historical_workload(start_date, end_date):
    """Get workload data between dates"""
    try:
        with sqlite3.connect(st.session_state.db_path) as conn:
            query = """
                SELECT 
                    date,
                    start_time,
                    end_time,
                    operating_hours,
                    load_percentage,
                    temperature,
                    humidity,
                    vibration,
                    pressure,
                    compressor_id
                FROM operating_conditions
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date],
                parse_dates=['date']
            )
            
            return df
            
    except Exception as e:
        st.error(f"Errore nel caricamento del carico operativo: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_historical_feedback(start_date, end_date):
    """Get user feedback between dates"""
    try:
        with sqlite3.connect(st.session_state.db_path) as conn:
            query = """
                SELECT 
                    date,
                    feedback_type,
                    title,
                    description,
                    rating,
                    comment,
                    compressor_id
                FROM feedback
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date],
                parse_dates=['date']
            )
            
            return df
            
    except Exception as e:
        st.error(f"Errore nel caricamento dei feedback: {str(e)}")
        return pd.DataFrame()

async def analyze_anomaly(timestamp, compressor_id):
    """Analyze a specific anomaly using the LLM assistant"""
    try:
        # Ensure timestamp is in correct format
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        analysis = await st.session_state.assistant.analyze_anomaly(timestamp, compressor_id)
        return analysis
    except Exception as e:
        st.error(f"Error analyzing anomaly: {str(e)}")
        st.exception(e)
        return "Failed to analyze anomaly."

async def ask_question(question):
    """Ask a question to the LLM assistant"""
    return await st.session_state.assistant.ask_question(question)

def historical_search_tab():
    """Ricerca Storica"""
    st.header("Ricerca Storica")
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Data Inizio",
            value=datetime(2024, 6, 4).date(),
            min_value=datetime(2024, 3, 1).date(),
            max_value=datetime(2024, 6, 30).date()
        )
    with col2:
        end_date = st.date_input(
            "Data Fine",
            value=datetime(2024, 6, 7).date(),
            min_value=start_date,
            max_value=datetime(2024, 6, 30).date()
        )
    
    if st.button("Cerca"):
        tabs = st.tabs(["Anomalie", "Manutenzioni", "Guasti", "Carico Operativo", "Feedback Utenti", "Meteo"])
        
        # Tab Anomalie
        with tabs[0]:
            st.subheader("Anomalie Rilevate")
            data = get_historical_values(start_date, end_date)
            if data and 'anomalies' in data and data['anomalies']:
                df_anomalies = pd.DataFrame(data['anomalies'])
                true_anomalies = df_anomalies[df_anomalies['is_anomaly'] == True]
                
                if not true_anomalies.empty:
                    # Statistiche
                    cols = st.columns(3)
                    cols[0].metric("Totale Anomalie", len(true_anomalies))
                    cols[1].metric("Timestamp Unici", len(true_anomalies['timestamp'].unique()))
                    cols[2].metric("Parametri Coinvolti", true_anomalies['parameter'].nunique())
                    
                    # Grafico timeline
                    fig = px.scatter(true_anomalies, x='timestamp', y='error_value',
                                   color='parameter',
                                   title='Timeline Anomalie')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabella dettagliata
                    st.dataframe(true_anomalies, use_container_width=True)
                else:
                    st.info("Nessuna anomalia trovata nel periodo selezionato")
        
        # Tab Manutenzioni
        with tabs[1]:
            st.subheader("Storico Manutenzioni")
            maintenance_df = get_historical_maintenance(start_date, end_date)
            
            if not maintenance_df.empty:
                # Metriche riassuntive
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Totale Interventi", len(maintenance_df))
                with col2:
                    st.metric("Compressori Coinvolti", maintenance_df['compressor_id'].nunique())
                
                # Grafico interventi nel tempo
                fig = px.scatter(
                    maintenance_df,
                    x='date',
                    y='intervention_type',
                    color='compressor_id',
                    title='Interventi di Manutenzione nel Tempo',
                    hover_data=['activities', 'anomalies', 'recommendations']
                )
                # Personalizza il layout
                fig.update_traces(marker=dict(size=12, symbol='diamond'))
                fig.update_layout(
                    xaxis_title="Data",
                    yaxis_title="Tipo di Intervento",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Grafico a barre per la distribuzione dei tipi di intervento
                intervention_counts = maintenance_df['intervention_type'].value_counts().reset_index()
                intervention_counts.columns = ['Tipo di Intervento', 'Numero di Interventi']
                
                fig2 = px.bar(
                    intervention_counts,
                    x='Tipo di Intervento',
                    y='Numero di Interventi',
                    title='Distribuzione Tipi di Intervento'
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Tabella dettagliata
                st.subheader("Dettaglio Interventi")
                st.dataframe(
                    maintenance_df,
                    use_container_width=True,
                    column_config={
                        "date": "Data",
                        "intervention_type": "Tipo Intervento",
                        "activities": "Attività",
                        "anomalies": "Anomalie",
                        "recommendations": "Raccomandazioni",
                        "compressor_id": "Compressore",
                        "operating_hours": "Ore di Funzionamento"
                    }
                )
            else:
                st.info("Nessuna manutenzione trovata nel periodo selezionato")
        
        # Tab Guasti
        with tabs[2]:
            st.subheader("Storico Guasti")
            failures_df = get_historical_failures(start_date, end_date)
            if not failures_df.empty:
                # Add summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Totale Guasti", len(failures_df))
                with col2:
                    st.metric("Compressori Coinvolti", failures_df['compressor_id'].nunique())
                
                # Show interactive table
                st.dataframe(
                    failures_df,
                    use_container_width=True,
                    column_config={
                        "date": "Data",
                        "failure_type": "Tipo Guasto",
                        "frequency": "Frequenza",
                        "cause": "Causa",
                        "solution": "Soluzione",
                        "additional_info": "Informazioni Aggiuntive",
                        "feedback": "Feedback",
                        "compressor_id": "Compressore"
                    }
                )
                
                # Add failure frequency chart
                fig = px.histogram(
                    failures_df,
                    x='failure_type',
                    title='Distribuzione dei Tipi di Guasto'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nessun guasto trovato nel periodo selezionato")
        
        # Tab Carico Operativo
        with tabs[3]:
            st.subheader("Carico Operativo")
            workload_df = get_historical_workload(start_date, end_date)
            
            if not workload_df.empty:
                # Main metrics charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.line(
                        workload_df,
                        x='date',
                        y='load_percentage',
                        color='compressor_id',
                        title='Andamento Carico Operativo'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.line(
                        workload_df,
                        x='date',
                        y=['temperature', 'humidity', 'pressure'],
                        title='Condizioni Operative'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.dataframe(
                    workload_df,
                    use_container_width=True,
                    column_config={
                        "date": "Data",
                        "load_percentage": "Carico %",
                        "temperature": "Temperatura",
                        "humidity": "Umidità",
                        "pressure": "Pressione",
                        "vibration": "Vibrazioni",
                        "compressor_id": "Compressore"
                    }
                )
            else:
                st.info("Nessun dato di carico operativo trovato nel periodo selezionato")
        
        # Tab Feedback Utenti
        with tabs[4]:
            st.subheader("Feedback Utenti")
            feedback_df = get_historical_feedback(start_date, end_date)
            
            if not feedback_df.empty:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Totale Feedback", len(feedback_df))
                with col2:
                    avg_rating = feedback_df['rating'].mean()
                    st.metric("Valutazione Media", f"{avg_rating:.1f}/5")
                with col3:
                    negative_feedback = len(feedback_df[feedback_df['rating'] <= 3])
                    st.metric("Feedback Negativi", negative_feedback)
                
                # Ratings distribution chart
                fig = px.histogram(
                    feedback_df,
                    x='rating',
                    title='Distribuzione Valutazioni',
                    nbins=5
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.dataframe(
                    feedback_df,
                    use_container_width=True,
                    column_config={
                        "date": "Data",
                        "feedback_type": "Tipo",
                        "title": "Titolo",
                        "description": "Descrizione",
                        "rating": "Valutazione",
                        "comment": "Commento",
                        "compressor_id": "Compressore"
                    }
                )
            else:
                st.info("Nessun feedback trovato nel periodo selezionato")
        
        # Tab Meteo
        with tabs[5]:
            st.subheader("Dati Meteo")
            if data and 'weather' in data and data['weather']:
                weather_df = pd.DataFrame(data['weather'])
                st.dataframe(weather_df, use_container_width=True)
            else:
                st.info("Nessun dato meteo disponibile per le date selezionate")

def get_compressor_status(compressor_id: str, current_time: datetime) -> dict:
    """Get current status of a compressor at specified time"""
    try:
        # Check for anomalies in the last 5 minutes
        current_anomalies = get_anomalies(
            compressor_id,
            current_time - timedelta(minutes=5),
            current_time
        )
        
        if current_anomalies is not None:
            true_anomalies = current_anomalies[current_anomalies['is_anomaly'] == True]
            has_anomalies = not true_anomalies.empty
            anomaly_count = len(true_anomalies) if has_anomalies else 0
        else:
            has_anomalies = False
            anomaly_count = 0
            
        return {
            'has_anomalies': has_anomalies,
            'anomaly_count': anomaly_count
        }
    except Exception as e:
        logger.error(f"Error getting compressor status: {str(e)}")
        return {'has_anomalies': False, 'anomaly_count': 0}

def render_compressor_dashboard():
    """Render real-time compressor monitoring dashboard"""
    st.header("Dashboard Monitoraggio Compressori")
    
    # Custom CSS per lo stile delle card e dei pulsanti
    st.markdown("""
        <style>
        .compressor-card {
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #ddd;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .compressor-card:hover {
            border-color: #0066cc;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card-header {
            font-size: 1.2rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        .card-status {
            color: #666;
            margin-top: 0.5rem;
        }
        .anomaly-details {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #eee;
        }
        .parameter-list {
            margin: 0.5rem 0;
            padding-left: 1rem;
        }
        .analyze-button {
            margin-top: 1rem;
        }
        
        .stButton button {
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 0.9rem 0.9rem 0.9rem 0.9rem;
            margin: 1rem 0 0 0 !important;
        }
        .stButton button p {
            text-align: left !important;
            width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Inizializza stato della card selezionata se non esiste
    if 'selected_card' not in st.session_state:
        st.session_state.selected_card = None
    
    # Layout a griglia per i compressori e meteo (3 colonne invece di 2)
    col1, col2, col3 = st.columns([1, 1, 1])  # Distribuisci equamente lo spazio
    
    # Setup compressors (come prima)
    compressors = {
        'CSD102': {'model': 'CSD102'},
        'BSD72': {'model': 'BSD72'},
        'BS61': {'model': 'BS61'},
        'SK21': {'model': 'SK21'}
    }
    
    # Aggiorna lo stato di ogni compressore
    current_time = st.session_state.simulated_datetime
    for comp_id in compressors:
        status = get_compressor_status(comp_id, current_time)
        compressors[comp_id].update(status)
    
    # Ottieni i dati meteo per il timestamp corrente
    weather_data = get_weather_data(current_time)
    
    # Renderizza i compressori nelle prime due colonne
    for idx, (comp_id, info) in enumerate(compressors.items()):
        with col1 if idx < 2 else col2:
            # Card del compressore
            card = st.container()
            
            # Status con emoji e testo su due righe
            if info['has_anomalies']:
                status_text = (
                    f"🔴 {info['model']}\n"  # Prima riga
                    f"⚠️ {info['anomaly_count']} Anomalie Rilevate"  # Seconda riga
                )
            else:
                status_text = (
                    f"🟢 {info['model']}\n"  # Prima riga
                    f"✅ Operativo"  # Seconda riga
                )
            
            # Pulsante che mostra lo status - usiamo markdown
            if card.button(
                status_text.replace('\n', '  \n'),  # Usiamo markdown line break
                key=f"card_{comp_id}",
                use_container_width=True,
                help="Click per dettagli"
            ):
                # Toggle selezione card
                if st.session_state.selected_card == comp_id:
                    st.session_state.selected_card = None
                else:
                    st.session_state.selected_card = comp_id
            
            # Espandi la card se selezionata e ci sono anomalie
            if st.session_state.selected_card == comp_id and info['has_anomalies']:
                anomalies = get_anomalies(comp_id, current_time - timedelta(minutes=5), current_time)
                if not anomalies.empty:
                    # Filtra le anomalie vere e calcola le deviazioni
                    true_anomalies = anomalies[anomalies['is_anomaly'] == True]
                    
                    with card:
                        st.markdown("##### Parametri Anomali:")
                        for _, row in true_anomalies.iterrows():
                            # Calcola la deviazione percentuale
                            deviation = ((row['true_value'] - row['predicted_value'])/row['predicted_value'] * 100)
                            # Formatta la stringa con il segno e la percentuale
                            deviation_str = f"{'+' if deviation > 0 else ''}{deviation:.1f}%"
                            # Mostra il parametro con la sua deviazione
                            st.markdown(f"- {row['parameter']}: {deviation_str}")
                        
                        # Pulsante analisi
                        if st.button("Analizza in 🔍 Analisi Anomalie", key=f"analyze_{comp_id}"):
                            st.session_state.analysis_timestamp = current_time
                            st.session_state.analysis_compressor = comp_id
                            st.rerun()
    
    # Aggiungi la card del meteo nella terza colonna
    with col3:
        # Aggiungi lo stile CSS personalizzato per la weather card
        st.markdown("""
            <style>
            /* Rimuovi spazi extra dai container di Streamlit */
            [data-testid="column"] > div:has(> .weather-card) {
                padding: 0 !important;
                margin: 0 !important;
            }
            
            [data-testid="column"] > div {
                padding-top: 0 !important;
                margin-top: 0 !important;
            }
            
            /* Stile della weather card */
            .weather-card {
                padding: 1.2rem;
                border: 1px solid rgb(49, 51, 63);
                border-radius: 0.5rem;
                background-color: transparent;
                margin: 0;
                width: 100%;
                white-space: pre-wrap;
                text-align: left !important;
                display: block;
                line-height: 1.6;
                font-size: 16px;
            }

            .weather-title {
                font-size: 18px;
                font-weight: bold;
                display: block;
            }
            </style>
        """, unsafe_allow_html=True)
        
        weather_data = get_weather_data(current_time)
        
        # Fix the DataFrame check
        if weather_data is not None and isinstance(weather_data, pd.DataFrame) and not weather_data.empty:
            data = weather_data.iloc[0]
            weather_status = f"""<span class="weather-title">🌡️ Meteo Santa Croce sull'Arno</span>
                🌞 Max: {data['max_temp']:.1f}°C
                ❄️ Min: {data['min_temp']:.1f}°C
                🌡️ Media: {data['avg_temp']:.1f}°C
                💧 Umidità: {data['humidity']:.0f}%"""
        else:
            weather_status = """<span class="weather-title">🌡️ Meteo Santa Croce sull'Arno</span>

                ⚠️ Dati meteo non disponibili"""

        # Formatta il testo con lo stesso spacing dei bottoni
        formatted_weather = "\n".join(line.strip() for line in weather_status.split("\n"))
        
        # Usa markdown con la classe personalizzata e assicurati che lo spacing sia corretto
        st.markdown(
            f'<div class="weather-card">{formatted_weather.replace(chr(10), "<br>")}</div>', 
            unsafe_allow_html=True
        )

@st.cache_data(ttl=300)
def get_current_environmental_values(timestamp):
    """Get environmental values for the exact simulated timestamp"""
    try:
        with sqlite3.connect(st.session_state.db_path) as conn:
            # Prima verifichiamo la struttura della tabella
            schema_query = "PRAGMA table_info(weather_data)"
            schema = pd.read_sql_query(schema_query, conn)
            logger.info(f"Weather table schema: {schema['name'].tolist()}")
            
            # Verifichiamo quali dati abbiamo per questa data
            all_columns = ", ".join(schema['name'].tolist())
            debug_query = f"""
                SELECT {all_columns}
                FROM weather_data 
                WHERE date = DATE(?)
            """
            debug_df = pd.read_sql_query(debug_query, conn, params=[timestamp])
            logger.info(f"DEBUG - Weather data for {timestamp.date()}: Found {len(debug_df)} records")
            
            if not debug_df.empty:
                logger.info(f"Sample record: {debug_df.iloc[0].to_dict()}")
                
                # Mappiamo i nomi delle colonne corrette (basandoci sul file config.py)
                # Modifica: usa sempre PRESSIONESLM mb per la pressione, mai PRESSIONEMEDIA mb
                column_mappings = {
                    'avg_temp': 'temperature',
                    'humidity': 'humidity',
                    'pressure_slm': 'pressure',  # Rinominato per chiarezza
                    'PRESSIONESLM mb': 'pressure',  # Usa sempre questa colonna per la pressione
                    # Alternative column names that might exist
                    'TMEDIA °C': 'temperature',
                    'UMIDITA %': 'humidity'
                    # 'PRESSIONEMEDIA mb' è stato rimosso intenzionalmente
                }
                
                # Verifichiamo quali colonne sono effettivamente presenti
                result = {}
                for db_col, api_col in column_mappings.items():
                    if db_col in debug_df.columns:
                        result[api_col] = debug_df.iloc[0][db_col]
                    
                # Ensure all required fields are present
                for field in ['temperature', 'humidity', 'pressure']:
                    if field not in result:
                        logger.warning(f"Field {field} not found in weather_data table")
                        result[field] = None
                
                logger.info(f"Mapped result: {result}")
                return result
            else:
                logger.warning(f"No weather data found for date {timestamp.date()}")
                return {
                    'temperature': None, 
                    'humidity': None, 
                    'pressure': None
                }
                
    except Exception as e:
        logger.error(f"Error getting environmental values: {str(e)}")
        return {'temperature': None, 'humidity': None, 'pressure': None}

@st.cache_data(ttl=300)
def get_current_operational_values(timestamp):
    """Get operational values for the exact simulated timestamp"""
    try:
        with sqlite3.connect(st.session_state.db_path) as conn:
            # Prima verifichiamo la struttura della tabella
            schema_query = "PRAGMA table_info(operating_conditions)"
            schema = pd.read_sql_query(schema_query, conn)
            logger.info(f"Operating conditions table schema: {schema['name'].tolist()}")
            
            # Verifichiamo quali dati abbiamo per questa data
            all_columns = ", ".join(schema['name'].tolist())
            debug_query = f"""
                SELECT {all_columns}
                FROM operating_conditions 
                WHERE date = DATE(?)
            """
            debug_df = pd.read_sql_query(debug_query, conn, params=[timestamp])
            logger.info(f"DEBUG - Operating conditions for {timestamp.date()}: Found {len(debug_df)} records")
            
            if not debug_df.empty:
                logger.info(f"Sample record: {debug_df.iloc[0].to_dict()}")
                
                # Mappiamo i nomi delle colonne corrette
                column_mappings = {
                    'load_percentage': 'load_percentage',
                    'operating_hours': 'operating_hours',
                    # Alternative column names that might exist
                    'Carico_Operativo': 'load_percentage',
                    'Ore_Funzionamento': 'operating_hours'
                }
                
                # Verifichiamo quali colonne sono effettivamente presenti
                result = {}
                for db_col, api_col in column_mappings.items():
                    if db_col in debug_df.columns:
                        result[api_col] = debug_df.iloc[0][db_col]
                
                # Ensure all required fields are present
                for field in ['load_percentage', 'operating_hours']:
                    if field not in result:
                        logger.warning(f"Field {field} not found in operating_conditions table")
                        result[field] = None
                
                logger.info(f"Mapped result: {result}")
                return result
            else:
                logger.warning(f"No operating data found for date {timestamp.date()}")
                return {
                    'load_percentage': None, 
                    'operating_hours': None
                }
                
    except Exception as e:
        logger.error(f"Error getting operational values: {str(e)}")
        return {'load_percentage': None, 'operating_hours': None}

def prediction_simulator_tab():
    """Tab Previsioni"""
    st.header("Previsioni")
    
    current_time = st.session_state.simulated_datetime
    st.info(f"Data e ora simulata: {current_time.strftime('%d/%m/%Y %H:%M')}")
    
    # Recupera i valori attuali
    try:
        with sqlite3.connect(st.session_state.db_path) as conn:
            # Verifichiamo prima la struttura della tabella meteo
            schema_query = "PRAGMA table_info(weather_data)"
            weather_schema = pd.read_sql_query(schema_query, conn)
            logger.info(f"Weather table schema: {weather_schema['name'].tolist()}")
            
            # Recupera dati meteo
            weather_query = """
                SELECT * 
                FROM weather_data
                WHERE date = DATE(?)
                LIMIT 1
            """
            weather_df = pd.read_sql_query(weather_query, conn, params=[current_time.strftime('%Y-%m-%d')])
            
            # Imposta valori meteo di default sicuri
            current_env = {
                'temperature': 20.0,
                'humidity': 50.0,
                'pressure': 1013.25  # Valore di default sicuro per la pressione
            }
            
            # Aggiorna con valori dal DB se disponibili
            if not weather_df.empty:
                # Mappa i nomi corretti delle colonne
                if 'avg_temp' in weather_df.columns:
                    current_env['temperature'] = weather_df['avg_temp'].iloc[0]
                if 'humidity' in weather_df.columns:
                    current_env['humidity'] = weather_df['humidity'].iloc[0]
                if 'avg_pressure' in weather_df.columns:
                    current_env['pressure'] = weather_df['avg_pressure'].iloc[0]
                    
            # Recupera dati operativi
            op_query = """
                SELECT * 
                FROM operating_conditions
                WHERE date = DATE(?)
                LIMIT 1
            """
            op_df = pd.read_sql_query(op_query, conn, params=[current_time.strftime('%Y-%m-%d')])
            
            # Imposta valori operativi di default
            current_op = {
                'load_percentage': 75,
                'operating_hours': 8
            }
            
            # Aggiorna con valori dal DB se disponibili
            if not op_df.empty:
                if 'load_percentage' in op_df.columns:
                    current_op['load_percentage'] = op_df['load_percentage'].iloc[0]
                if 'operating_hours' in op_df.columns:
                    current_op['operating_hours'] = op_df['operating_hours'].iloc[0]
    except Exception as e:
        st.error(f"Errore nel caricamento dei dati: {str(e)}")
        # Garantiamo valori di default sicuri in caso di errore
        current_env = {'temperature': 20.0, 'humidity': 50.0, 'pressure': 1013.25}
        current_op = {'load_percentage': 75, 'operating_hours': 8}
    
    # Assicurati che i valori siano nel range corretto e non None
    if current_env['pressure'] is None or current_env['pressure'] < 900.0:
        current_env['pressure'] = 1013.25
    
    if current_env['temperature'] is None:
        current_env['temperature'] = 20.0
        
    if current_env['humidity'] is None or not (0 <= current_env['humidity'] <= 100):
        current_env['humidity'] = 50.0
    
    if current_op['load_percentage'] is None or not (0 <= current_op['load_percentage'] <= 100):
        current_op['load_percentage'] = 75
    
    if current_op['operating_hours'] is None or not (0 <= current_op['operating_hours'] <= 24):
        current_op['operating_hours'] = 8
    
    with st.form("prediction_form"):
        st.subheader("Modifica Parametri")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Parametri Ambientali")
            
            temperature = st.number_input(
                "Temperatura (°C)", 
                min_value=-20.0, 
                max_value=50.0, 
                value=float(current_env['temperature']),
                step=0.1,
                help="Temperatura ambiente in gradi Celsius"
            )
            
            humidity = st.number_input(
                "Umidità (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=float(current_env['humidity']),
                step=1.0,
                help="Percentuale di umidità relativa"
            )
            
            pressure = st.number_input(
                "Pressione Atmosferica (mbar)", 
                min_value=900.0, 
                max_value=1100.0, 
                value=float(current_env['pressure']),
                step=0.1,
                help="Pressione atmosferica in millibar"
            )
        
        with col2:
            st.markdown("### Parametri Operativi")
            
            load_percentage = st.slider(
                "Percentuale di Carico (%)", 
                min_value=0, 
                max_value=100, 
                value=int(current_op['load_percentage']),
                help="Percentuale di carico del compressore"
            )
            
            operating_hours = st.number_input(
                "Ore di Funzionamento", 
                min_value=0, 
                max_value=24, 
                value=int(current_op['operating_hours']),
                help="Numero di ore di funzionamento al giorno"
            )
        
        prediction_horizon = st.slider(
            "Orizzonte di Previsione (ore)",
            min_value=1,
            max_value=24,
            value=6,
            help="Quante ore nel futuro prevedere"
        )
        
        submitted = st.form_submit_button("Genera Previsione")
        
        if submitted:
            try:
                modifications = {
                    'weather': {
                        'temperature': temperature,
                        'humidity': humidity,
                        'pressure': pressure
                    },
                    'operational': {
                        'load_percentage': load_percentage,
                        'operating_hours': operating_hours
                    }
                }
                
                prediction_service = PredictionService(base_path=st.session_state.base_path)
                results = prediction_service.predict_with_modified_conditions(
                    base_time=current_time,
                    modifications=modifications
                )
                
                st.subheader("Risultati della Previsione")
                
                # Mostra le condizioni effettive dal database
                with st.expander("Condizioni di riferimento dal database"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("#### Parametri Ambientali")
                        for param, value in results['actual_conditions']['weather'].items():
                            st.metric(label=param.replace('_', ' ').title(), value=f"{value:.2f}")
                    
                    with col_b:
                        st.markdown("#### Parametri Operativi")
                        for param, value in results['actual_conditions']['operational'].items():
                            st.metric(label=param.replace('_', ' ').title(), value=f"{value:.2f}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Valori Attuali")
                    for param, value in results['base_prediction'].items():
                        st.metric(
                            label=param.replace('_', ' ').title(),
                            value=f"{value:.2f}"
                        )
                
                with col2:
                    st.markdown("### Valori Previsti")
                    for param, value in results['modified_prediction'].items():
                        diff = results['differences'][param]
                        st.metric(
                            label=param.replace('_', ' ').title(),
                            value=f"{value:.2f}",
                            delta=f"{diff:+.2f}"
                        )
                
                with col3:
                    st.markdown("### Analisi dell'Impatto")
                    if results.get('has_significant_changes', False):
                        impact_analysis = analyze_prediction_impact(results)
                        st.write(impact_analysis)
                    else:
                        st.info("Nessuna variazione significativa prevista")
                
            except Exception as e:
                st.error(f"Errore nella generazione della previsione: {str(e)}")
                st.exception(e)  # Mostra lo stack trace completo per debug

def create_prediction_plot(results):
    """Create interactive plot with predictions"""
    try:
        # Create subplot for each parameter
        fig = make_subplots(
            rows=len(results['base_prediction']),
            cols=1,
            subplot_titles=[param.replace('_', ' ').title() for param in results['base_prediction'].keys()]
        )
        
        for i, (param, base_value) in enumerate(results['base_prediction'].items(), 1):
            mod_value = results['modified_prediction'][param]
            
            fig.add_trace(
                go.Scatter(
                    x=['Current', 'Predicted'],
                    y=[base_value, mod_value],
                    name=param,
                    mode='lines+markers'
                ),
                row=i,
                col=1
            )
        
        fig.update_layout(
            height=200 * len(results['base_prediction']),
            showlegend=False,
            title_text="Prediction Comparison"
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating prediction plot: {str(e)}")
        return None

def analyze_prediction_impact(results):
    """Analizza e formatta l'impatto delle variazioni previste"""
    try:
        impact_text = []
        
        for param, diff in results['differences'].items():
            param_name = param.replace('_', ' ').title()
            if abs(diff) > 0:
                change = "aumento" if diff > 0 else "diminuzione"
                percentage = (diff / results['base_prediction'][param]) * 100
                impact_text.append(
                    f"**{param_name}**: {abs(percentage):.1f}% di {change}\n"
                    f"Impatto: {'Alto' if abs(percentage) > 10 else 'Medio' if abs(percentage) > 5 else 'Basso'}"
                )
        
        return "\n\n".join(impact_text) if impact_text else "Nessuna variazione significativa prevista"
        
    except Exception as e:
        logger.error(f"Errore nell'analisi dell'impatto: {str(e)}")
        return "Errore nell'analisi dell'impatto"

# Renderizza la chat nel sidebar
def render_chat():
    st.sidebar.markdown("---")
    st.sidebar.header("💬 Assistente Conversazionale")
    
    # Add chat styles
    st.sidebar.markdown(get_chat_styles(), unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        role_class = "assistant-message" if message["role"] == "assistant" else "user-message"
        with st.sidebar.container():
            st.markdown(f'<div class="chat-message {role_class}">{message["content"]}</div>', unsafe_allow_html=True)
            
            # Show sources in a collapsible box if available for assistant messages
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.sidebar.expander("📚 Fonti consultate"):
                    # Format sources as a bulleted list
                    sources_list = "\n".join([f"- {source}" for source in message["sources"]])
                    st.markdown(sources_list)
    
    # Chat input
    with st.sidebar.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Scrivi un messaggio:", key="user_message")
        submitted = st.form_submit_button("Invia")
        
        if submitted and user_input:
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            # Get assistant response
            with st.spinner("L'assistente sta elaborando..."):
                # Make sure the chat assistant has the current simulated time
                st.session_state.chat_assistant.set_current_time(st.session_state.simulated_datetime)
                response = asyncio.run(st.session_state.chat_assistant.chat(user_input))
                
            # Add assistant response to chat history
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": response["answer"],
                "sources": response["sources"]
            })
            
            # Force refresh to show new messages
            st.rerun()
    
    # Add button to clear chat history
    if st.sidebar.button("🗑️ Cancella Conversazione"):
        st.session_state.chat_messages = []
        st.session_state.chat_assistant.clear_conversation_history()
        st.rerun()

def main():
    st.title("🔧 Dashboard Monitoraggio Compressori")
    
    # Sidebar per la selezione del compressore e il time control
    st.sidebar.header("Impostazioni")
    
    # Compressor selection
    compressors = get_compressors()
    selected_compressor = st.sidebar.selectbox(
        "Seleziona Compressore",
        compressors
    )
    
    # Time Control Card
    with st.sidebar.expander("🕒 Simulazione orario di sistema", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            new_date = st.date_input(
                "Data",
                value=st.session_state.simulated_datetime.date(),
                min_value=datetime(2024, 3, 1).date(),
                max_value=datetime(2024, 6, 30).date()
            )
        with col2:
            # Creiamo una lista di ore intere (00:00, 01:00, ..., 23:00)
            hours = [f"{h:02d}:00" for h in range(24)]
            current_hour = st.session_state.simulated_datetime.strftime("%H:00")
            
            selected_hour = st.selectbox(
                "Ora",
                hours,
                index=hours.index(current_hour)
            )
            # Convertiamo l'ora selezionata in oggetto time
            new_time = datetime.strptime(selected_hour, "%H:%M").time()
        
        # Aggiorna il datetime simulato quando viene modificato
        new_datetime = datetime.combine(new_date, new_time)
        if new_datetime != st.session_state.simulated_datetime:
            # Store the old datetime for potential data reloading
            old_datetime = st.session_state.simulated_datetime
            st.session_state.simulated_datetime = new_datetime
            
            # If we're moving forward in time, ensure data is up to date
            if new_datetime > old_datetime:
                logger.info(f"Time moved forward: {old_datetime} -> {new_datetime}")
                # Clear chat assistant's context if time changed
                st.session_state.chat_assistant.clear_conversation_history()
                # Load data up to the new simulated time
                st.session_state.chat_assistant.load_data_up_to(new_datetime)
            
            # Sync assistants' timestamps
            st.session_state.assistant.set_current_time(new_datetime)
            st.session_state.chat_assistant.set_current_time(new_datetime)
            
            # Force cache clear to update all data
            st.cache_data.clear()
            st.rerun()
    
    # Add Chat Interface in Sidebar
    render_chat()
    
    # Renderizza la dashboard dei compressori
    render_compressor_dashboard()
    
    # Layout principale con tabs - aggiornato rimuovendo il tab Q&A
    tab_names = ["📊 Panoramica", "🔍 Analisi Anomalie", "📈 Metriche", "🔎 Ricerca Storico Anomalie", "🎯 Simulatore Predizioni"]
    tabs = st.tabs(tab_names)
    
    # Tab 1: Panoramica
    with tabs[0]:
        st.header("Panoramica Compressore")
        st.subheader(f"Ora Corrente: {st.session_state.simulated_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # KPIs in colonne
        col1, col2, col3, col4 = st.columns(4)
        kpis = calculate_kpis(selected_compressor)
        
        with col1:
            st.metric("Anomalie negli ultimi 7gg", kpis['anomaly_count'])
        with col2:
            st.metric("Fattore di Potenza Medio negli ultimi 7gg", f"{kpis['avg_cosphi']:.3f}")
        with col3:
            st.metric("Affidabilità negli ultimi 7gg", f"{kpis['reliability']:.1%}")
        with col4:
            st.metric("Tempo di Funzionamento (ore) negli ultimi 7gg", f"{kpis['total_runtime']:.1f}")
        
        # Grafici recenti
        st.subheader("Misurazioni Recenti")
        measurements = get_measurements(
            selected_compressor,
            st.session_state.simulated_datetime - timedelta(days=1),  # Usiamo la data simulata
            st.session_state.simulated_datetime,
            _sim_time=st.session_state.simulated_datetime  # Per cache invalidation
        )
        
        if not measurements.empty:
            fig = px.line(measurements, x='timestamp', y='cosphi',
                         title='Andamento Fattore di Potenza')
            st.plotly_chart(fig, use_container_width=True)
            
            # Aggiungi spiegazione collassabile
            with st.expander("📊 Informazioni sul grafico", expanded=False):
                st.markdown("""
                    Il grafico mostra l'andamento del fattore di potenza (cos φ) nel tempo. 
                    Un cos φ più vicino a 1 indica un utilizzo efficiente dell'energia elettrica.
                """)
        else:
            st.info("Nessuna misurazione recente disponibile")
    
    # Tab 2: Analisi Anomalie
    with tabs[1]:
        st.header("Analisi Anomalie")
        
        # Controlla se c'è un'analisi pendente e mostrala per prima
        if 'analysis_timestamp' in st.session_state and 'analysis_compressor' in st.session_state:
            with st.spinner("Analisi anomalie in corso..."):
                analysis = asyncio.run(analyze_anomaly(
                    st.session_state.analysis_timestamp,
                    st.session_state.analysis_compressor
                ))
                st.markdown("### Analisi Dettagliata dell'Anomalia")
                st.markdown(analysis)
                # Aggiungi un separatore
                st.markdown("---")
                # Pulisci i dati dell'analisi dopo averla mostrata
                del st.session_state.analysis_timestamp
                del st.session_state.analysis_compressor
        
        # Timeline delle anomalie (il resto del codice rimane uguale)
        anomalies = get_anomalies(
            selected_compressor,
            st.session_state.simulated_datetime - timedelta(days=30),
            st.session_state.simulated_datetime,
            _sim_time=st.session_state.simulated_datetime
        )
        
        if not anomalies.empty:
            # Filtra solo le vere anomalie
            true_anomalies = anomalies[anomalies['is_anomaly'] == True]
            
            if not true_anomalies.empty:
                st.subheader("Linea Temporale Anomalie")
                fig = px.scatter(true_anomalies, x='timestamp', y='error_value',
                               color='parameter',
                               title='Anomaly Detection Timeline')
                st.plotly_chart(fig, use_container_width=True)
                
                # Aggiungi spiegazione collassabile
                with st.expander("📊 Informazioni sul grafico", expanded=False):
                    st.markdown("""
                        Il grafico a dispersione mostra le anomalie rilevate nel tempo.
                        Ogni punto rappresenta un'anomalia, con colori diversi per parametri diversi.
                        L'altezza del punto indica la gravità dell'anomalia.
                    """)
                
                # Raggruppa le anomalie per timestamp per l'analisi multipla
                grouped_anomalies = true_anomalies.groupby('timestamp')
                
                # Crea lista di timestamp unici con anomalie
                unique_timestamps = sorted(true_anomalies['timestamp'].unique())
                timestamp_options = []
                
                for ts in unique_timestamps:
                    anomalies_at_ts = true_anomalies[true_anomalies['timestamp'] == ts]
                    params = sorted(anomalies_at_ts['parameter'].unique())
                    timestamp_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                    param_str = ", ".join(params)
                    display_str = f"{timestamp_str} - Parameters: {param_str}"
                    timestamp_options.append({"label": display_str, "value": timestamp_str})
                
                # Selezione anomalia specifica con dettagli dei parametri
                selected_option = st.selectbox(
                    "Seleziona Anomalia per Analisi Dettagliata",
                    options=[opt["value"] for opt in timestamp_options],
                    format_func=lambda x: next(opt["label"] for opt in timestamp_options if x == opt["value"])
                )
                
                if selected_option and st.button("Analizza Anomalia Selezionata"):
                    with st.spinner("Analisi in corso..."):
                        analysis_timestamp = pd.to_datetime(selected_option)
                        analysis = asyncio.run(analyze_anomaly(
                            analysis_timestamp,
                            selected_compressor
                        ))
                        
                        # Mostra prima il testo dell'analisi
                        if analysis and not analysis.startswith("Error"):
                            st.markdown("### 📊 Analisi Dettagliata")
                            st.markdown(analysis)
                            
                            # Mostra le sezioni principali solo se ci sono
                            sections = analysis.split('\n\n')
                            for section in sections:
                                if section.strip():  # Verifica che la sezione non sia vuota
                                    if section.startswith('1. ANOMALIE ATTUALI'):
                                        st.markdown("#### Anomalie Attuali")
                                        st.markdown(section.replace('1. ANOMALIE ATTUALI:', '').strip())
                                    elif section.startswith('2. CORRELAZIONI STORICHE'):
                                        st.markdown("#### Correlazioni Storiche")
                                        st.markdown(section.replace('2. CORRELAZIONI STORICHE:', '').strip())
                                    elif section.startswith('3. ANALISI E RACCOMANDAZIONI'):
                                        st.markdown("#### Analisi e Raccomandazioni")
                                        st.markdown(section.replace('3. ANALISI E RACCOMANDAZIONI:', '').strip())
                            
                            # Box collassabile per le fonti alla fine
                            with st.expander("📚 Fonti Consultate"):
                                st.markdown("""
                                    - Database Anomalie
                                    - Storico Manutenzioni
                                    - Storico Guasti
                                    - Dati Operativi
                                    - Knowledge Base Tecnica
                                """)
                        else:
                            st.error(f"Errore nell'analisi: {analysis}")

            else:
                st.info("Nessuna anomalia trovata nel periodo selezionato")
        else:
            st.info("Nessuna misurazione trovata negli ultimi 30 giorni")
    
    # Tab 3: Metrics (precedentemente Tab 4)
    with tabs[2]:
        st.header("Metriche di Prestazione")
        
        timeframe = st.selectbox(
            "Seleziona Periodo",
            ['24h', '7g', '30g']
        )
        
        # Calcola le metriche usando gli stessi parametri dell'overview
        metrics = calculate_kpis(
            selected_compressor,
            timeframe.replace('g', 'd')  # Converti il formato da 7g a 7d per compatibilità
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Metriche di Affidabilità")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics['reliability'] * 100,
                number={'suffix': "%", 'font': {'size': 26}},
                title={'text': "Indice di Affidabilità", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 60], 'color': "#ff9999"},  # Rosso chiaro
                        {'range': [60, 80], 'color': "#ffdd99"},  # Giallo
                        {'range': [80, 100], 'color': "#9fdf9f"}  # Verde chiaro
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=50, b=10),
                font={'size': 16}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📊 Informazioni sul grafico", expanded=False):
                st.markdown("""
                    L'indicatore mostra l'affidabilità complessiva del compressore:
                    - 🔴 0-60%: Affidabilità bassa
                    - 🟡 60-80%: Affidabilità media
                    - 🟢 80-100%: Affidabilità ottimale
                    
                    La linea rossa indica la soglia minima consigliata (90%).
                """)
            
        with col2:
            st.subheader("Distribuzione Anomalie")
            if not anomalies.empty:
                fig = px.histogram(anomalies, x='error_value',
                                 title='Error Value Distribution')
                st.plotly_chart(fig)
                
                # Aggiungi spiegazione collassabile
                with st.expander("📊 Informazioni sul grafico", expanded=False):
                    st.markdown("""
                        L'istogramma mostra la distribuzione degli errori rilevati.
                        Più la distribuzione è concentrata vicino allo zero, migliore è il funzionamento.
                    """)
            else:
                st.info("No anomaly data available for the selected timeframe")
    
    # Tab 4: Historical Search (precedentemente Tab 5)
    with tabs[3]:
        historical_search_tab()
    
    # Add the new prediction simulator tab
    with tabs[4]:
        prediction_simulator_tab()

if __name__ == "__main__":
    main()