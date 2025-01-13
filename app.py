import os
from dotenv import load_dotenv
import pickle
import pandas as pd
import utils as ut
import numpy as np
import streamlit as st
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from investment_strategy import MarketStrategy
import plotly.express as px
from datetime import datetime, timedelta
from streamlit_chat import message
import time

# Configuration de la page Streamlit avec un thème personnalisé
st.set_page_config(
    page_title="Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour améliorer le design
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e4057;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1a2634;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    .chart-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    .header-container {
        padding: 2rem;
        background: linear-gradient(135deg, #2e4057 0%, #1a2634 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .input-container {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    .stTextInput>div>div>input {
        background-color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #2e4057;
        color: white;
    }
    .chat-message.bot {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# En-tête élégant
st.markdown(
    """
    <div class="header-container">
        <h1>📈 Market Prediction Analysis</h1>
        <p style='color: #e0e0e0; font-style: italic;'>Advanced Financial Market Analysis Tool</p>
    </div>
    """, 
    unsafe_allow_html=True
)

load_dotenv()  # Charge les variables depuis .env

class CustomRandomForest(RandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return super().predict_proba(X)

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ['GROQ_API_KEY']
)

random_forest_model = None

try:
    with open('models/tunned_rf_model.pkl', 'rb') as f:
        random_forest_model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'models/tunned_rf_model.pkl' exists.")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

def load_model(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    filepath = os.path.join(model_dir, filename)
    try:
        with open(filepath, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Warning: Model file {filename} not found")
        return None

def prepare_input(XAU_BGNL, VIX, LUMSTRUU, LMBITR, LUACTRUU, LUAGTRUU, USGG10YR):
    input_dict = {
        'XAU BGNL': XAU_BGNL,
        'VIX': VIX,
        'LUMSTRUU': LUMSTRUU,
        'LMBITR': LMBITR,
        'LUACTRUU': LUACTRUU,
        'LUAGTRUU': LUAGTRUU,
        'USGG10YR': USGG10YR
    }
    
    # Créer le DataFrame avec les colonnes dans le bon ordre
    input_df = pd.DataFrame([input_dict])
    
    return input_df, input_dict

# Initialisez la stratégie
strategy = MarketStrategy()

def make_predictions(input_df):
    try:
        if random_forest_model is None:
            st.error("Model not loaded properly. Please check the model file.")
            return None
            
        probability = random_forest_model.predict_proba(input_df)[0][1]
        probabilities = {
            'Random Forest': probability
        }

        avg_probability = np.mean(list(probabilities.values()))
        
        # Convert input_df to dict for the indicators
        indicators = input_df.iloc[0].to_dict()
        
        # Generate trading signal with both required arguments
        signal = strategy.generate_signals(avg_probability, indicators)

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Display trading metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Signal", signal['action'], 
                     delta=signal['confidence'])
        with col2:
            st.metric("Position Size", f"${signal['size']:,.0f}", 
                     delta=f"{(signal['size']/strategy.current_capital)*100:.1f}% of Capital")
        with col3:
            st.metric("Risk Levels", 
                     f"SL: {signal['stop_loss']*100:.1f}%", 
                     delta=f"TP: {signal['take_profit']*100:.1f}%")

        # Market Analysis
        st.markdown("### 📈 Market Analysis")
        st.markdown(signal['analysis'])
        
        # Market Context
        st.markdown("### 🔍 Market Context")
        context_col1, context_col2, context_col3 = st.columns(3)
        with context_col1:
            st.metric("VIX Level", signal['market_context']['vix_level'])
        with context_col2:
            st.metric("Interest Rate", signal['market_context']['interest_rate_environment'])
        with context_col3:
            st.metric("Gold Trend", signal['market_context']['gold_trend'])

        # Trading Rationale
        st.markdown("### 💡 Trading Rationale")
        st.markdown(signal['rationale'])

        st.markdown('</div>', unsafe_allow_html=True)
        
        return avg_probability, signal

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def explain_prediction(probability, input_dict):
    prompt = f"""As a financial market expert, analyze the situation:

    Prediction: {probability:.2%} probability of market increase
    Current Indicators: {input_dict}

    1. Market Analysis:
    - Explain current market conditions
    - Identify key risk factors
    
    2. Investment Recommendation:
    - Propose asset allocation strategy
    - Suggest entry/exit levels
    
    3. Risk Management:
    - Recommend stop-loss levels
    - Identify potential catalysts
    
    Please provide actionable insights."""

    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{"role": "user", "content": prompt}]
    )

    return raw_response.choices[0].message.content

# Interface principale
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown("### Market Indicators")

col1, col2 = st.columns(2)

with col1:
    XAU_BGNL = st.number_input('🏆 Gold Price (XAU BGNL)', 
                              value=1800.0,
                              help="Current gold price in USD")
    VIX = st.number_input('📊 Volatility Index (VIX)', 
                         value=20.0,
                         help="Market fear gauge")
    LUMSTRUU = st.number_input('🌍 Global Bond Index', 
                              value=100.0,
                              help="Global bond market performance")
    LMBITR = st.number_input('📈 Market Index', 
                            value=100.0,
                            help="Overall market performance")

with col2:
    LUACTRUU = st.number_input('💰 Aggregate Bond Index', 
                              value=100.0,
                              help="US bond market indicator")
    LUAGTRUU = st.number_input('📋 Treasury Index', 
                              value=100.0,
                              help="Treasury market performance")
    USGG10YR = st.number_input('🏛️ US 10Y Treasury Yield', 
                              value=2.0,
                              help="Benchmark interest rate")

st.markdown('</div>', unsafe_allow_html=True)

# Bouton d'analyse avec style
if st.button("🔍 Analyze Market"):
    with st.spinner('Analyzing market conditions...'):
        input_df, input_dict = prepare_input(
            XAU_BGNL=XAU_BGNL,
            VIX=VIX,
            LUMSTRUU=LUMSTRUU,
            LMBITR=LMBITR,
            LUACTRUU=LUACTRUU,
            LUAGTRUU=LUAGTRUU,
            USGG10YR=USGG10YR
        )
        
        result = make_predictions(input_df)
        
        if result is not None:
            avg_probability, signal = result
            
            # Afficher la stratégie
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            
            # Signal de trading
            st.markdown("### 📊 Trading Signal")
            signal_col1, signal_col2, signal_col3 = st.columns(3)
            with signal_col1:
                st.metric("Action", signal['action'], 
                         delta=signal['confidence'])
            with signal_col2:
                st.metric("Position Size", f"${signal['size']:,.0f}", 
                         delta=f"{(signal['size']/strategy.current_capital)*100:.1f}%")
            with signal_col3:
                st.metric("Risk Level", 
                         f"SL: {signal['stop_loss']*100:.1f}%", 
                         delta=f"TP: {signal['take_profit']*100:.1f}%")
            
            # Analyse du marché
            st.markdown("### 📈 Market Analysis")
            st.markdown(signal['analysis'])
            
            # Portfolio Stats
            st.markdown("### 📊 Portfolio Statistics")
            stats = strategy.get_portfolio_stats()
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("Capital", f"${stats['current_capital']:,.0f}", 
                         f"{stats['total_return']*100:.1f}%")
            with stats_col2:
                st.metric("Total Trades", stats['num_trades'])
            
            st.markdown('</div>', unsafe_allow_html=True)

# Chatbot section
st.markdown("---")
st.markdown("### 💬 AI Investment Advisor")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create two columns - one for chat, one for suggested questions
chat_col, suggest_col = st.columns([2, 1])

with chat_col:
    # Chat input
    user_input = st.text_input("Ask about market strategy, risks, or portfolio:", key="user_input")
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
    
    if user_input:
        with st.spinner('Analyzing your question...'):
            # Get chatbot response
            context = {
                'portfolio_value': strategy.current_capital,
                'total_trades': len(strategy.trades),
                'recent_performance': sum(strategy.returns[-30:]) if strategy.returns else 0
            }
            
            prompt = f"""As an AI Investment Advisor, respond to this question. 
            
            Current Portfolio Context:
            - Portfolio Value: ${context['portfolio_value']:,.2f}
            - Total Trades: {context['total_trades']}
            - Recent Performance: {context['recent_performance']:.1%}
            
            User Question: {user_input}
            
            Provide a clear, professional response focusing on investment strategy and market analysis.
            """
            
            response = client.chat.completions.create(
                model="llama-3.2-3b-preview",
                messages=[{"role": "user", "content": prompt}]
            )
            
            bot_response = response.choices[0].message.content
            st.session_state.chat_history.append((user_input, bot_response))
    
    # Display chat history
    st.markdown("#### Chat History")
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        message(user_msg, is_user=True, key=f"user_msg_{i}")
        message(bot_msg, is_user=False, key=f"bot_msg_{i}")

with suggest_col:
    st.markdown("#### 💡 Suggested Questions")
    questions = [
        "What is the current market strategy?",
        "How should I manage risk?",
        "Explain the latest signals",
        "Key indicators to watch?",
        "Portfolio performance?"
    ]
    
    for q in questions:
        if st.button(q, key=f"btn_{q}"):
            with st.spinner('Analyzing...'):
                context = {
                    'portfolio_value': strategy.current_capital,
                    'total_trades': len(strategy.trades),
                    'recent_performance': sum(strategy.returns[-30:]) if strategy.returns else 0
                }
                
                prompt = f"""As an AI Investment Advisor, respond to this question. 
                
                Current Portfolio Context:
                - Portfolio Value: ${context['portfolio_value']:,.2f}
                - Total Trades: {context['total_trades']}
                - Recent Performance: {context['recent_performance']:.1%}
                
                User Question: {q}
                
                Provide a clear, professional response focusing on investment strategy and market analysis.
                """
                
                response = client.chat.completions.create(
                    model="llama-3.2-3b-preview",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                bot_response = response.choices[0].message.content
                st.session_state.chat_history.append((q, bot_response))

    
