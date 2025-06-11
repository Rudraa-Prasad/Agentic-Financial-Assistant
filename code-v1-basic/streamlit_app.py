import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List

try:
    from main_2 import run_financial_assistant
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    st.error("⚠️ Agent module not found. Please ensure 'paste.py' is in the same directory.")

# Page configuration
st.set_page_config(
    page_title=" Agentic Financial Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd, #bbdefb);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .agent-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        color: #c62828;
    }
    
    .success-message {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent_stats' not in st.session_state:
    st.session_state.agent_stats = {
        'total_queries': 0,
        'fraud_detections': 0,
        'emails_sent': 0,
        'loan_decisions': 0,
        'reviews_analyzed': 0
    }

# Main header
st.markdown("""
<div class="main-header">
    🏦 Agentic Financial Assistant
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Control Panel")
    
    # Agent Status
    st.markdown("### 🤖 Agent Status")
    if AGENT_AVAILABLE:
        st.success("✅ Agent System Online")
    else:
        st.error("❌ Agent System Offline")
    
    # Agent Statistics
    st.markdown("###  Statistics")
    stats = st.session_state.agent_stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", stats['total_queries'])
        st.metric("Fraud Alerts", stats['fraud_detections'])
    with col2:
        st.metric("Emails Sent", stats['emails_sent'])
        st.metric("Loan Decisions", stats['loan_decisions'])
    
    # Quick Actions
    st.markdown("###  Quick Actions")
    if st.button(" Check Fraud Status", use_container_width=True):
        st.session_state.quick_query = "Monitor current transactions for unusual activity"
    
    if st.button(" Send Balance Alerts", use_container_width=True):
        st.session_state.quick_query = "Send email to customers with balance less than 5000"
    
    if st.button(" Analyze Reviews", use_container_width=True):
        st.session_state.quick_query = "Analyze all customer reviews and suggest improvements"
    
    if st.button(" Loan Analysis", use_container_width=True):
        st.session_state.quick_query = "Should we approve loan for customer ID 123?"
    
    # Clear chat
    if st.button(" Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# # Main content area


st.markdown("## 💬 Chat with Financial Assistant")

# Display agent capabilities
st.markdown("###  Available Agents:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="agent-card">
        <h4>🔍 Fraud Detection Agent</h4>
        <p>Real-time monitoring and historical analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="agent-card">
        <h4>📧 Mail Agent</h4>
        <p>Email product advertisment, threats, campaigns and notifications</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="agent-card">
        <h4>🏦 Loan Agent</h4>
        <p>Loan decisions and credit analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="agent-card">
        <h4>⭐ Review Agent</h4>
        <p>Customer feedback analysis</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="agent-card">
        <h4>🔧 General Agent</h4>
        <p>Database queries and general assistance</p>
    </div>
    """, unsafe_allow_html=True)

# Chat interface
st.markdown("---")

# Display chat history
chat_container = st.container()

with chat_container:
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

# Input area
st.markdown("### Ask your question:")

# Handle quick query from sidebar
default_query = ""
if 'quick_query' in st.session_state:
    default_query = st.session_state.quick_query
    del st.session_state.quick_query

# Text input with example queries
user_input = st.text_area(
    "",
    value=default_query,
    height=100,
    placeholder="Type your question here..."
)

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    send_button = st.button("Send", use_container_width=True, type="primary")


# Handle send button
if send_button and user_input.strip() and AGENT_AVAILABLE:
    # Add user message to chat
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })
    
    # Show processing message
    with st.spinner("🤖 Agent processing your request..."):
        try:
            # Call your agent
            response = run_financial_assistant(user_input)
            
            # Update statistics based on query type
            st.session_state.agent_stats['total_queries'] += 1
            
            query_lower = user_input.lower()
            if any(word in query_lower for word in ["fraud", "suspicious", "security"]):
                st.session_state.agent_stats['fraud_detections'] += 1
            elif any(word in query_lower for word in ["email", "send", "mail"]):
                st.session_state.agent_stats['emails_sent'] += 1
            elif any(word in query_lower for word in ["loan", "credit", "borrow"]):
                st.session_state.agent_stats['loan_decisions'] += 1
            elif any(word in query_lower for word in ["review", "feedback", "rating"]):
                st.session_state.agent_stats['reviews_analyzed'] += 1
            
            # Add assistant response to chat
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing request: {str(e)}")

elif send_button and not AGENT_AVAILABLE:
    st.error("❌ Agent system is not available. Please check the configuration.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
     Built by Rudra Prasad
</div>
""", unsafe_allow_html=True)