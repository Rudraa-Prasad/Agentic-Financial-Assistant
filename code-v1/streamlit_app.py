import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List
#from main_2 import run_financial_assistant, create_financial_assistant_workflow
# Import your agent (assuming it's in the same directory)
try:
    from main_2 import run_financial_assistant
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    st.error("⚠️ Agent module not found. Please ensure 'paste.py' is in the same directory.")

# Page configuration
st.set_page_config(
    page_title="🏦 SBI Agentic Financial Assistant",
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
    🏦 SBI Agentic Financial Assistant
    <br><small>Multi-Agent Banking System with Fraud Detection</small>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # Agent Status
    st.markdown("### 🤖 Agent Status")
    if AGENT_AVAILABLE:
        st.success("✅ Agent System Online")
    else:
        st.error("❌ Agent System Offline")
    
    # Agent Statistics
    st.markdown("### 📊 Statistics")
    stats = st.session_state.agent_stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", stats['total_queries'])
        st.metric("Fraud Alerts", stats['fraud_detections'])
    with col2:
        st.metric("Emails Sent", stats['emails_sent'])
        st.metric("Loan Decisions", stats['loan_decisions'])
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    if st.button("🔍 Check Fraud Status", use_container_width=True):
        st.session_state.quick_query = "Monitor current transactions for unusual activity"
    
    if st.button("📧 Send Balance Alerts", use_container_width=True):
        st.session_state.quick_query = "Send email to customers with balance less than 5000"
    
    if st.button("📋 Analyze Reviews", use_container_width=True):
        st.session_state.quick_query = "Analyze all customer reviews and suggest improvements"
    
    if st.button("🏦 Loan Analysis", use_container_width=True):
        st.session_state.quick_query = "Should we approve loan for customer ID 123?"
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Interface", "📊 Dashboard", "🤖 Agent Flow", "📖 Documentation"])

with tab1:
    st.markdown("## 💬 Chat with Financial Assistant")
    
    # Display agent capabilities
    st.markdown("### 🎯 Available Agents:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
            <h4>🔍 Fraud Detection Agent</h4>
            <p>Real-time monitoring and historical analysis</p>
            <small>Keywords: fraud, suspicious, security, block</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="agent-card">
            <h4>📧 Mail Agent</h4>
            <p>Email campaigns and notifications</p>
            <small>Keywords: email, send, notify, alert</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="agent-card">
            <h4>🏦 Loan Agent</h4>
            <p>Loan decisions and credit analysis</p>
            <small>Keywords: loan, credit, borrow, lending</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="agent-card">
            <h4>⭐ Review Agent</h4>
            <p>Customer feedback analysis</p>
            <small>Keywords: review, feedback, rating</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="agent-card">
            <h4>🔧 General Agent</h4>
            <p>Database queries and general assistance</p>
            <small>Keywords: customer, account, balance</small>
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
    st.markdown("### 💭 Ask your question:")
    
    # Handle quick query from sidebar
    default_query = ""
    if 'quick_query' in st.session_state:
        default_query = st.session_state.quick_query
        del st.session_state.quick_query
    
    # Text input with example queries
    user_input = st.text_area(
        "Type your question here...",
        value=default_query,
        height=100,
        placeholder="Examples:\n• Check for fraud in real-time transactions\n• Send threat alert about digital arrest scam\n• How many customers joined this month?\n• Analyze customer reviews for improvements"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        send_button = st.button("🚀 Send", use_container_width=True, type="primary")
    
    with col2:
        example_button = st.button("💡 Examples", use_container_width=True)
    
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
    
    # Handle examples button
    if example_button:
        st.markdown("### 💡 Example Queries:")
        examples = [
            "🔍 **Fraud Detection**: 'Check for suspicious transactions in real-time'",
            "📧 **Email Campaign**: 'Send balance alert to customers with low balance'",
            "🏦 **Loan Analysis**: 'Should we approve loan for customer with ID 123?'",
            "⭐ **Review Analysis**: 'Analyze all customer reviews and suggest improvements'",
            "🔧 **General Query**: 'How many customers joined this month?'",
            "🚨 **Security Alert**: 'Send threat alert about digital arrest scam'",
            "📊 **Pattern Analysis**: 'Analyze historical fraud patterns'",
            "👥 **Customer Data**: 'Show me customer account details'"
        ]
        
        for example in examples:
            st.markdown(f"• {example}")

with tab2:
    st.markdown("## 📊 Dashboard")
    
    # Mock data for demonstration
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2196f3;">🏦</h3>
            <h2>1,247</h2>
            <p>Total Customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ff9800;">⚠️</h3>
            <h2>23</h2>
            <p>Fraud Alerts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #4caf50;">✅</h3>
            <h2>89</h2>
            <p>Loans Approved</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #9c27b0;">📧</h3>
            <h2>456</h2>
            <p>Emails Sent</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Agent Usage Over Time")
        
        # Mock time series data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        agent_usage = pd.DataFrame({
            'Date': dates,
            'Fraud Agent': [5, 8, 12, 3, 7, 15, 9, 6, 11, 8, 4, 13, 7, 9, 12, 6, 8, 10, 5, 14, 8, 7, 11, 9, 6, 12, 8, 10, 7, 9, 11],
            'Loan Agent': [12, 15, 8, 20, 18, 11, 16, 14, 9, 17, 13, 10, 19, 12, 8, 15, 11, 13, 16, 9, 14, 12, 18, 10, 15, 11, 13, 8, 16, 12, 14],
            'Mail Agent': [25, 30, 28, 35, 32, 29, 31, 33, 27, 36, 30, 28, 34, 31, 29, 33, 28, 32, 30, 35, 29, 31, 33, 28, 32, 30, 34, 29, 31, 33, 35],
            'Review Agent': [3, 5, 2, 7, 4, 6, 3, 5, 8, 4, 6, 2, 7, 5, 3, 6, 4, 5, 7, 3, 6, 4, 5, 7, 3, 6, 4, 8, 5, 3, 6]
        })
        
        fig = px.line(
            agent_usage, 
            x='Date', 
            y=['Fraud Agent', 'Loan Agent', 'Mail Agent', 'Review Agent'],
            title="Daily Agent Usage"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🥧 Query Distribution")
        
        query_types = ['General Queries', 'Fraud Detection', 'Loan Processing', 'Email Campaigns', 'Review Analysis']
        query_counts = [156, 89, 123, 201, 67]
        
        fig = px.pie(
            values=query_counts,
            names=query_types,
            title="Query Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity
    st.markdown("### 🕒 Recent Activity")
    
    recent_activities = pd.DataFrame({
        'Time': ['2 min ago', '5 min ago', '12 min ago', '18 min ago', '25 min ago'],
        'Agent': ['Fraud Agent', 'Mail Agent', 'Loan Agent', 'Review Agent', 'General Agent'],
        'Action': [
            '🚨 High-risk transaction detected and blocked',
            '📧 Sent 45 balance alert emails to customers',
            '✅ Approved loan application for customer #1234',
            '⭐ Analyzed 23 customer reviews - 4.2/5 avg rating',
            '🔍 Retrieved customer account information'
        ],
        'Status': ['Critical', 'Completed', 'Completed', 'Completed', 'Completed']
    })
    
    for _, row in recent_activities.iterrows():
        status_color = "🔴" if row['Status'] == 'Critical' else "🟢"
        st.markdown(f"""
        <div style="padding: 0.5rem; margin: 0.25rem 0; background: #f8f9fa; border-radius: 5px; border-left: 3px solid {'#f44336' if row['Status'] == 'Critical' else '#4caf50'};">
            <strong>{status_color} {row['Time']}</strong> - {row['Agent']}<br>
            {row['Action']}
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("## 🤖 Agent Flow Visualization")
    
    st.markdown("""
    ### 🔄 Multi-Agent Workflow
    
    This diagram shows how your financial assistant routes queries through different specialized agents:
    """)
    
    # Create a visual representation of the agent flow
    st.markdown("""
    ```
    🚀 START
       ⬇️
    🎯 ROUTER
       ⬇️
    ┌─────────────────────────────────────────────────────┐
    │  🔍 FRAUD AGENT → 🛡️ FRAUD ROUTER                    │
    │                    ⬇️                ⬇️              │
    │           📊 HISTORICAL      ⚡ REAL-TIME           │
    │           ANALYZER           MONITOR                │
    │                    ⬇️                ⬇️              │
    │                  🚨 FRAUD ACTION HANDLER            │
    ├─────────────────────────────────────────────────────┤
    │  🏦 LOAN AGENT → 💰 Loan Processing                 │
    ├─────────────────────────────────────────────────────┤
    │  📧 MAIL AGENT → ✉️ Email Campaigns                 │
    ├─────────────────────────────────────────────────────┤
    │  ⭐ REVIEW AGENT → 📝 Review Analysis                │
    ├─────────────────────────────────────────────────────┤
    │  🔧 GENERAL AGENT → 🗃️ Database Queries             │
    └─────────────────────────────────────────────────────┘
       ⬇️
    🏁 END
    ```
    """)
    
    # Agent descriptions
    st.markdown("### 🎯 Agent Responsibilities:")
    
    agents_info = {
        "🔍 Fraud Detection Agent": {
            "description": "Specialized in detecting and preventing fraudulent activities",
            "capabilities": [
                "Real-time transaction monitoring",
                "Historical pattern analysis", 
                "Risk scoring and assessment",
                "Automated blocking and alerts"
            ],
            "triggers": ["fraud", "suspicious", "security", "block", "unusual"]
        },
        "🏦 Loan Agent": {
            "description": "Handles loan applications and credit decisions",
            "capabilities": [
                "Credit score analysis",
                "Risk assessment",
                "Loan approval/rejection",
                "External API integration"
            ],
            "triggers": ["loan", "credit", "borrow", "lending"]
        },
        "📧 Mail Agent": {
            "description": "Manages email campaigns and notifications",
            "capabilities": [
                "Customer segmentation",
                "Email template generation",
                "Bulk email sending",
                "Threat alerts and notifications"
            ],
            "triggers": ["mail", "email", "send", "notify", "alert"]
        },
        "⭐ Review Agent": {
            "description": "Analyzes customer feedback and reviews",
            "capabilities": [
                "Sentiment analysis",
                "Review summarization",
                "Issue identification",
                "Improvement recommendations"
            ],
            "triggers": ["review", "feedback", "rating", "complaint"]
        },
        "🔧 General Agent": {
            "description": "Handles general database queries and assistance",
            "capabilities": [
                "Database queries",
                "Customer information retrieval",
                "Account balance checks",
                "General assistance"
            ],
            "triggers": ["customer", "account", "balance", "general queries"]
        }
    }
    
    for agent_name, info in agents_info.items():
        with st.expander(f"{agent_name}"):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown("**Capabilities:**")
            for capability in info['capabilities']:
                st.markdown(f"• {capability}")
            st.markdown("**Trigger Keywords:**")
            st.markdown(f"• {', '.join(info['triggers'])}")

with tab4:
    st.markdown("## 📖 Documentation")
    
    st.markdown("""
    ### 🎯 Getting Started
    
    Welcome to the SBI Agentic Financial Assistant! This system uses multiple specialized AI agents to handle different banking tasks efficiently.
    
    #### 🚀 How to Use:
    
    1. **Choose Your Query Type**: The system automatically routes your question to the right agent based on keywords
    2. **Ask Natural Questions**: Use conversational language - no need for specific commands
    3. **Monitor Results**: Check the dashboard for real-time statistics and activity
    
    #### 🔧 System Requirements:
    
    - Python 3.8+
    - Required environment variables:
      - `GROQ_API_KEY`: Your Groq API key
      - `NEO4J_URI`: Neo4j database URI
      - `NEO4J_USERNAME`: Neo4j username
      - `NEO4J_PASSWORD`: Neo4j password
    
    #### 📊 Features:
    
    - **Multi-Agent Architecture**: Specialized agents for different tasks
    - **Real-time Fraud Detection**: Advanced monitoring and alerting
    - **Email Automation**: Bulk email campaigns and notifications
    - **Loan Processing**: Automated credit decisions
    - **Review Analysis**: Customer feedback processing
    - **Interactive Dashboard**: Real-time metrics and visualizations
    
    #### 🛠️ Configuration:
    
    Make sure your `.env` file contains:
    ```
    GROQ_API_KEY=your_groq_api_key_here
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your_password
    ```
    
    #### 🚨 Troubleshooting:
    
    - **Agent Offline**: Check if `paste.py` is in the same directory
    - **Database Errors**: Verify Neo4j connection settings
    - **Email Issues**: Confirm SMTP configuration
    - **API Errors**: Check your Groq API key
    
    #### 📞 Support:
    
    For technical support or questions about the system:
    - Check the error messages in the chat interface
    - Review the agent logs in the terminal
    - Verify all environment variables are set correctly
    
    #### 🔄 Updates:
    
    The system supports real-time updates and can be extended with additional agents as needed.
    """)
    
    # System status
    st.markdown("### 🔧 System Status")
    
    status_items = [
        ("Agent System", "✅ Online" if AGENT_AVAILABLE else "❌ Offline"),
        ("Database Connection", "🔶 Checking..."),
        ("Email Service", "🔶 Checking..."),
        ("External APIs", "🔶 Checking..."),
    ]
    
    for item, status in status_items:
        st.markdown(f"• **{item}**: {status}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    🏦 SBI Agentic Financial Assistant v1.0 | Built with Streamlit & LangGraph
</div>
""", unsafe_allow_html=True)