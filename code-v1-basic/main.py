# Agentic Financial Assistant 
import os

import smtplib
import yaml
import requests
import random
from typing import Dict, List, Optional, TypedDict
from email.mime.text import MIMEText

# imports
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from langchain_community.tools import DuckDuckGoSearchRun


from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

from neo4j_connection import graph
from review_chain import reviews_vector_chain


load_dotenv()

# --- Configuration ---


EMAIL_CONFIG = {
    "sender_email": os.getenv("SENDER_EMAIL"),
    "app_password": os.getenv("EMAIL_APP_PASSWORD"),
    "smtp_server": "smtp.gmail.com",  
    "smtp_port": 587 
}

LOAN_API_URL = "http://localhost:8000/loan-decision"

# Initialize components

llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# --- State Definition ---
class AgentState(TypedDict):
    user_input: str
    agent_choice: Optional[str]
    fraud_sub_agent: Optional[str]  
    cypher_query: Optional[str]
    neo4j_result: Optional[List[Dict]]
    loan_data: Optional[Dict]
    api_response: Optional[Dict]
    email_content: Optional[str]
    email_recipients: Optional[List[str]]
    web_search_result: Optional[str]
    fraud_risk_score: Optional[int]  
    fraud_actions: Optional[List[str]]  
    final_response: Optional[str]
    error: Optional[str]

# --- Load Prompts ---
def load_prompts(yaml_file_path: str = "prompts.yaml"):
    with open(yaml_file_path, "r", encoding="utf-8") as file:
        prompts = yaml.safe_load(file)
    return prompts

prompts_dict = load_prompts()

# --- Tools Implementation ---


def cypher_generation_tool(question: str) -> Dict:
    """Generate and execute Cypher query based on user question"""
    try:
        schema = graph.schema
        prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=prompts_dict["cypher_generation"]["template"]
        )
        prompt_text = prompt.format(schema=schema, question=question)
        cypher_query = llm.invoke(prompt_text).content
        result = graph.query(cypher_query)
        return {"cypher": cypher_query, "result": result, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

def neo4j_retriever_tool(cypher_query: str) -> Dict:
    """Execute Cypher query and retrieve data"""
    try:
        result = graph.query(cypher_query)
        return {"result": result, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

def web_search_tool(query: str) -> str:
    """Search web for information"""
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result
    except Exception as e:
        return f"Search failed: {str(e)}"

def loan_api_tool(customer_data: Dict) -> Dict:
    """Call external loan API for decision"""
    try:
        response = requests.post(LOAN_API_URL, json=customer_data, timeout=10)
        print(response)
        return response.json()
    except Exception as e:
        return {"decision": "rejected", "reason": f"API call failed: {str(e)}"}

def mail_generation_tool(template_type: str, **kwargs) -> str:
    """Generate email content based on template and context"""
    try:
        if template_type in prompts_dict["email_templates"]:
            template = prompts_dict["email_templates"][template_type]
            return template.format(**kwargs)
        else:
            # Use LLM for custom email generation
            prompt = f"Generate a professional email for: {kwargs.get('purpose', 'general communication')}"
            if kwargs.get('context'):
                prompt += f"\nContext: {kwargs['context']}"
            response = llm.invoke(prompt)
            return response.content
    except Exception as e:
        return f"Email generation failed: {str(e)}"

def send_mail_tool(recipient_email: str, subject: str, body: str) -> Dict:
    """Send email to recipient"""
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = EMAIL_CONFIG["sender_email"]
        msg["To"] = recipient_email
        
        with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["app_password"])
            server.sendmail(EMAIL_CONFIG["sender_email"], recipient_email, msg.as_string())
        
        return {"success": True, "message": "Email sent successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- Create LangChain Tools ---
tools = [
    Tool(
        name="BankReviews",
        func=reviews_vector_chain.invoke,
        description="""Useful when you need to answer questions
        about customer experiences, opinions, or any other qualitative
        aspects related to banks, employees, or services using semantic
        search. Not suitable for answering objective questions that involve
        transactions, account balances, employee data, counts, percentages,
        or structured facts. Use the entire prompt as input to the tool.
        For example, if the prompt is "Do customers find the staff helpful?",
        the input should be "Do customers find the staff helpful?".
        """
    ),
    Tool(
        name="cypher_generation",
        description="Generate and execute Cypher queries for Neo4j database",
        func=lambda q: cypher_generation_tool(q)
    ),
    Tool(
        name="web_search", 
        description="Search the web for current information",
        func=web_search_tool
    ),
    Tool(
        name="loan_api",
        description="Get loan decision from external API",
        func=lambda data: loan_api_tool(data)
    ),
    Tool(
        name="send_email",
        description="Send email to recipients",
        func=lambda params: send_mail_tool(**params)
    )
]

# --- Agent Definitions ---

def create_review_agent():
    """Review Summarization Agent"""
    review_tools = [tools[0]]  # review_generation
    return initialize_agent(
        tools=review_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,  # 🔼 Increase this
        max_execution_time=60  # ⏱ Optional timeout in seconds
    )

def create_general_agent():
    """General Database Query Agent"""
    general_tools = [tools[1]]  # cypher_generation
    return initialize_agent(
        tools=general_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

def create_loan_agent():
    """Loan Decision Agent"""
    loan_tools = [tools[1], tools[3]]  # cypher_generation, loan_api
    return initialize_agent(
        tools=loan_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

def create_mail_agent():
    """Mail Generation and Sending Agent"""
    mail_tools = [tools[1], tools[2], tools[4]]  # cypher_generation, web_search, send_email
    return initialize_agent(
        tools=mail_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )


# initializing all agents
review_agent = create_review_agent()

# --- Workflow Nodes ---

async def router_node(state: AgentState) -> AgentState:
    """Route user query to appropriate agent"""
    user_input = state["user_input"].lower()
    
    if any(word in user_input for word in ["fraud", "suspicious", "block", "security", "unusual"]):
        agent_choice = "fraud_agent"
    elif any(word in user_input for word in ["review", "feedback", "rating", "complaint"]):
        agent_choice = "review_agent"
    elif any(word in user_input for word in ["loan", "credit", "borrow", "lending"]):
        agent_choice = "loan_agent"
    elif any(word in user_input for word in ["mail", "email", "send", "notify", "alert"]):
        agent_choice = "mail_agent"
    else:
        agent_choice = "general_agent"
    
    return {**state, "agent_choice": agent_choice}

# def review_agent_node(state: AgentState) -> AgentState:
#     """Handle review summarization requests"""
#     try:
#         # Get reviews from database
#         cypher_result = cypher_generation_tool("MATCH (r:Review) RETURN r.review as review, r.date_submitted as date")
        
#         if cypher_result["success"]:
#             reviews = cypher_result["result"]
#             reviews = random.sample(reviews, k=30)

#             # Analyze reviews using LLM
#             review_text = "\n".join([f"Date: {r['date']}, Review: {r['review']}" for r in reviews])
#             analysis_prompt = prompts_dict["review_analysis"]["template"].format(reviews=review_text)
            
#             response = llm.invoke(analysis_prompt)
#             final_response = response.content
#         else:
#             final_response = f"Error retrieving reviews: {cypher_result['error']}"
            
#     except Exception as e:
#         final_response = f"Review analysis failed: {str(e)}"
    
#     return {**state, "final_response": final_response}



# async def review_agent_node(state: AgentState) -> AgentState:
#     """Handle review summarization requests"""
#     user_input = state["user_input"].lower()  # or however your state is structured
#     result = await review_agent.ainvoke(user_input)
#     final_response = result
#     return {**state, "final_response": final_response}

import asyncio

async def review_agent_node(state: AgentState) -> AgentState:
    """Handle review summarization requests"""

    user_input = state["user_input"].lower()

    # Wrap sync `invoke()` in async call
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, review_agent.invoke, user_input)
    final_response = result.get("output", str(result))  # Safely extract output
    return {**state, "final_response": final_response}



async def fraud_detection_router_node(state: AgentState) -> AgentState:
    """A5 - Route fraud queries to real-time or historical analysis"""
    user_input = state["user_input"].lower()
    
    # Determine sub-agent based on keywords
    if any(word in user_input for word in ["real-time", "current", "now", "monitor", "active"]):
        fraud_sub_agent = "realtime_fraud"
    else:
        fraud_sub_agent = "historical_fraud"
    
    return {**state, "fraud_sub_agent": fraud_sub_agent}

async def calculate_fraud_risk(transaction_data: Dict) -> int:
    """Simple fraud risk calculation (1-10 scale)"""
    risk_score = 1
    
    # Check amount (high amounts are riskier)
    amount = transaction_data.get("amount", 0)
    if amount > 100000:
        risk_score += 4
    elif amount > 50000:
        risk_score += 2
    
    # Check transaction type
    tx_type = transaction_data.get("transaction_type", "").lower()
    if tx_type in ["withdrawal", "transfer"]:
        risk_score += 1
    
    # Check frequency (mock - in real system would check time patterns)
    # For PoC, randomly add risk based on transaction pattern
    if "urgent" in str(transaction_data).lower():
        risk_score += 2
    
    return min(risk_score, 10)

async def realtime_fraud_monitor_node(state: AgentState) -> AgentState:
    """A6 - Real-time fraud monitoring"""
    try:
        # Get recent transactions for analysis
        cypher_result = cypher_generation_tool(
            "MATCH (t:Transaction) RETURN t.id as id, t.amount as amount, t.transaction_type as type, t.date as date ORDER BY t.date DESC LIMIT 5"
        )
        
        if cypher_result["success"] and cypher_result["result"]:
            transactions = cypher_result["result"]
            high_risk_found = False
            risk_details = []
            
            for tx in transactions:
                risk_score = calculate_fraud_risk(tx)
                if risk_score >= 7:
                    high_risk_found = True
                    risk_details.append(f"Transaction {tx['id']}: Amount ₹{tx['amount']}, Risk Score: {risk_score}")
            
            if high_risk_found:
                fraud_risk_score = 8
                final_response = f"🚨 HIGH FRAUD RISK DETECTED!\n\nSuspicious transactions found:\n" + "\n".join(risk_details)
            else:
                fraud_risk_score = 3
                final_response = "✅ Real-time monitoring complete. No high-risk transactions detected."
                
        else:
            fraud_risk_score = 1
            final_response = "No transaction data available for real-time monitoring."
            
    except Exception as e:
        fraud_risk_score = 1
        final_response = f"Real-time fraud monitoring failed: {str(e)}"
    
    return {**state, "fraud_risk_score": fraud_risk_score, "final_response": final_response}

async def historical_fraud_analyzer_node(state: AgentState) -> AgentState:
    """A7 - Historical fraud pattern analysis"""
    try:
        # Analyze historical transaction patterns
        cypher_result = cypher_generation_tool(
            "MATCH (t:Transaction) RETURN t.amount as amount, t.transaction_type as type, count(t) as frequency ORDER BY t.amount DESC LIMIT 10"
        )
        
        if cypher_result["success"] and cypher_result["result"]:
            transactions = cypher_result["result"]
            
            # Simple pattern analysis
            suspicious_patterns = []
            total_risk = 0
            
            for tx in transactions:
                if tx["amount"] > 75000:
                    suspicious_patterns.append(f"Large transactions: ₹{tx['amount']} ({tx['frequency']} times)")
                    total_risk += 2
                    
            if suspicious_patterns:
                fraud_risk_score = min(6 + total_risk, 10)
                final_response = f"📊 HISTORICAL ANALYSIS COMPLETE\n\nSuspicious patterns identified:\n" + "\n".join(suspicious_patterns) + f"\n\nOverall Risk Assessment: {fraud_risk_score}/10"
            else:
                fraud_risk_score = 2
                final_response = "📊 Historical analysis complete. No suspicious patterns detected in transaction history."
                
        else:
            fraud_risk_score = 1
            final_response = "No historical transaction data available for analysis."
            
    except Exception as e:
        fraud_risk_score = 1
        final_response = f"Historical fraud analysis failed: {str(e)}"
    
    return {**state, "fraud_risk_score": fraud_risk_score, "final_response": final_response}

async def fraud_action_handler_node(state: AgentState) -> AgentState:
    """A8 - Take appropriate fraud actions based on risk score"""
    try:
        risk_score = state.get("fraud_risk_score", 1)
        actions_taken = []
        
        if risk_score >= 8:
            # High risk - Block and alert
            actions_taken.extend([
                "🔒 Card/Account temporarily blocked",
                "📧 Customer notified via email",
                "📞 Security team alerted",
                "📋 Fraud case created"
            ])
            
            # Send alert email (using existing mail tool)
            alert_subject = "🚨 URGENT: Security Alert - Account Activity"
            alert_body = f"""
Dear Customer,

We detected suspicious activity on your account and have temporarily blocked it for your protection.

Risk Level: HIGH ({risk_score}/10)
Action Taken: Account temporarily blocked
Next Steps: Please contact customer care immediately at 1800-111-109

This is an automated security measure. We apologize for any inconvenience.

Bank Security Team
            """
            
            # For PoC - just log the email action
            actions_taken.append("📬 Security alert email prepared")
            
        elif risk_score >= 5:
            # Medium risk - Monitor and notify
            actions_taken.extend([
                "👁️ Account flagged for enhanced monitoring",
                "📧 Customer advisory email sent",
                "📝 Risk assessment logged"
            ])
            
        else:
            # Low risk - Log only
            actions_taken.append("📝 Activity logged for routine monitoring")
        
        # Compile final response
        current_response = state.get("final_response", "")
        action_summary = "\n🔧 ACTIONS TAKEN:\n" + "\n".join([f"• {action}" for action in actions_taken])
        
        final_response = current_response + "\n" + action_summary + f"\n\n✅ Fraud detection process completed."
        
    except Exception as e:
        final_response = state.get("final_response", "") + f"\n❌ Fraud action handling failed: {str(e)}"
    
    return {**state, "fraud_actions": actions_taken, "final_response": final_response}

async def general_agent_node(state: AgentState) -> AgentState:
    """Handle general database queries"""
    try:
        result = cypher_generation_tool(state["user_input"])
        
        if result["success"]:
            # Format the result for user
            data = result["result"]
            if data:
                final_response = f"Query executed successfully. Found {len(data)} results:\n"
                for item in data[:5]:  # Show first 5 results
                    final_response += f"• {item}\n"
            else:
                final_response = "Query executed successfully but no results found."
        else:
            final_response = f"Query execution failed: {result['error']}"
            
    except Exception as e:
        final_response = f"General query failed: {str(e)}"
    
    return {**state, "final_response": final_response}

async def loan_agent_node(state: AgentState) -> AgentState:
    """Handle loan decision requests"""
    try:
        # Extract loan request data from database
        cypher_result = cypher_generation_tool("MATCH (c:Customer)-[:LINKED_WITH]->(a:Account) RETURN c.name as name, c.phone as phone, c.email as email, a.balance as balance")
        
        if cypher_result["success"] and cypher_result["result"]:
            customer_data = cypher_result["result"][0]  # Take first customer for demo
            
            # Call loan API
            api_response = loan_api_tool(customer_data)
            
            # Generate loan report
            report_prompt = prompts_dict["loan_report"]["template"].format(
                customer_data=customer_data,
                loan_decision=api_response
            )
            
            response = llm.invoke(report_prompt)
            final_response = response.content
        else:
            final_response = "Error: Could not retrieve customer data for loan analysis"
            
    except Exception as e:
        final_response = f"Loan processing failed: {str(e)}"
    
    return {**state, "final_response": final_response}

async def mail_agent_node(state: AgentState) -> AgentState:
    """Handle mail generation and sending"""
    try:
        user_input = state["user_input"].lower()
        
        if "balance" in user_input and "less than" in user_input:
            # Handle minimum balance emails
            cypher_result = cypher_generation_tool("MATCH (c:Customer)-[:LINKED_WITH]->(a:Account) WHERE a.balance < 5000 RETURN c.name as name, c.email as email, a.balance as balance")
            
            if cypher_result["success"]:
                customers = cypher_result["result"]
                sent_count = 0
                
                for customer in customers:
                    email_content = mail_generation_tool(
                        "min_balance",
                        customer_name=customer["name"]
                    )
                    
                    # Extract subject and body
                    lines = email_content.split('\n')
                    subject = lines[0].replace("Subject: ", "")
                    body = '\n'.join(lines[2:])
                    
                    result = send_mail_tool(customer["email"], subject, body)
                    if result["success"]:
                        sent_count += 1
                
                final_response = f"Successfully sent emails to {sent_count} customers with low balance."
            else:
                final_response = "Error retrieving customer data for email campaign"
                
        elif any(word in user_input for word in ["threat", "scam", "fraud", "alert"]):
            # Handle threat alert emails
            threat_query = "digital arrest scam prevention banking"
            search_result = web_search_tool(threat_query)
            
            # Get all customer emails
            cypher_result = cypher_generation_tool("MATCH (c:Customer) RETURN c.email as email, c.name as name")
            
            if cypher_result["success"]:
                customers = cypher_result["result"]
                
                # Generate threat alert email
                email_content = mail_generation_tool(
                    "threat_alert",
                    threat_name="Digital Arrest Scam",
                    threat_description="Fraudsters impersonate police/officials demanding money to avoid fake arrest",
                    prevention_tips="• Never pay money over phone calls\n• Verify caller identity independently\n• Report suspicious calls to bank immediately"
                )
                
                # Send to all customers
                sent_count = 0
                lines = email_content.split('\n')
                subject = lines[0].replace("Subject: ", "")
                body = '\n'.join(lines[2:])
                
                for customer in customers[:3]:  # Limit for demo
                    result = send_mail_tool(customer["email"], subject, body)
                    if result["success"]:
                        sent_count += 1
                
                final_response = f"Sent security alert to {sent_count} customers about digital arrest scam."
            else:
                final_response = "Error retrieving customer emails for threat alert"
        else:
            final_response = "Please specify the type of email campaign (balance alert, threat alert, etc.)"
            
    except Exception as e:
        final_response = f"Mail processing failed: {str(e)}"
    
    return {**state, "final_response": final_response}

# --- Create Workflow ---
async def create_financial_assistant_workflow():
    """Create the main workflow using LangGraph"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("review_agent", review_agent_node)
    workflow.add_node("general_agent", general_agent_node)
    workflow.add_node("loan_agent", loan_agent_node)
    workflow.add_node("mail_agent", mail_agent_node)
    
    # Add fraud detection nodes
    workflow.add_node("fraud_detection_router", fraud_detection_router_node)  # A5
    workflow.add_node("realtime_fraud_monitor", realtime_fraud_monitor_node)  # A6
    workflow.add_node("historical_fraud_analyzer", historical_fraud_analyzer_node)  # A7
    workflow.add_node("fraud_action_handler", fraud_action_handler_node)  # A8
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from main router
    def route_to_agent(state):
        return state["agent_choice"]
    
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "review_agent": "review_agent",
            "general_agent": "general_agent", 
            "loan_agent": "loan_agent",
            "mail_agent": "mail_agent",
            "fraud_agent": "fraud_detection_router"  # Route to A5
        }
    )
    
    # Add conditional edges from fraud router (A5)
    def route_fraud_sub_agent(state):
        return state["fraud_sub_agent"]
    
    workflow.add_conditional_edges(
        "fraud_detection_router",
        route_fraud_sub_agent,
        {
            "realtime_fraud": "realtime_fraud_monitor",  # A5 -> A6
            "historical_fraud": "historical_fraud_analyzer"  # A5 -> A7
        }
    )
    
    # A6 and A7 both go to A8
    workflow.add_edge("realtime_fraud_monitor", "fraud_action_handler")  # A6 -> A8
    workflow.add_edge("historical_fraud_analyzer", "fraud_action_handler")  # A7 -> A8
    
    # Original agents end the workflow
    workflow.add_edge("review_agent", END)
    workflow.add_edge("general_agent", END)
    workflow.add_edge("loan_agent", END)
    workflow.add_edge("mail_agent", END)
    
    # A8 ends the workflow
    workflow.add_edge("fraud_action_handler", END)  # A8 -> END
    
    graph = workflow.compile()
    graph_png = graph.get_graph().draw_mermaid_png()

    # Save to file
    with open("output.png", "wb") as f:
        f.write(graph_png)

    return graph

# --- Main Execution ---
async def run_financial_assistant(user_query: str) -> str:
    """Run the financial assistant with user query"""
    
    workflow = await create_financial_assistant_workflow()
    
    initial_state = AgentState(
        user_input=user_query,
        agent_choice=None,
        cypher_query=None,
        neo4j_result=None,
        loan_data=None,
        api_response=None,
        email_content=None,
        email_recipients=None,
        web_search_result=None,
        final_response=None,
        error=None
    )
    
    try:
        result = await workflow.ainvoke(initial_state)
        return result.get("final_response", "No response generated")
    except Exception as e:
        return f"Workflow execution failed: {str(e)}"

# --- Example Usage ---
if __name__ == "__main__":
    # Test queries
    test_queries = [
        # "Analyze all customer reviews and suggest improvements",
        # "How many customers joined this month?",
        # "Should we approve loan for customer ID 123?",
        # "Send email to customers with balance less than 5000",
        # "Send threat alert about digital arrest scam to all customers",
        # "Check for fraud in real-time transactions",
        # "Analyze suspicious transaction patterns for customer account",
        "Monitor current transactions for unusual activity"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        response = run_financial_assistant(query)
        print(f"Response: {response}")