# Agentic Financial Assistant - Complete Implementation
import os
import yaml
import smtplib
import requests
from typing import Dict, List, Optional, TypedDict, Annotated
from email.mime.text import MIMEText
from datetime import datetime

# LangChain imports
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_community.tools import DuckDuckGoSearchRun

# LangGraph imports
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
NEO4J_CONFIG = {
    "url": os.getenv("NEO4J_URI"),
    "username": os.getenv("NEO4J_USERNAME"), 
    "password": os.getenv("NEO4J_PASSWORD")
}

EMAIL_CONFIG = {
    "sender_email": "pdrudra.121@gmail.com",
    "app_password": "svij fcje mylm igsr",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587
}

LOAN_API_URL = "http://localhost:8000/loan-decision"

# Initialize components
graph = Neo4jGraph(**NEO4J_CONFIG)
llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# --- State Definition ---
class AgentState(TypedDict):
    user_input: str
    agent_choice: Optional[str]
    cypher_query: Optional[str]
    neo4j_result: Optional[List[Dict]]
    loan_data: Optional[Dict]
    api_response: Optional[Dict]
    email_content: Optional[str]
    email_recipients: Optional[List[str]]
    web_search_result: Optional[str]
    final_response: Optional[str]
    error: Optional[str]

# --- Load Prompts ---
def load_prompts(yaml_file_path: str = "prompts.yaml"):
    prompts = {
        "cypher_generation": {
            "template": """
            You are a Neo4j Cypher query generator. Given the database schema and user question, 
            generate a precise Cypher query.
            
            Schema: {schema}
            Question: {question}
            
            Generate only the Cypher query, nothing else.
            """
        },
        "review_analysis": {
            "template": """
            Analyze the following customer reviews and provide insights:
            
            Reviews: {reviews}
            
            Provide:
            1. Overall sentiment summary
            2. Key issues identified
            3. Specific recommendations for improvement
            4. Priority actions needed
            """
        },
        "loan_report": {
            "template": """
            Based on the loan decision and customer data, generate a comprehensive report:
            
            Customer Data: {customer_data}
            Loan Decision: {loan_decision}
            
            Provide:
            1. Decision summary (Approved/Rejected)
            2. Key factors considered
            3. Risk assessment
            4. Recommendations
            """
        },
        "email_templates": {
            "min_balance": """
            Subject: Important Notice: Low Account Balance
            
            Dear {customer_name},
            
            We noticed your account balance is below ₹5,000. To avoid any inconvenience:
            - Consider maintaining a minimum balance
            - Set up balance alerts
            - Contact us for assistance
            
            Best regards,
            SBI Team
            """,
            "product_ad": """
            Subject: Exciting New Banking Product - {product_name}
            
            Dear Valued Customer,
            
            We're excited to introduce our new {product_name}:
            {product_details}
            
            Benefits for you:
            {benefits}
            
            Contact us to learn more!
            
            Best regards,
            SBI Team
            """,
            "threat_alert": """
            Subject: Security Alert: Protect Yourself from {threat_name}
            
            Dear Valued Customer,
            
            We want to alert you about {threat_name}:
            
            What it is:
            {threat_description}
            
            How to protect yourself:
            {prevention_tips}
            
            Remember: We will never ask for your personal details over phone/email.
            
            Stay safe,
            SBI Security Team
            """
        }
    }
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
        print(f"cypher generated query : {cypher_query} ")
        return {"cypher": cypher_query, "result": result, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

def neo4j_retriever_tool(cypher_query: str) -> Dict:
    """Execute Cypher query and retrieve data"""
    try:
        result = graph.query(cypher_query)
        print(f"neo4j_retriever : {result} ")
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
            print(f"drafted mail : {response} ")
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
    review_tools = [tools[0]]  # cypher_generation
    return initialize_agent(
        tools=review_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

def create_general_agent():
    """General Database Query Agent"""
    general_tools = [tools[0]]  # cypher_generation
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
    loan_tools = [tools[0], tools[2]]  # cypher_generation, loan_api
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
    mail_tools = [tools[0], tools[1], tools[3]]  # cypher_generation, web_search, send_email
    return initialize_agent(
        tools=mail_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

# --- Workflow Nodes ---

def router_node(state: AgentState) -> AgentState:
    """Route user query to appropriate agent"""
    user_input = state["user_input"].lower()
    
    if any(word in user_input for word in ["review", "feedback", "rating", "complaint"]):
        agent_choice = "review_agent"
    elif any(word in user_input for word in ["loan", "credit", "borrow", "lending"]):
        agent_choice = "loan_agent"
    elif any(word in user_input for word in ["mail", "email", "send", "notify", "alert"]):
        agent_choice = "mail_agent"
    else:
        agent_choice = "general_agent"
    
    return {**state, "agent_choice": agent_choice}

def review_agent_node(state: AgentState) -> AgentState:
    """Handle review summarization requests"""
    try:
        # Get reviews from database
        cypher_result = cypher_generation_tool("MATCH (r:Review) RETURN r.review as review, r.date_submitted as date")
        
        if cypher_result["success"]:
            reviews = cypher_result["result"]
            
            # Analyze reviews using LLM
            review_text = "\n".join([f"Date: {r['date']}, Review: {r['review']}" for r in reviews])
            analysis_prompt = prompts_dict["review_analysis"]["template"].format(reviews=review_text)
            
            response = llm.invoke(analysis_prompt)
            final_response = response.content
        else:
            final_response = f"Error retrieving reviews: {cypher_result['error']}"
            
    except Exception as e:
        final_response = f"Review analysis failed: {str(e)}"
    
    return {**state, "final_response": final_response}

def general_agent_node(state: AgentState) -> AgentState:
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

def loan_agent_node(state: AgentState) -> AgentState:
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

def mail_agent_node(state: AgentState) -> AgentState:
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
def create_financial_assistant_workflow():
    """Create the main workflow using LangGraph"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("review_agent", review_agent_node)
    workflow.add_node("general_agent", general_agent_node)
    workflow.add_node("loan_agent", loan_agent_node)
    workflow.add_node("mail_agent", mail_agent_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router
    def route_to_agent(state):
        return state["agent_choice"]
    
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "review_agent": "review_agent",
            "general_agent": "general_agent", 
            "loan_agent": "loan_agent",
            "mail_agent": "mail_agent"
        }
    )
    
    # All agents end the workflow
    workflow.add_edge("review_agent", END)
    workflow.add_edge("general_agent", END)
    workflow.add_edge("loan_agent", END)
    workflow.add_edge("mail_agent", END)
    graph = workflow.compile()
    graph_png = graph.get_graph().draw_mermaid_png()

    # Save to file
    with open("output.png", "wb") as f:
        f.write(graph_png)

    return graph

# --- Main Execution ---
def run_financial_assistant(user_query: str) -> str:
    """Run the financial assistant with user query"""
    
    workflow = create_financial_assistant_workflow()
    
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
        result = workflow.invoke(initial_state)
        return result.get("final_response", "No response generated")
    except Exception as e:
        return f"Workflow execution failed: {str(e)}"

# --- Example Usage ---
if __name__ == "__main__":
    # Test queries
    test_queries = [
        # "Analyze all customer reviews and suggest improvements",
        # "How many customers joined this month?",
        "Should we approve loan for customer ID 123?",
        # "Send email to customers with balance less than 5000",
        # "Send threat alert about digital arrest scam to all customers"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        response = run_financial_assistant(query)
        print(f"Response: {response}")