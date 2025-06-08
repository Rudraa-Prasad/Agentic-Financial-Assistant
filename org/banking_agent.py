import os
from typing import TypedDict, Literal, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import json
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


# Define tools for different agents
@tool
def get_account_balance(account_number: str) -> str:
    """Get account balance for a given account number."""
    # Simulate database lookup
    balances = {
        "ACC001": "$5,234.67",
        "ACC002": "$12,890.45", 
        "ACC003": "$856.23",
        "ACC004": "$45,678.90"
    }
    return balances.get(account_number, "Account not found")

@tool
def get_transaction_history(account_number: str, days: int = 30) -> str:
    """Get transaction history for an account for specified number of days."""
    # Simulate transaction data
    transactions = [
        f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}: {'Debit' if random.random() > 0.5 else 'Credit'} ${random.randint(10, 500)}.{random.randint(10,99)} - {'ATM Withdrawal' if random.random() > 0.7 else 'Online Purchase' if random.random() > 0.5 else 'Direct Deposit'}"
        for _ in range(random.randint(3, 8))
    ]
    return "\n".join(transactions[-5:])  # Return last 5 transactions

@tool
def freeze_unfreeze_account(account_number: str, action: str) -> str:
    """Freeze or unfreeze an account. Action should be 'freeze' or 'unfreeze'."""
    if action.lower() not in ['freeze', 'unfreeze']:
        return "Invalid action. Use 'freeze' or 'unfreeze'"
    return f"Account {account_number} has been successfully {action}d"

@tool
def process_transaction(from_account: str, to_account: str, amount: float) -> str:
    """Process a transaction between accounts."""
    if amount <= 0:
        return "Invalid amount"
    if amount > 10000:
        return "Transaction amount exceeds daily limit. Requires manager approval."
    return f"Transaction of ${amount:.2f} from {from_account} to {to_account} processed successfully. Transaction ID: TXN{random.randint(100000, 999999)}"

@tool
def check_transaction_status(transaction_id: str) -> str:
    """Check the status of a transaction."""
    statuses = ["Completed", "Pending", "Failed", "Under Review"]
    return f"Transaction {transaction_id} status: {random.choice(statuses)}"

@tool
def get_loan_eligibility(account_number: str, loan_amount: float) -> str:
    """Check loan eligibility for a customer."""
    # Simulate credit check
    credit_scores = {
        "ACC001": 720,
        "ACC002": 650,
        "ACC003": 580,
        "ACC004": 780
    }
    score = credit_scores.get(account_number, 600)
    
    if score >= 700:
        return f"Approved: Credit score {score}. Eligible for ${loan_amount:.2f} at 3.5% APR"
    elif score >= 650:
        return f"Conditional: Credit score {score}. Eligible for ${loan_amount * 0.8:.2f} at 4.5% APR"
    else:
        return f"Declined: Credit score {score}. Not eligible for requested amount"

@tool
def calculate_loan_payment(principal: float, rate: float, years: int) -> str:
    """Calculate monthly loan payment."""
    monthly_rate = rate / 100 / 12
    num_payments = years * 12
    if rate == 0:
        payment = principal / num_payments
    else:
        payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    return f"Monthly payment: ${payment:.2f} for ${principal:.2f} at {rate}% for {years} years"

# Define state
class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_agent: str
    customer_request: str
    account_number: str
    routing_decision: str
    agent_response: str
    retry_count: int
    requires_manager_approval: bool

# Group tools for each agent
account_tools = [get_account_balance, get_transaction_history, freeze_unfreeze_account]
transaction_tools = [process_transaction, check_transaction_status]
loan_tools = [get_loan_eligibility, calculate_loan_payment]

# Helper function to execute tools
def execute_tool(tool_name: str, tool_args: dict, available_tools: list):
    """Execute a tool by name with given arguments."""
    for tool in available_tools:
        if tool.name == tool_name:
            return tool.invoke(tool_args)
    return f"Tool {tool_name} not found"

# Orchestrator Agent
def orchestrator_agent(state: AgentState) -> AgentState:
    """Main orchestrator that routes requests to appropriate agents."""
    
    customer_request = state["customer_request"].lower()
    
    # Routing logic
    if any(keyword in customer_request for keyword in ["balance", "account", "freeze", "unfreeze", "statement", "history"]):
        routing_decision = "account_agent"
    elif any(keyword in customer_request for keyword in ["transfer", "transaction", "payment", "send money", "status"]):
        routing_decision = "transaction_agent"
    elif any(keyword in customer_request for keyword in ["loan", "credit", "mortgage", "borrow", "eligibility"]):
        routing_decision = "loan_agent"
    else:
        routing_decision = "account_agent"  # Default fallback
    
    prompt = f"""
    You are a banking orchestrator agent. A customer request has been received: "{state['customer_request']}"
    
    Based on the request, I'm routing this to: {routing_decision}
    
    Provide a brief acknowledgment and explain what will happen next.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        **state,
        "routing_decision": routing_decision,
        "current_agent": "orchestrator",
        "messages": state["messages"] + [response],
    }

# Account Management Agent
def account_agent(state: AgentState) -> AgentState:
    """Handles account-related requests."""
    
    prompt = f"""
    You are a banking account management specialist. Handle this customer request: "{state['customer_request']}"
    
    Account Number: {state.get('account_number', 'Not provided')}
    
    Available tools:
    - get_account_balance: Get current account balance
    - get_transaction_history: Get recent transactions
    - freeze_unfreeze_account: Freeze or unfreeze account
    
    Provide helpful assistance. If you need to use tools, I'll execute them for you.
    Be professional and thorough in your response.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Simulate tool usage based on request
    tool_results = []
    request_lower = state["customer_request"].lower()
    account_num = state.get("account_number", "ACC001")
    
    if "balance" in request_lower and account_num:
        balance = execute_tool("get_account_balance", {"account_number": account_num}, account_tools)
        tool_results.append(f"Balance check: {balance}")
    
    if "history" in request_lower or "transaction" in request_lower:
        history = execute_tool("get_transaction_history", {"account_number": account_num}, account_tools)
        tool_results.append(f"Transaction history:\n{history}")
    
    if "freeze" in request_lower:
        freeze_result = execute_tool("freeze_unfreeze_account", {"account_number": account_num, "action": "freeze"}, account_tools)
        tool_results.append(f"Freeze action: {freeze_result}")
    
    final_response = response.content
    if tool_results:
        final_response += "\n\nTool Results:\n" + "\n".join(tool_results)
    
    return {
        **state,
        "current_agent": "account_agent",
        "agent_response": final_response,
        "messages": state["messages"] + [AIMessage(content=final_response)],
    }

# Transaction Processing Agent
def transaction_agent(state: AgentState) -> AgentState:
    """Handles transaction-related requests."""
    
    prompt = f"""
    You are a banking transaction specialist. Handle this customer request: "{state['customer_request']}"
    
    Account Number: {state.get('account_number', 'Not provided')}
    
    Available tools:
    - process_transaction: Process transfers between accounts
    - check_transaction_status: Check status of existing transactions
    
    Be careful with large transactions and security protocols.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Simulate tool usage
    tool_results = []
    request_lower = state["customer_request"].lower()
    
    if "transfer" in request_lower or "send" in request_lower:
        # Simulate transaction processing
        result = execute_tool("process_transaction", {
            "from_account": state.get("account_number", "ACC001"),
            "to_account": "ACC999",
            "amount": 500.0
        }, transaction_tools)
        tool_results.append(f"Transaction result: {result}")
        
        # Check if requires manager approval
        if "manager approval" in result:
            state["requires_manager_approval"] = True
    
    if "status" in request_lower:
        status = execute_tool("check_transaction_status", {"transaction_id": "TXN123456"}, transaction_tools)
        tool_results.append(f"Status check: {status}")
    
    final_response = response.content
    if tool_results:
        final_response += "\n\nTransaction Details:\n" + "\n".join(tool_results)
    
    return {
        **state,
        "current_agent": "transaction_agent",
        "agent_response": final_response,
        "messages": state["messages"] + [AIMessage(content=final_response)],
    }

# Loan Services Agent (with conditional logic)
def loan_agent(state: AgentState) -> AgentState:
    """Handles loan-related requests with conditional approval logic."""
    
    prompt = f"""
    You are a banking loan specialist. Handle this customer request: "{state['customer_request']}"
    
    Account Number: {state.get('account_number', 'Not provided')}
    
    Available tools:
    - get_loan_eligibility: Check if customer qualifies for loan
    - calculate_loan_payment: Calculate monthly payments
    
    Follow bank policies for loan processing.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Tool usage with conditional logic
    tool_results = []
    request_lower = state["customer_request"].lower()
    account_num = state.get("account_number", "ACC001")
    
    if "loan" in request_lower or "eligibility" in request_lower:
        eligibility = execute_tool("get_loan_eligibility", {
            "account_number": account_num,
            "loan_amount": 50000.0
        }, loan_tools)
        tool_results.append(f"Eligibility check: {eligibility}")
        
        # Conditional logic based on eligibility
        if "Approved" in eligibility:
            # Calculate payment for approved loan
            payment = execute_tool("calculate_loan_payment", {
                "principal": 50000.0,
                "rate": 3.5,
                "years": 15
            }, loan_tools)
            tool_results.append(f"Payment calculation: {payment}")
        elif "Conditional" in eligibility:
            # Calculate payment for conditional approval
            payment = execute_tool("calculate_loan_payment", {
                "principal": 40000.0,  # Reduced amount
                "rate": 4.5,
                "years": 15
            }, loan_tools)
            tool_results.append(f"Conditional payment: {payment}")
            tool_results.append("Additional documentation required for final approval.")
        else:
            tool_results.append("Loan application declined. Consider improving credit score.")
    
    final_response = response.content
    if tool_results:
        final_response += "\n\nLoan Analysis:\n" + "\n".join(tool_results)
    
    return {
        **state,
        "current_agent": "loan_agent", 
        "agent_response": final_response,
        "messages": state["messages"] + [AIMessage(content=final_response)],
    }

# Conditional node for manager approval
def manager_approval_check(state: AgentState) -> Literal["manager_approval", "complete"]:
    """Conditional node to check if manager approval is needed."""
    if state.get("requires_manager_approval", False):
        return "manager_approval"
    return "complete"

# Manager approval node
def manager_approval_node(state: AgentState) -> AgentState:
    """Handle requests requiring manager approval."""
    
    approval_message = """
    🔴 MANAGER APPROVAL REQUIRED 🔴
    
    This request requires manager review due to:
    - High transaction amount
    - Security protocols
    - Policy compliance
    
    A manager will review this request within 2 business hours.
    Customer will be notified via email and SMS.
    
    Ticket ID: MGR-{random_id}
    Status: Pending Manager Review
    """.format(random_id=random.randint(10000, 99999))
    
    return {
        **state,
        "current_agent": "manager_approval",
        "agent_response": approval_message,
        "messages": state["messages"] + [AIMessage(content=approval_message)],
    }

# Route to specific agent
def route_to_agent(state: AgentState) -> Literal["account_agent", "transaction_agent", "loan_agent"]:
    """Route to the appropriate specialized agent."""
    return state["routing_decision"]

# Create the workflow graph
def create_banking_workflow():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.add_node("account_agent", account_agent)
    workflow.add_node("transaction_agent", transaction_agent)
    workflow.add_node("loan_agent", loan_agent)
    workflow.add_node("manager_approval", manager_approval_node)
    
    # Add edges
    workflow.set_entry_point("orchestrator")
    
    # Route from orchestrator to appropriate agent
    workflow.add_conditional_edges(
        "orchestrator",
        route_to_agent,
        {
            "account_agent": "account_agent",
            "transaction_agent": "transaction_agent", 
            "loan_agent": "loan_agent"
        }
    )
    
    # Conditional edges for manager approval
    workflow.add_conditional_edges(
        "account_agent",
        manager_approval_check,
        {
            "manager_approval": "manager_approval",
            "complete": END
        }
    )
    
    workflow.add_conditional_edges(
        "transaction_agent", 
        manager_approval_check,
        {
            "manager_approval": "manager_approval",
            "complete": END
        }
    )
    
    workflow.add_conditional_edges(
        "loan_agent",
        manager_approval_check,
        {
            "manager_approval": "manager_approval", 
            "complete": END
        }
    )
    
    workflow.add_edge("manager_approval", END)
    
    return workflow.compile()

# Example usage and testing
def run_banking_agent_example():
    """Run example scenarios with the banking agent."""
    
    app = create_banking_workflow()
    
    # Test scenarios
    scenarios = [
        {
            "customer_request": "I need to check my account balance and recent transactions",
            "account_number": "ACC001"
        },
        {
            "customer_request": "I want to transfer $8000 to another account",
            "account_number": "ACC002"
        },
        {
            "customer_request": "I'm interested in applying for a home loan of $200,000",
            "account_number": "ACC003"
        },
        {
            "customer_request": "Please freeze my account immediately, I lost my card",
            "account_number": "ACC004"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i}: {scenario['customer_request']}")
        print(f"Account: {scenario['account_number']}")
        print('='*60)
        
        # Initialize state
        initial_state = {
            "messages": [],
            "current_agent": "",
            "customer_request": scenario["customer_request"],
            "account_number": scenario["account_number"],
            "routing_decision": "",
            "agent_response": "",
            "retry_count": 0,
            "requires_manager_approval": False
        }
        
        # Run the workflow
        try:
            result = app.invoke(initial_state)
            
            print(f"\nFinal Agent: {result['current_agent']}")
            print(f"Routing Decision: {result['routing_decision']}")
            if result.get('requires_manager_approval'):
                print("⚠️  Manager approval required")
            
            print(f"\nFinal Response:")
            print(result['agent_response'])
            
        except Exception as e:
            print(f"Error processing scenario: {e}")

if __name__ == "__main__":
    # Make sure to set your GROQ_API_KEY environment variable
    if not os.getenv("GROQ_API_KEY"):
        print("Please set your GROQ_API_KEY environment variable")
    else:
        run_banking_agent_example()