# Agentic Financial Assistant

A multi-agent AI system that handles banking operations using specialized agents that work together.

## What it does

This project creates 5 different AI agents that handle specific banking tasks:

- **Loan Agent** - Makes loan decisions using customer data and external APIs
- **General Agent** - Handles database queries and general banking questions  
- **Mail Agent** - Sends emails for campaigns, alerts, and notifications
- **Review Agent** - Analyzes customer feedback and reviews
- **Fraud Agent** - Monitors transactions for suspicious activity 

## How it works

When you ask a question, the system figures out which agent should handle it. For example:
- "Should we approve this loan?" → Goes to Loan Agent
- "Send balance alerts" → Goes to Mail Agent  
- "Check for fraud" → Goes to Fraud Agent

Each agent has its own tools and can call external services, query databases, or send emails.

## Tech Stack

- **LangGraph** - For connecting agents together
- **Neo4j** - Graph database for customer/transaction data
- **Groq/LLaMA** - Language model for understanding and responses
- **Python** - Main programming language
- **SMTP** - For sending emails
- **REST APIs** - For loan decisions

