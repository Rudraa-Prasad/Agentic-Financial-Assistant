🧠 Agents in the Workflow
agent_summ – Review summarization and improvement suggestions

general_agent – General Neo4j query response generator

loan_agent – Loan approval decision and reasoning agent

mail_agent – Email drafting and sending agent for marketing or alerts

🔧 Tools in the Workflow
cypher_query_generator – Converts natural language queries to Cypher queries

neo4j_retriever – Executes Cypher queries and retrieves data from the Neo4j database

review_reflector – Cleans, validates, and possibly retries review summarization

loan_approval_api_tool – Calls external loan approval API

mail_intent_classifier – Classifies email type (e.g., min balance alert vs. product advertisement)

send_mail – Sends the crafted email to customers retrieved from the database

