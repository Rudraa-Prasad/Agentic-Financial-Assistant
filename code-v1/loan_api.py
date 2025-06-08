# Simple Loan Decision API - Run on localhost:8000
from fastapi import FastAPI  , HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
# import uvicorn

app = FastAPI(title="Bank Loan Decision API", version="1.0.0")

class CustomerData(BaseModel):
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    balance: float
    additional_info: Optional[Dict] = None

class LoanResponse(BaseModel):
    decision: str  # "approved" or "rejected"
    amount_approved: Optional[float] = None
    interest_rate: Optional[float] = None
    tenure_months: Optional[int] = None
    reason: str
    risk_score: int  # 1-10 scale
    conditions: Optional[list] = None

def calculate_loan_decision(customer_data: CustomerData) -> LoanResponse:
    """
    Simple loan decision logic based on account balance and other factors
    In a real system, this would involve complex ML models and credit scoring
    """
    
    balance = customer_data.balance
    name = customer_data.name
    
    # Simple decision logic
    if balance >= 50000:
        # High balance customers - approved
        return LoanResponse(
            decision="approved",
            amount_approved=min(balance * 2, 500000),  # Max 5 lakh
            interest_rate=8.5,
            tenure_months=60,
            reason="Excellent account balance and banking relationship",
            risk_score=2,
            conditions=[
                "Maintain minimum balance of ₹25,000",
                "Set up auto-debit for EMI payments",
                "Provide salary certificate if employed"
            ]
        )
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("loan_api:app", host="127.0.0.1", port=8000, reload=True)
