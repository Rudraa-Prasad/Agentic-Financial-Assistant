from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn

app = FastAPI(title="Bank Loan Decision API", version="1.0.0")

# ---------- Data Models ----------
class CustomerData(BaseModel):
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    balance: float
    additional_info: Optional[Dict] = None

class LoanResponse(BaseModel):
    decision: str
    amount_approved: Optional[float] = None
    interest_rate: Optional[float] = None
    tenure_months: Optional[int] = None
    reason: str
    risk_score: int
    conditions: Optional[List[str]] = None

# ---------- Loan Logic ----------
def calculate_loan_decision(customer_data: CustomerData) -> LoanResponse:
    if customer_data.balance >= 50000:
        return LoanResponse(
            decision="approved",
            amount_approved=100000,
            interest_rate=8.5,
            tenure_months=60,
            reason="Good balance",
            risk_score=2,
            conditions=["Maintain ₹25,000 minimum balance"]
        )
    return LoanResponse(
        decision="rejected",
        reason="Low balance",
        risk_score=7
    )

# ---------- API Route ----------
@app.get("/info")
async def get_info():
    return {"info": "This is a FastAPI app for loan approval/rejected decision!"}

@app.post("/loan-decision")
async def loan_decision_endpoint(customer_data: CustomerData):
    return calculate_loan_decision(customer_data)

# ---------- Run Server ----------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


