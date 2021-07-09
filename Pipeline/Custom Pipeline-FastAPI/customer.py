from pydantic import BaseModel

# pydantic enforces type hints at runtime, and provides user friendly errors when data is invalid.
class Customer(BaseModel):
    customerID:str        
    gender:str
    SeniorCitizen:int
    Partner:str
    Dependents:str
    tenure:int
    PhoneService:str
    MultipleLines:str
    InternetService:str
    OnlineSecurity:str
    OnlineBackup:str
    DeviceProtection:str
    TechSupport:str
    StreamingTV:str
    StreamingMovies:str
    Contract:str
    PaperlessBilling:str
    PaymentMethod:str
    MonthlyCharges:float
    TotalCharges:str
        
    
    