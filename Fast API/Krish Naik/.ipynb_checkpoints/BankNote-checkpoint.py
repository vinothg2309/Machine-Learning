# -*- coding: utf-8 -*-

from pydantic import BaseModel

# pydantic enforces type hints at runtime, and provides user friendly errors when data is invalid.
class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float
    