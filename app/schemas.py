from pydantic import BaseModel, EmailStr
from typing import List


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    phone: str
    password: str


class AdvocateOut(BaseModel):
    id: str
    name: str
    rating: float
    cases: int
    experience: int
    specialization: List[str]
    city: str
    phone: str
    email: str
    address: str