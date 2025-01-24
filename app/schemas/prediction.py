from pydantic import BaseModel

class DiseasePredictionResponse(BaseModel):
    id: int
    name: str
    disease: str
    confidence: float
