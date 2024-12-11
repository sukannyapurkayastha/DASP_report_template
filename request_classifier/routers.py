from fastapi import APIRouter, HTTPException


from pydantic import BaseModel

router = APIRouter()

class RawInput(BaseModel):
    data: list[dict]  # Raw input data in JSON format


@router.post("/classify_request")
async def classify_request(request: RawInput):
    data = request.data
