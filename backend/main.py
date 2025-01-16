from fastapi import FastAPI
import uvicorn
from routers import router

app = FastAPI()

# Include the router for this endpoint
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
