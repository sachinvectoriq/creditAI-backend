from fastapi import FastAPI

# This is the ASGI app Uvicorn will run
main = FastAPI()

@main.get("/")
async def root():
    return {"message": "Hello from microservices-architecture!"}
