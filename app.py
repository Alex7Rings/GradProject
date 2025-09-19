from fastapi import FastAPI
from db.session import Base, engine
from routers.routers import router

# Create all tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Portfolio API")

app.include_router(router)
