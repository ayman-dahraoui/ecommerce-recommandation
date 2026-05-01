from fastapi import FastAPI
from app.database import engine, Base
import app.models

app = FastAPI(
    title="Système de Recommandation E-commerce",
    description="API pour les recommandations de produits",
    version="1.0.0"
)

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de recommandation !"}

@app.get("/health")
def health_check():
    return {"status": "ok"}