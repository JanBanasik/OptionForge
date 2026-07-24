"""FastAPI application entry point for OptionForge."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from optionforge.api.routes import router

app = FastAPI(
    title="OptionForge",
    description="Interactive Monte Carlo option pricing laboratory",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
