import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from apps.backend.core.config import settings
from apps.backend.database.session import engine, Base
from apps.backend.api.v1.endpoints import patients, predict, explain, rag, models_info

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="MedIntel AI — Multimodal Medical Intelligence & Clinical Decision-Support Research Platform"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=FileResponse)
def serve_dashboard():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/dashboard", response_class=FileResponse)
def serve_dashboard_alias():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next):
    response = await call_next(request)
    return response

@app.get(f"{settings.API_V1_STR}/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "disclaimer": "Research and educational decision-support platform only."
    }

# Include API v1 Routers
app.include_router(patients.router, prefix=f"{settings.API_V1_STR}/patients", tags=["Patients"])
app.include_router(predict.router, prefix=f"{settings.API_V1_STR}/predict", tags=["Predict"])
app.include_router(explain.router, prefix=f"{settings.API_V1_STR}/explain", tags=["Explainability"])
app.include_router(rag.router, prefix=f"{settings.API_V1_STR}/rag", tags=["Knowledge Assistant RAG"])
app.include_router(models_info.router, prefix=f"{settings.API_V1_STR}/models", tags=["Models Registry"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
