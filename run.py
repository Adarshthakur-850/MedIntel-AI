import sys
import os
import webbrowser
import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print("  Starting MedIntel AI — Multimodal Medical Intelligence Platform")
    print("  Web Dashboard: http://localhost:8000/")
    print("  API Documentation: http://localhost:8000/docs")
    print("=" * 65)
    
    # Open browser automatically after 1.5s
    webbrowser.open("http://localhost:8000/")
    
    uvicorn.run("apps.backend.main:app", host="127.0.0.1", port=8000, reload=True)
