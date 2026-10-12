"""Render entry point for the existing travel UI and backend."""
import sys
from pathlib import Path
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
from main import app

@app.get("/", include_in_schema=False)
def home(embedded: str | None = None):
    if embedded is None:
        return RedirectResponse("/?embedded=1")
    from fastapi.responses import FileResponse
    return FileResponse(ROOT / "static" / "index.html")

app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="frontend")
