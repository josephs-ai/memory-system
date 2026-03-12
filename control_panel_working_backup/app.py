from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import CONTROL_PANEL_DIR
from views import router

app = FastAPI(title="OpenClaw Control Panel")
app.mount("/static", StaticFiles(directory=str(CONTROL_PANEL_DIR / "static")), name="static")
app.include_router(router)

