"""FastAPI entrypoint: core API (/api/v1), intelligence API (/api/v1), frontend API (/api) and the static UI."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.api import routes_core, routes_frontend, routes_intel, routes_ops
from backend.app.config import ROOT, get_settings
from backend.app.deps import build_container, get_container, set_container

logger = logging.getLogger("upayshield")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    set_container(build_container(get_settings()))
    yield
    set_container(None)


def _error(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details}})


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="UpayShield Trust & Risk Intelligence API",
        description="Case-centric fraud detection, graph analytics and a grounded AI assistant for mobile money.",
        version="1.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])

    # every error uses the contract shape {"error": {"code", "message", "details"}}
    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            return JSONResponse(status_code=exc.status_code, content=detail)
        code = {404: "not_found", 405: "method_not_allowed", 503: "service_unavailable"}.get(exc.status_code, f"http_{exc.status_code}")
        return _error(exc.status_code, code, str(detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        return _error(422, "validation_error", "Request validation failed",
                      {"errors": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]})

    app.include_router(routes_core.router)       # /api/v1/score, /cases, /config, /metrics, /warning/check
    app.include_router(routes_ops.router)        # /api/v1/stream (SSE), /stream/control, /admin/thresholds/*
    app.include_router(routes_intel.router)      # /api/v1/graph, /agents, /cases/{id}/evidence|narrative|ask|report
    app.include_router(routes_frontend.router)   # /api/overview, /transactions, /stream, /cases, /graph, /score ...

    @app.get("/health", tags=["meta"])
    def health():
        c = get_container()
        sc = c.scoring
        return {"status": "ok" if sc else "degraded", "model_loaded": sc is not None,
                "model": sc.engine.name if sc else None, "model_version": sc.engine.version if sc else None,
                "cases": len(getattr(c.cases, "_cases", {})), "intelligence": "live" if c.intel.live else "stubs"}

    frontend = ROOT / "Frontend"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
__all__ = ["app", "HTTPException"]
