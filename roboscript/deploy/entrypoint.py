"""Container entrypoint.

    python -m roboscript.deploy.entrypoint            # uvicorn on $PORT (default 8000)
    python -m roboscript.deploy.entrypoint serve      # SageMaker: port 8080, /ping + /invocations

SageMaker real-time endpoints call ``docker run <image> serve`` and expect
GET /ping and POST /invocations on port 8080; both are thin aliases over the
regular API (/v1/health and /v1/act).
"""
from __future__ import annotations

import os
import sys

import uvicorn
from fastapi import Response

from roboscript.serve.api import TelemetryRequest, app


@app.get("/ping")
async def ping() -> Response:
    return Response(status_code=200)


@app.post("/invocations")
async def invocations(req: TelemetryRequest):
    for route in app.routes:
        if getattr(route, "path", None) == "/v1/act":
            return await route.endpoint(req)
    raise RuntimeError("/v1/act not registered")


def main() -> None:
    sagemaker = len(sys.argv) > 1 and sys.argv[1] == "serve"
    port = 8080 if sagemaker else int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
