"""Correct local reference service."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .common import (
    CreateOrderRequest,
    OrderResponse,
    OrderStore,
    configure_exception_handlers,
)


def create_app() -> FastAPI:
    app = FastAPI(title="Reference Order Service", version="0.1.0")
    store = OrderStore()
    configure_exception_handlers(app)

    @app.middleware("http")
    async def synthetic_faults(
        request: Request,
        call_next,
    ):
        x_simulate_fault = request.headers.get("X-Simulate-Fault")
        if request.url.path == "/orders" and request.method == "POST":
            if x_simulate_fault == "resource_exhausted":
                return JSONResponse(
                    status_code=507,
                    content={
                        "error": "synthetic order capacity exhausted",
                        "category": "resource_exhausted",
                    },
                )
            if x_simulate_fault == "computation_failure":
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "synthetic local computation failure",
                        "category": "computation_failure",
                    },
                )
        return await call_next(request)

    @app.post("/reset", status_code=204)
    def reset() -> None:
        store.reset()

    @app.post("/orders", response_model=OrderResponse, status_code=201)
    def create_orders(payload: CreateOrderRequest) -> OrderResponse:
        order_id = store.create(payload.id)
        return OrderResponse(id=order_id, state="draft")

    @app.post("/orders/{order_id}/submit", response_model=OrderResponse)
    def submit_order(order_id: str) -> OrderResponse:
        state = store.state(order_id)
        if state != "draft":
            return JSONResponse(
                status_code=409,
                content={
                    "error": "only a draft order may be submitted",
                    "category": "state_conflict",
                },
            )
        store.replace(order_id, "submitted")
        return OrderResponse(id=order_id, state="submitted")

    @app.post("/orders/{order_id}/close", response_model=OrderResponse)
    def close_order(order_id: str) -> OrderResponse:
        state = store.state(order_id)
        if state != "submitted":
            return JSONResponse(
                status_code=409,
                content={
                    "error": "only a submitted order may be closed",
                    "category": "state_conflict",
                },
            )
        store.replace(order_id, "closed")
        return OrderResponse(id=order_id, state="closed")

    @app.post("/orders/{order_id}/cancel", response_model=OrderResponse)
    def cancel_order(order_id: str) -> OrderResponse:
        state = store.state(order_id)
        if state not in {"draft", "submitted"}:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "only draft or submitted orders may be cancelled",
                    "category": "state_conflict",
                },
            )
        store.replace(order_id, "cancelled")
        return OrderResponse(id=order_id, state="cancelled")

    @app.get("/orders/{order_id}", response_model=OrderResponse)
    def get_order(order_id: str) -> OrderResponse:
        return OrderResponse(id=order_id, state=store.state(order_id))

    return app


app = create_app()
