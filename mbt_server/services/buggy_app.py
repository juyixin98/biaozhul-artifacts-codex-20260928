"""Deliberately stateful local service with independent transition code."""

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
    app = FastAPI(title="Buggy Order Service", version="0.1.0")
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
        if state == "draft":
            store.replace(order_id, "submitted")
            return OrderResponse(id=order_id, state="submitted")
        if state == "closed":
            # Bug: an already-closed order is accepted as submitted.
            store.replace(order_id, "submitted")
            return OrderResponse(id=order_id, state="submitted")
        return JSONResponse(
            status_code=409,
            content={
                "error": "submission is not allowed in this state",
                "category": "state_conflict",
            },
        )

    @app.post("/orders/{order_id}/close", response_model=OrderResponse)
    def close_order(order_id: str) -> OrderResponse:
        state = store.state(order_id)
        if state == "submitted":
            store.replace(order_id, "closed")
            return OrderResponse(id=order_id, state="closed")
        if state == "draft":
            # Bug: an out-of-order close is reported as successful without
            # persisting a state change, which directly distinguishes model
            # behavior from backing state.
            return OrderResponse(id=order_id, state="draft")
        if state == "closed":
            # Bug: repeat close is idempotent-looking but should conflict.
            return OrderResponse(id=order_id, state="closed")
        return JSONResponse(
            status_code=409,
            content={
                "error": "close is not allowed in this state",
                "category": "state_conflict",
            },
        )

    @app.post("/orders/{order_id}/cancel", response_model=OrderResponse)
    def cancel_order(order_id: str) -> OrderResponse:
        state = store.state(order_id)
        if state in {"draft", "submitted"}:
            store.replace(order_id, "cancelled")
            return OrderResponse(id=order_id, state="cancelled")
        if state == "closed":
            # Bug: cancellation of a closed order is accepted but not stored.
            return OrderResponse(id=order_id, state="closed")
        return JSONResponse(
            status_code=409,
            content={
                "error": "cancel is not allowed in this state",
                "category": "state_conflict",
            },
        )

    @app.get("/orders/{order_id}", response_model=OrderResponse)
    def get_order(order_id: str) -> OrderResponse:
        return OrderResponse(id=order_id, state=store.state(order_id))

    return app


app = create_app()
