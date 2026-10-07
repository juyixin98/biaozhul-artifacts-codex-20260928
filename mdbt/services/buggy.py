"""Order service with deliberately planted stateful bugs.

It looks almost identical to the reference service, but its transition
table contains three extra edges that an order-lifecycle model must reject.
The bugs only appear after specific histories, which is exactly why a
stateless "happy path" smoke test misses them:

1. **Repeated close** - ``close`` is accepted again from ``closed``
   (idempotent-looking 200, instead of a 409 state conflict).
2. **Out-of-order submit** - ``submit`` is accepted from ``paid`` and rolls
   the order back to ``submitted``.
3. **Cancel after delivery** - ``cancel`` is accepted from ``delivered``.

Unknown actions, missing resources and malformed bodies are still handled
correctly, so the bugs cannot be explained away as input handling errors.
"""
from __future__ import annotations

import threading

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from .common import (
    ok,
    reject,
    unhandled_exception_handler,
    validation_exception_handler,
)


# Same correct edges as the reference service ...
CORRECT_TRANSITIONS: dict[tuple[str, str], str] = {
    ("draft", "submit"): "submitted",
    ("submitted", "approve"): "paid",
    ("submitted", "cancel"): "cancelled",
    ("paid", "deliver"): "delivered",
    ("paid", "cancel"): "cancelled",
    ("delivered", "close"): "closed",
}

# ... plus the three planted stateful defects.
PLANTED_TRANSITIONS: dict[tuple[str, str], str] = {
    ("closed", "close"): "closed",        # bug 1: repeated close
    ("paid", "submit"): "submitted",      # bug 2: out-of-order submit (rollback)
    ("delivered", "cancel"): "cancelled",  # bug 3: cancel after delivery
}

TRANSITIONS = {**CORRECT_TRANSITIONS, **PLANTED_TRANSITIONS}
KNOWN_ACTIONS = {"create", *{a for (_s, a) in TRANSITIONS}}
TERMINAL_STATES = {"closed", "cancelled"}


class OrderRequest(BaseModel):
    resource_id: str = Field(min_length=1, max_length=128)
    action: str = Field(min_length=1, max_length=64)


def create_app() -> FastAPI:
    app = FastAPI(title="buggy-order-service", version="1.0.0")
    app.state.orders = {}
    app.state.lock = threading.Lock()
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "service": "buggy"}

    @app.get("/orders/{resource_id}")
    def get_order(resource_id: str):
        with app.state.lock:
            state = app.state.orders.get(resource_id)
        if state is None:
            return reject(404, "order not found", exists=False)
        return {"ok": True, "state": state, "exists": True}

    @app.post("/orders")
    def operate(req: OrderRequest):
        rid, action = req.resource_id, req.action

        if action not in KNOWN_ACTIONS:
            with app.state.lock:
                current = app.state.orders.get(rid)
            return reject(
                400, f"unknown action {action!r}", state=current,
                exists=current is not None,
            )

        with app.state.lock:
            current = app.state.orders.get(rid)

            if action == "create":
                if current is not None:
                    return reject(
                        409,
                        f"order already exists in state {current!r}",
                        state=current,
                    )
                app.state.orders[rid] = "draft"
                return ok("draft", f"order {rid} created", code=201)

            if current is None:
                return reject(404, "order not found", exists=False)

            target = TRANSITIONS.get((current, action))
            if target is None:
                if current in TERMINAL_STATES:
                    return reject(
                        409,
                        f"order is {current}; no further actions are allowed",
                        state=current,
                    )
                return reject(
                    409,
                    f"action {action!r} is not allowed in state {current!r}",
                    state=current,
                )

            app.state.orders[rid] = target
            return ok(target, f"{action} applied: {current} -> {target}")

    return app


app = create_app()
