"""Reference order service (correct implementation).

Independently written: its transition rules live here as an explicit table
and are *not* generated from the testing core.  The JSON fixture and this
table are two hand-maintained descriptions of the same intended behaviour,
which lets the tests prove the tester does not rubber-stamp its own model.

Lifecycle::

            create                pay                 deliver
    (none) --------> draft -------------> paid --------------> delivered
                       |                   |                     |
                       | submit            | cancel              | close
                       v                   v                     v
                   submitted ----> cancelled              closed
                       |
                       | approve
                       v
                    paid
                       | cancel (refund)
                       v
                   cancelled

Illegal attempts get HTTP 409 (state_conflict); unknown resources 404;
unknown actions / malformed bodies 400/422 (input_error).
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


# Hand-written transition table (source of truth for this service).
TRANSITIONS: dict[tuple[str, str], str] = {
    ("draft", "submit"): "submitted",
    ("submitted", "approve"): "paid",
    ("submitted", "cancel"): "cancelled",
    ("paid", "deliver"): "delivered",
    ("paid", "cancel"): "cancelled",
    ("delivered", "close"): "closed",
}

KNOWN_ACTIONS = {"create", *{a for (_s, a) in TRANSITIONS}}
TERMINAL_STATES = {"closed", "cancelled"}


class OrderRequest(BaseModel):
    resource_id: str = Field(min_length=1, max_length=128)
    action: str = Field(min_length=1, max_length=64)


def create_app() -> FastAPI:
    app = FastAPI(title="reference-order-service", version="1.0.0")
    app.state.orders = {}
    app.state.lock = threading.Lock()
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "service": "reference"}

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

            if current in TERMINAL_STATES:
                return reject(
                    409,
                    f"order is {current}; no further actions are allowed",
                    state=current,
                )

            target = TRANSITIONS.get((current, action))
            if target is None:
                return reject(
                    409,
                    f"action {action!r} is not allowed in state {current!r}",
                    state=current,
                )

            app.state.orders[rid] = target
            return ok(target, f"{action} applied: {current} -> {target}")

    return app


app = create_app()
