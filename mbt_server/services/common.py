"""Small local synthetic order-service helpers used by both SUT variants."""

from __future__ import annotations

from threading import Lock
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

OrderState = Literal["draft", "submitted", "closed", "cancelled"]


class CreateOrderRequest(BaseModel):
    id: str | None = Field(default=None, min_length=1, max_length=80)


class OrderResponse(BaseModel):
    id: str
    state: OrderState


class ErrorResponse(BaseModel):
    error: str
    category: str


class OrderStore:
    def __init__(self) -> None:
        self.orders: dict[str, OrderState] = {}
        self.counter = 0
        self.lock = Lock()

    def reset(self) -> None:
        with self.lock:
            self.orders.clear()
            self.counter = 0

    def create(self, order_id: str | None) -> str:
        with self.lock:
            if order_id is not None:
                if order_id in self.orders:
                    raise KeyError(order_id)
                self.orders[order_id] = "draft"
                return order_id
            self.counter += 1
            order_id = f"order-{self.counter}"
            self.orders[order_id] = "draft"
            return order_id

    def state(self, order_id: str) -> OrderState:
        with self.lock:
            if order_id not in self.orders:
                raise LookupError(order_id)
            return self.orders[order_id]

    def replace(self, order_id: str, state: OrderState) -> None:
        with self.lock:
            if order_id not in self.orders:
                raise LookupError(order_id)
            self.orders[order_id] = state


def error_body(error: str, category: str) -> JSONResponse:
    return JSONResponse(
        status_code=409 if category == "state_conflict" else 400,
        content={"error": error, "category": category},
    )


def configure_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LookupError)
    async def not_found_handler(_: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"unknown resource: {exc}",
                "category": "resource_not_found",
            },
        )

    @app.exception_handler(KeyError)
    async def duplicate_handler(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "error": f"resource already exists: {exc.args[0]}",
                "category": "state_conflict",
            },
        )
