"""HTTP boundary to a local system-under-test."""

from __future__ import annotations

from typing import Any

import httpx

from .domain import ErrorCategory, Observation, PlannedOperation


class HttpSutAdapter:
    """Render a planned operation and invoke one local HTTP service."""

    def __init__(self, base_url: str, timeout: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpSutAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def render(
        plan: PlannedOperation,
        placeholder: str,
        resource_id: str | None,
    ) -> tuple[str, dict[str, str]]:
        values = {} if resource_id is None else {placeholder: resource_id}
        try:
            path = plan.path.format(**values)
            body = {
                key: value.format(**values)
                for key, value in plan.body.items()
            }
        except KeyError as exc:
            raise ValueError(f"missing template value: {exc}") from exc
        return path, body

    def invoke(
        self,
        plan: PlannedOperation,
        placeholder: str,
        resource_id: str | None,
        fault: str | None = None,
    ) -> Observation:
        path, body = self.render(plan, placeholder, resource_id)
        headers = {"X-Simulate-Fault": fault} if fault else {}
        try:
            response = self._client.request(
                plan.method,
                path,
                json=body if plan.method != "GET" else None,
                headers=headers,
            )
            response_data: Any
            try:
                response_data = response.json()
            except ValueError:
                response_data = response.text
            category = self.category_for_status(response.status_code, response_data)
            return Observation(
                status_code=response.status_code,
                category=category,
                response_body=response_data,
            )
        except (httpx.RequestError, ValueError) as exc:
            return Observation(
                status_code=0,
                category=ErrorCategory.COMPUTATION_FAILURE,
                transport_error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def category_for_status(status_code: int, body: Any) -> ErrorCategory | None:
        if 200 <= status_code < 300:
            return None
        explicit = None
        if isinstance(body, dict):
            value = body.get("category")
            try:
                explicit = ErrorCategory(value) if value else None
            except ValueError:
                explicit = None
        if explicit is not None:
            return explicit
        if status_code == 404:
            return ErrorCategory.RESOURCE_NOT_FOUND
        if status_code == 409:
            return ErrorCategory.STATE_CONFLICT
        if status_code in (422, 400):
            return ErrorCategory.INPUT_ERROR
        if status_code == 507:
            return ErrorCategory.RESOURCE_EXHAUSTED
        return ErrorCategory.COMPUTATION_FAILURE
