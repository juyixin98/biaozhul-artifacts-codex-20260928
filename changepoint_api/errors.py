class ChangepointError(Exception):
    """Base error whose category is persisted instead of returning success."""

    status_code = 500
    error_code = "kernel_error"

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.run_id = run_id
        if status_code is not None:
            self.status_code = status_code
        if error_code is not None:
            self.error_code = error_code


class NumericalGuardError(ChangepointError):
    status_code = 500
    error_code = "numerical_guard_failed"


class UnsupportedConfigurationError(ChangepointError):
    status_code = 422
    error_code = "unsupported_configuration"


class StorageError(ChangepointError):
    status_code = 500
    error_code = "storage_error"
