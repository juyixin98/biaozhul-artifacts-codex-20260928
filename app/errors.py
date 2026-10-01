class QPError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.request_id = request_id


class InvalidProblemError(QPError):
    pass


class NumericalError(QPError):
    def __init__(self, code: str, message: str):
        super().__init__(code, message, status_code=500)
