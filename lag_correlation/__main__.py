"""Run with ``python -m lag_correlation``."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("lag_correlation.app:app", host="127.0.0.1", port=8000, reload=False)
