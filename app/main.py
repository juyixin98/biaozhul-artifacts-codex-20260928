"""Runnable service entry point.

    uvicorn app.main:app --host 127.0.0.1 --port 8000

or simply:

    python -m app.main
"""
from .service import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
