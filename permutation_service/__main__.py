"""`python -m permutation_service` entry point."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "permutation_service.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )

