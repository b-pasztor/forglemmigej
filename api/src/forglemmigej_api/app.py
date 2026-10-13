from fastapi import FastAPI

app = FastAPI(title="forglemmigej API")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
