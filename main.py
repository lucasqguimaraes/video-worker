import os
import subprocess
from fastapi import FastAPI
import redis
import psycopg2

app = FastAPI(title="Video Machine Worker", version="1.0.0")

REDIS_HOST = os.getenv("REDIS_HOST", "n8n_redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

DB_HOST = os.getenv("DB_HOST", "n8n_postgres")
DB_NAME = os.getenv("DB_NAME", "video_machine")
DB_USER = os.getenv("DB_USER", "n8n")
DB_PASS = os.getenv("DB_PASS", "")

SHARED_DIR = "/shared_media"

@app.get("/health")
def health_check():
    volume_ok = os.path.exists(SHARED_DIR) and os.access(SHARED_DIR, os.W_OK)

    try:
        res = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        ffmpeg_ok = "ffmpeg version" in res.stdout
    except Exception:
        ffmpeg_ok = False

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=2, decode_responses=True)
        redis_ok = r.ping()
    except Exception as e:
        redis_ok = f"Erro: {str(e)}"

    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        conn.close()
        postgres_ok = True
    except Exception as e:
        postgres_ok = f"Erro: {str(e)}"

    return {
        "status": "online",
        "shared_volume_writable": volume_ok,
        "ffmpeg_installed": ffmpeg_ok,
        "redis_db2_connected": redis_ok,
        "postgres_video_machine_connected": postgres_ok
    }
