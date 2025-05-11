from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import pathlib, os, uuid, json, logging
from datetime import datetime
from typing import Optional, Dict, List

# ----------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------
MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR      = pathlib.Path(os.getenv("HUGGINGFACE_HUB_CACHE", "/models"))
RUNS_DIR       = CACHE_DIR / "fine-tuned-runs"        # each run saved here
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper: pick latest fine-tuned dir (if any)
# ----------------------------------------------------------------------
def latest_run_dir() -> Optional[pathlib.Path]:
    candidates = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()],
                        key=lambda p: p.name,
                        reverse=True)
    return candidates[0] if candidates else None

# ----------------------------------------------------------------------
# Model load at startup
# ----------------------------------------------------------------------
try:
    load_path = latest_run_dir()
    model = SentenceTransformer(str(load_path)) if load_path else SentenceTransformer(MODEL_NAME)
    logger.info(f"Model loaded from: {load_path if load_path else MODEL_NAME}")
except Exception as e:
    logger.exception("Failed to load model")
    raise

# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI()

# ----------------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------------
class Query(BaseModel):
    text: str

class TrainingPair(BaseModel):
    input: str
    target: str

class TrainingData(BaseModel):
    data: List[TrainingPair]

class KnowledgeBaseItem(BaseModel):
    description: str
    resolution: str

# simple in-memory KB (replace with DB later)
knowledge_base: List[KnowledgeBaseItem] = [
    {"description": "npm ERR! code ERESOLVE",
     "resolution": "Delete node_modules & package-lock.json, then run npm install."},
    {"description": "Script not running after package install",
     "resolution": "Check package.json scripts and dependencies."},
    {"description": "npm install hangs",
     "resolution": "Clear npm cache (npm cache clean --force) or check network."},
    {"description": "Update npm version",
     "resolution": "Run npm install -g npm@latest."},
]

# ----------------------------------------------------------------------
# Background trainer + job tracker
# ----------------------------------------------------------------------
jobs: Dict[str, Dict] = {}          # job_id -> {"status":..., "msg":...}

def _new_output_dir() -> pathlib.Path:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out = RUNS_DIR / f"fine-tuned-{ts}"
    out.mkdir(parents=True)
    return out

def fine_tune(job_id: str, pairs: List[TrainingPair]):
    global model
    try:
        jobs[job_id]["status"] = "running"

        examples = [InputExample(texts=[p.input, p.target]) for p in pairs]
        loader   = DataLoader(examples, shuffle=True, batch_size=8)
        loss_fn  = losses.CosineSimilarityLoss(model)

        model.fit(train_objectives=[(loader, loss_fn)],
                  epochs=1,
                  optimizer_params={"lr": 1e-5},
                  show_progress_bar=False)

        out_dir = _new_output_dir()
        model.save(str(out_dir))
        logger.info(f"Model saved to {out_dir}")

        # reload for inference
        model = SentenceTransformer(str(out_dir))
        jobs[job_id] = {"status": "finished", "msg": f"saved to {out_dir}"}

    except Exception as e:
        logger.exception("Training failed")
        jobs[job_id] = {"status": "failed", "msg": str(e)}

# ----------------------------------------------------------------------
# API endpoints
# ----------------------------------------------------------------------
@app.post("/train")
def train(payload: TrainingData, bg: BackgroundTasks):
    if not payload.data:
        raise HTTPException(400, "No training data received")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "msg": ""}
    bg.add_task(fine_tune, job_id, payload.data)
    return {"job_id": job_id, "note": f"{len(payload.data)} pairs accepted"}

@app.get("/train/{job_id}")
def train_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "job id not found")
    return jobs[job_id]

@app.post("/troubleshoot")
def troubleshoot(q: Query):
    try:
        query_emb = model.encode(q.text, convert_to_tensor=True)
        kb_embs   = model.encode([it["description"] for it in knowledge_base],
                                 convert_to_tensor=True)
        scores = util.cos_sim(query_emb, kb_embs)[0]
        best   = int(scores.argmax())
        return {"query": q.text,
                "response": knowledge_base[best]["resolution"],
                "similarity_score": float(scores[best])}
    except Exception as e:
        logger.exception("Troubleshoot failed")
        raise HTTPException(500, "internal error")

# Local dev convenience
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
