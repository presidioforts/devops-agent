from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import pathlib, os, uuid, json, logging
from datetime import datetime
from typing import Optional, Dict, List
import threading

# ======================================================================
# Pydantic models (define first)
# ======================================================================
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

# ======================================================================
# Paths & constants
# ======================================================================
LOCAL_MODELS_DIR = pathlib.Path(r"breakfix-kb-model")
BASE_MODEL_DIR = LOCAL_MODELS_DIR / "all-mpnet-base-v2"
RUNS_DIR = BASE_MODEL_DIR / "fine-tuned-runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_DIR = BASE_MODEL_DIR / "fine-tuned"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ======================================================================
# Logging
# ======================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ======================================================================
# Helper: pick latest fine-tuned dir (if any)
# ======================================================================
def latest_run_dir() -> Optional[pathlib.Path]:
    candidates = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()],
                        key=lambda p: p.name,
                        reverse=True)
    return candidates[0] if candidates else (LEGACY_DIR if LEGACY_DIR.exists() else None)

def latest_pairs_file() -> Optional[pathlib.Path]:
    files = sorted(RUNS_DIR.glob("*/pairs.json"), key=lambda p: p.parent.name, reverse=True)
    return files[0] if files else None

def load_pairs_from_disk() -> List[TrainingPair]:
    path = latest_pairs_file()
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [TrainingPair(**item) for item in raw]

# ======================================================================
# Build learned_pairs after model definition
# ======================================================================
defined_pairs: List[TrainingPair] = load_pairs_from_disk()
learned_pairs = defined_pairs

# ======================================================================
# Thread safety for model access
# ======================================================================
model_lock = threading.Lock()

# ======================================================================
# Model load at startup
# ======================================================================
try:
    load_path = latest_run_dir() or BASE_MODEL_DIR
    with model_lock:
        model = SentenceTransformer(str(load_path))
        _ = model.encode("health-check")
    logger.info(f"Model loaded from: {load_path}")
except Exception as e:
    logger.exception("Failed to load model")
    raise

# ======================================================================
# simple in-memory KB (replace with DB later)
# ======================================================================
knowledge_base: List[KnowledgeBaseItem] = [
    KnowledgeBaseItem(description="npm ERR! code ERESOLVE", resolution="Delete node_modules & package-lock.json, then run npm install."),
    KnowledgeBaseItem(description="Script not running after package install", resolution="Check package.json scripts and dependencies."),
    KnowledgeBaseItem(description="npm install hangs", resolution="Clear npm cache (npm cache clean --force) or check network."),
    KnowledgeBaseItem(description="Update npm version", resolution="Run npm install -g npm@latest."),
]

# ======================================================================
# Background trainer + job tracker
# ======================================================================
jobs: Dict[str, Dict] = {}          # job_id -> {"status":..., "msg":...}

def _new_output_dir() -> pathlib.Path:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out = RUNS_DIR / f"fine-tuned-{ts}"
    out.mkdir(parents=True, exist_ok=False)   # fail fast on duplicate timestamp
    return out

def fine_tune(job_id: str, pairs: List[TrainingPair]):
    global model
    try:
        jobs[job_id]["status"] = "running"
        with model_lock:
            examples = [InputExample(texts=[p.input, p.target], label=1.0) for p in pairs]
            loader = DataLoader(
                examples,
                shuffle=True,
                batch_size=8,
                collate_fn=model.smart_batching_collate
            )
            loss_fn = losses.CosineSimilarityLoss(model)
            model.fit([(loader, loss_fn)], epochs=1,
                      optimizer_params={"lr": 1e-5},
                      show_progress_bar=False)
            out_dir = _new_output_dir()
            model.save(str(out_dir))
            model = SentenceTransformer(str(out_dir))  # hot‑reload
            with open(out_dir / "pairs.json", "w", encoding="utf-8") as f:
                json.dump([p.dict() for p in pairs], f, ensure_ascii=False, indent=2)
            learned_pairs.extend(pairs)           # inside lock so /troubleshoot sees consistent list
        jobs[job_id] = {"status": "finished", "msg": f"saved to {out_dir}"}
    except Exception as e:
        logger.exception("Training failed")
        jobs[job_id] = {"status": "failed", "msg": str(e)}

# ======================================================================
# FastAPI app
# ======================================================================
app = FastAPI()

# ======================================================================
# API endpoints
# ======================================================================
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
        corpus_inputs  = [p.input  for p in learned_pairs] + [kb.description for kb in knowledge_base]
        corpus_answers = [p.target for p in learned_pairs] + [kb.resolution  for kb in knowledge_base]
        if not corpus_inputs:
            raise ValueError("Search corpus is empty")

        with model_lock:                       # lock only around encode calls
            query_emb = model.encode(q.text, convert_to_tensor=True)
            kb_embs   = model.encode(corpus_inputs, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, kb_embs)[0]
        best   = int(scores.argmax())
        return {"query": q.text,
                "response": corpus_answers[best],
                "similarity_score": float(scores[best])}
    except Exception as e:
        logger.exception("Troubleshoot failed")
        raise HTTPException(500, "internal error")

# Local dev convenience
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
