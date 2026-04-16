import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from extract import extract_clauses
from summarize import summarize_all

app = FastAPI(
    title="Legal Contract Clause Extractor",
    description="Extracts and summarizes legal clauses from PDF contracts using BERT and Gemini",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "running", "model": "bert-base-uncased fine-tuned on CUAD"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        clauses = extract_clauses(tmp_path)

        if not clauses:
            return {"filename": file.filename, "clauses": [], "message": "No clauses detected"}

        summarized = summarize_all(clauses)
        os.unlink(tmp_path)

        return {
            "filename": file.filename,
            "total_clauses_found": len(summarized),
            "clauses": summarized
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))