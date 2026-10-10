# University IT policy RAG exercise

This program loads the single supplied policy file through `policy_loader.py`, retrieves relevant sections with BM25, and asks Qwen3-0.6B to answer only from those sections. An exact Service Desk phone number is returned only if a policy explicitly states one.

## Setup on Windows

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe 'torch==2.13.0' --index https://download.pytorch.org/whl/cu130
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Set `QWEN_MODEL_PATH` to a local Qwen3-0.6B directory if the sibling `SCSE_26_GEX3/models/qwen3-0.6b` directory is unavailable. Otherwise the program uses that sibling directory, or downloads `Qwen/Qwen3-0.6B` from Hugging Face.

```powershell
.\.venv\Scripts\python.exe check_data.py
.\.venv\Scripts\python.exe agent.py
```

The expected answer to the default question is that the exact University IT Service Desk phone number is unknown because the supplied policies do not state it. `policy_loader.py` and the supplied data files are unchanged.
