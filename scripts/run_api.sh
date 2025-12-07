#!/usr/bin/env bash
export PYTHONPATH=.
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
