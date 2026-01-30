# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DiffCast is a CSDI-style diffusion model for probabilistic day-ahead electricity price forecasting on Nord Pool (10 Nordic bidding zones: NO1-5, SE1-4, FI). It generates 24-hour ahead probabilistic forecasts at hourly resolution.

## Common Commands

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/
pytest tests/test_model.py -v              # Single test file
pytest tests/test_model.py::TestCSDI -v    # Single test class

# Lint and type check
ruff check src/
ruff format src/
mypy src/

# Data pipeline
python scripts/download_data.py            # Requires ENTSOE_API_KEY env var
python scripts/prepare_dataset.py

# Training
python scripts/train.py
python scripts/train.py --fast-dev-run     # Quick validation
python scripts/train.py --max-epochs 10

# Evaluation
python scripts/evaluate.py checkpoints/best.ckpt
```

## Architecture

### CSDI Model (`src/diffcast/models/csdi/`)
- **diffusion.py**: Forward/reverse diffusion process with DDPM and DDIM sampling
- **transformer.py**: Dual transformer with alternating temporal and feature attention
- **conditioning.py**: History encoder, FiLM conditioning, timestep embeddings
- **model.py**: Full CSDI model combining all components

The model takes 7 days of historical context (168 hours) and generates 24-hour probabilistic forecasts via iterative denoising.

### Data Flow
1. **Sources** (`data/sources/`): ENTSO-E (prices, load, generation), Open-Meteo (weather), holidays
2. **Pipeline** (`data/pipeline.py`): Merges sources, adds calendar features, creates train/val/test splits
3. **Dataset** (`data/dataset.py`): `DiffCastDataset` for single-zone, `MultiZoneDataset` for cross-zone modeling

### Training (`src/diffcast/training/`)
- PyTorch Lightning module with noise prediction loss
- Primary metric: CRPS (Continuous Ranked Probability Score)
- Config in `configs/default.yaml`

## Key Configuration

Edit `configs/default.yaml`:
- `model.multi_zone`: false for per-zone models, true for joint cross-zone modeling
- `training.context_length`: 168 (7 days of history)
- `training.forecast_length`: 24 (1 day ahead)
- `diffusion.n_steps`: 50 training steps, use `inference.ddim_steps`: 20 for faster sampling

## Issue Tracking (beads)

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Session Completion

**MANDATORY** before ending a session:
1. Run quality gates if code changed (pytest, ruff)
2. Update issue status
3. Push to remote:
   ```bash
   git pull --rebase && bd sync && git push
   ```
4. Verify `git status` shows "up to date with origin"
