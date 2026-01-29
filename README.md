# DiffCast

Diffusion-based day-ahead electricity price forecasting for Nord Pool (Nordic core regions).

## Overview

DiffCast implements a CSDI-style (Conditional Score-based Diffusion model for Imputation) architecture for probabilistic day-ahead electricity price forecasting. The model generates probabilistic forecasts for 10 Nordic bidding zones (NO1-5, SE1-4, FI) with 24-hour ahead predictions at hourly resolution.

## Features

- **Probabilistic Forecasting**: Generates full predictive distributions, not just point forecasts
- **Multi-zone Support**: Models correlations across Nordic bidding zones
- **CSDI Architecture**: Dual transformer backbone with temporal and cross-zone attention
- **Fast Inference**: DDIM sampling for efficient generation
- **Comprehensive Evaluation**: CRPS, pinball loss, calibration metrics, and more

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/diffcast.git
cd diffcast

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Download Data

You need an ENTSO-E Transparency Platform API key. Get one at: https://transparency.entsoe.eu/

```bash
export ENTSOE_API_KEY=your_api_key
python scripts/download_data.py
```

### 2. Prepare Dataset

```bash
python scripts/prepare_dataset.py
```

### 3. Train Model

```bash
python scripts/train.py
```

### 4. Evaluate

```bash
python scripts/evaluate.py checkpoints/best.ckpt
```

## Project Structure

```
diffcast/
├── src/diffcast/
│   ├── data/           # Data loading and processing
│   │   ├── sources/    # ENTSO-E, weather, calendar data sources
│   │   ├── pipeline.py # Data processing pipeline
│   │   └── dataset.py  # PyTorch Dataset classes
│   ├── models/
│   │   ├── csdi/       # CSDI diffusion model
│   │   └── baselines/  # Baseline models
│   ├── training/       # Training infrastructure
│   ├── evaluation/     # Metrics and visualization
│   └── inference/      # Forecasting pipeline
├── scripts/            # CLI scripts
├── configs/            # Configuration files
└── tests/              # Unit tests
```

## Model Architecture

The CSDI model consists of:

1. **History Encoder**: Transformer encoder for historical context
2. **Diffusion Process**: Forward/reverse diffusion with quadratic noise schedule
3. **Dual Transformer Denoiser**: Alternating temporal and feature attention layers
4. **Conditioning Module**: Cross-attention on history, FiLM for timestep embedding

## Data Sources

| Data | Source |
|------|--------|
| Day-ahead prices | ENTSO-E Transparency Platform |
| Load (actual/forecast) | ENTSO-E Transparency Platform |
| Wind/Solar/Hydro generation | ENTSO-E Transparency Platform |
| Temperature | Open-Meteo Historical API |
| Holidays | `holidays` Python library |

## Configuration

Edit `configs/default.yaml` to customize:

- Model architecture (d_model, n_heads, n_layers)
- Diffusion process (n_steps, noise schedule)
- Training (batch_size, learning_rate, epochs)
- Data processing (zones, date range)

## Evaluation Metrics

- **CRPS**: Continuous Ranked Probability Score (primary metric)
- **Pinball Loss**: Quantile loss at 10%, 50%, 90%
- **Calibration**: Observed vs expected coverage
- **MAE/RMSE**: Point forecast accuracy (median)
- **Winkler Score**: Interval sharpness

## License

MIT

## Citation

If you use this code, please cite:

```bibtex
@software{diffcast2024,
  title={DiffCast: Diffusion-based Day-Ahead Price Forecasting},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/diffcast}
}
```

## References

- Tashiro et al. (2021). "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation"
- Ho et al. (2020). "Denoising Diffusion Probabilistic Models"
- Song et al. (2020). "Denoising Diffusion Implicit Models"
