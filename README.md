# RidgeDx — AI Blood Group Prediction from Fingerprints

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

**A biometric AI platform that predicts blood group from fingerprint images using deep learning (ResNet-18) with Grad-CAM explainability.**

[Live Demo](https://ridgedx.onrender.com) · [API Docs](https://ridgedx.onrender.com/api/docs) · [Report Bug](https://github.com/yourusername/ridgedx/issues)

</div>

---

## Features

- **ResNet-18 CNN** — Modified for grayscale fingerprint analysis, 88% test accuracy
- **8 Blood Groups** — Classifies A+, A−, B+, B−, AB+, AB−, O+, O−
- **Grad-CAM Heatmaps** — Visual explanation of which fingerprint regions drive predictions
- **REST API** — `POST /api/v1/predict` with Swagger documentation
- **Premium Dark UI** — Single-page app with glassmorphism design, zero dependencies
- **Docker Ready** — One-command deployment with Docker Compose

## Screenshots

### Landing Page
Dark-themed landing with animated fingerprint scan, feature cards, and gradient hero section.

### Prediction Flow
Drag-and-drop upload → AI analysis → blood group result with confidence ring, probability bars, and Grad-CAM heatmap.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Vanilla HTML/CSS/JS (SPA, hash routing) |
| **Backend** | Flask 3.x + flask-restx (Swagger) |
| **Model** | ResNet-18 (PyTorch, transfer learning from ImageNet) |
| **Explainability** | Grad-CAM (pytorch-grad-cam) |
| **Database** | SQLite (users + scan history) |
| **Deployment** | Docker + Gunicorn + Nginx |

## Quick Start

### Local Development

```bash
# Clone
git clone https://github.com/yourusername/ridgedx.git
cd ridgedx

# Install dependencies
cd Code
pip install -r requirements.txt

# Run
python app.py
# Visit http://localhost:5000
```

### Docker

```bash
cp .env.example .env
docker-compose up --build
# Visit http://localhost
```

## API Usage

```bash
# Predict blood group
curl -X POST http://localhost:5000/api/v1/predict \
  -F "file=@fingerprint.jpg"
```

**Response:**
```json
{
  "prediction": "A+",
  "confidence": 92.45,
  "all_probabilities": {"A+": 92.45, "A-": 2.31, ...},
  "grad_cam_base64": "<base64-png>",
  "inference_time_ms": 38.2
}
```

Interactive docs: `http://localhost:5000/api/docs`

## Model Performance

| Metric | Value |
|--------|-------|
| Architecture | ResNet-18 (modified: 1ch input, 8 classes) |
| Parameters | 11,174,344 |
| Test Accuracy | **88.07%** |
| Dataset | 5,976 fingerprint images |
| Training Time | 4.7 min (RTX 5050 GPU) |

### Per-Class Accuracy

| Blood Group | Accuracy |
|:-----------:|:--------:|
| B- | 99.1% |
| AB+ | 92.4% |
| B+ | 91.8% |
| O- | 88.8% |
| O+ | 87.5% |
| A+ | 86.9% |
| A- | 86.8% |
| AB- | 72.8% |

## Project Structure

```
├── Code/
│   ├── app.py                    # Flask backend + API
│   ├── config.py                 # Environment-based configuration
│   ├── train_model_v2.py         # ResNet-18 training pipeline
│   ├── resnet18_bloodgroup.pth   # Trained model weights (44MB)
│   ├── requirements.txt          # Python dependencies
│   ├── templates/
│   │   └── index_new.html        # Frontend SPA (~1800 lines)
│   └── static/
│       └── uploads/              # User-uploaded images
├── Dockerfile                    # Docker image
├── docker-compose.yml            # Multi-container setup
├── nginx.conf                    # Reverse proxy config
├── render.yaml                   # Render deployment blueprint
├── .env.example                  # Environment template
└── README.md
```

## Science Behind It

Dermatoglyphics research shows statistically significant correlations between fingerprint patterns (loops, whorls, arches) and ABO blood groups. Both are determined during fetal development and influenced by genes on **chromosome 9**. Our CNN learns these subtle pattern-blood group correlations from image data.

> **Disclaimer**: This is a screening/research tool, not a medical diagnostic device. Always confirm blood groups with standard laboratory testing.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

Built by **Adish** — a solo portfolio project demonstrating full-stack AI development.
