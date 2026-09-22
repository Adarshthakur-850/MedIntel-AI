# MedIntel AI — Multimodal Medical Intelligence Platform

> [!IMPORTANT]
> **Research & Educational Disclaimer**: MedIntel AI is a research and educational decision-support demonstration system. It does not provide medical diagnosis or treatment recommendations and must not replace a qualified healthcare professional. All data used is synthetic or properly de-identified.

---

## 1. Project Overview
**MedIntel AI** is a production-quality, research-oriented multimodal healthcare AI platform. It integrates structured patient/lab measurements, chest X-ray image analysis, clinical notes processing, and curated medical knowledge retrieval (RAG) to generate an AI-assisted multimodal risk assessment and transparent model explanation.

---

## 2. Core Features & Modalities

### 📊 Clinical Tabular ML
- Models: Logistic Regression, Random Forest, XGBoost, LightGBM, and Deep Neural Networks.
- Features: Age, Blood Pressure (Sys/Dia), Heart Rate, Temperature, Glucose, Cholesterol, Hemoglobin, Creatinine, BMI, Sex, Smoking Status, Diabetes History.
- Explainability: Local and global SHAP (SHapley Additive exPlanations) values.

### 🖼️ Medical Computer Vision
- Architecture: ResNet18 / EfficientNet PyTorch backbones.
- Explainability: Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps highlighting image regions driving model risk scores.

### 📝 Clinical NLP
- Processing: Transformers & Named Entity Recognition (NER) extracting Symptoms, Duration, Conditions, and Medications from clinical notes.
- Feature Extraction: Semantic text embeddings for multimodal fusion.

### 🔀 Multimodal Fusion Engine
- Architecture: Combines Tabular Encoder + Image Encoder + Text Encoder.
- Fusion Methods: Weighted Late Fusion & Learned PyTorch Neural Fusion.

### 📚 Medical Knowledge RAG
- Engine: Sentence Transformers embeddings paired with FAISS vector retrieval.
- Capabilities: Interactive chat assistant providing cited answers from medical consensus guidelines with explicit disclaimers against medical prescribing.

### 💻 Production Full-Stack Application
- **Backend**: FastAPI REST API with Pydantic schemas, JWT authentication, rate limiting, and PostgreSQL storage.
- **Frontend**: React + TypeScript interactive dashboard with visual risk indicators, SHAP charts, Grad-CAM overlays, and RAG chat interface.

---

## 3. System Architecture

```
                                  MedIntel AI Architecture
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   React + TypeScript Frontend                               │
│     (Patient Overview | AI Analysis | Explainability | RAG Assistant | Model Registry)       │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │ HTTP / REST APIs
┌──────────────────────────────────────────────▼──────────────────────────────────────────────┐
│                                     FastAPI Backend API                                     │
│            (Auth JWT | Patient Service | Inference Service | RAG Service | Audit)           │
└────────┬──────────────────────┬──────────────────────┬──────────────────────┬───────────────┘
         │                      │                      │                      │
┌────────▼────────┐    ┌────────▼────────┐    ┌────────▼────────┐    ┌────────▼────────┐
│   Tabular ML    │    │ Vision Engine   │    │   NLP Engine    │    │  RAG Knowledge  │
│ (XGB/RF/LGB/NN) │    │ (PyTorch/ResNet)│    │ (Transformers)  │    │ (FAISS/Embed)   │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │                      │
         └──────────────────────┴──────────┬───────────┴──────────────────────┘
                                           │
                                ┌──────────▼──────────┐
                                │  Multimodal Fusion  │
                                │ (Late & Neural Fused│
                                └──────────┬──────────┘
                                           │
                                ┌──────────▼──────────┐
                                │   Explainability    │
                                │  (SHAP & Grad-CAM)  │
                                └─────────────────────┘
```

---

## 4. Repository Structure

```
medintel-ai/
│
├── apps/
│   ├── frontend/         # React + TypeScript Dashboard
│   └── backend/          # FastAPI REST API Backend
│
├── ml/
│   ├── tabular/          # Tabular ML training & evaluation
│   ├── vision/           # PyTorch computer vision & dataset loader
│   ├── nlp/              # Clinical NLP & entity extraction
│   ├── multimodal/       # Late & Neural Multimodal Fusion models
│   └── explainability/   # SHAP & Grad-CAM explainability generators
│
├── rag/                  # Medical RAG pipeline, vector store & LLM chat
│
├── data/
│   ├── raw/              # Raw synthetic datasets
│   ├── processed/        # Processed data arrays
│   └── schemas/          # Data schemas & validation
│
├── pipelines/            # Data generation & quality validator scripts
├── monitoring/           # Prometheus/Grafana configs
├── tests/                # Unit, integration, ML & end-to-end tests
├── configs/              # System YAML configuration
├── docker/               # Dockerfiles for frontend/backend
├── docker-compose.yml    # Full-stack Docker orchestration
├── pyproject.toml        # Dependencies & package metadata
├── Makefile              # Development shortcuts
└── README.md
```

---

## 5. Quick Start & Setup

### Prerequisites
- Python >= 3.10
- Node.js >= 18 & npm (for frontend)
- Docker & Docker Compose (optional for containerized deployment)

### 1. Installation
```bash
# Clone repository
git clone https://github.com/Adarshthakur-850/Event_Impact_AI.git
cd Event_Impact_AI

# Install Python dependencies
py -3.13 -m pip install -e .[dev]
```

### 2. Generate Synthetic Multimodal Dataset
```bash
py -3.13 pipelines/data_generator.py
```

### 3. Train ML Models
```bash
py -3.13 ml/tabular/train.py
py -3.13 ml/vision/train.py
```

### 4. Run Backend & Frontend
```bash
# Terminal 1: FastAPI Backend
py -3.13 -m uvicorn apps.backend.main:app --reload --port 8000

# Terminal 2: React Frontend
cd apps/frontend
npm install
npm start
```

---

## 6. Docker Deployment
Launch all services (Backend, Frontend, PostgreSQL, Redis, Prometheus) with Docker Compose:
```bash
docker-compose up --build
```
Access the application:
- Frontend Dashboard: `http://localhost:3000`
- FastAPI Docs: `http://localhost:8000/docs`
- Prometheus Metrics: `http://localhost:9090`

---

## 7. Testing & Quality Assurance
Run the test suite with pytest:
```bash
py -3.13 -m pytest tests/
```

---

## 8. License
This project is licensed under the MIT License — see the [LICENSE](file:///d:/lpu/project%20ml/MedIntel%20AI/LICENSE) file for details.
