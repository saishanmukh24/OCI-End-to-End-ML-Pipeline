# 🚀 End-to-End ML Pipeline on Oracle Cloud Infrastructure

[![Oracle Cloud](https://img.shields.io/badge/Oracle-Cloud-red?logo=oracle)](https://cloud.oracle.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker)](https://www.docker.com/)
[![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-88%25-success)]()

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [CI/CD Pipeline](#cicd-pipeline)
- [Contributing](#contributing)
- [License](#license)

## 📖 Project Overview

This project demonstrates a complete end-to-end machine learning pipeline built on **Oracle Cloud Infrastructure (OCI)**. The pipeline achieves **88% model accuracy** for predictive analytics tasks, leveraging cloud-native services, distributed computing, and automated MLOps practices.

The solution encompasses data ingestion, preprocessing, feature engineering, model training, hyperparameter optimization, deployment, and continuous monitoring—all orchestrated on OCI's robust infrastructure.

## ✨ Key Features

- 🔄 **Data Processing**: Comprehensive data preprocessing pipeline with feature selection and engineering for optimal model performance
- 🤖 **Model Training**: Random Forest and XGBoost algorithms implemented on distributed systems for scalable training
- ⚡ **AutoML Optimization**: Automated hyperparameter tuning reducing training time by 50% while maintaining high accuracy
- 🔁 **CI/CD Pipeline**: Fully automated pipeline for model retraining, deployment, and versioning
- 📊 **Monitoring Dashboards**: Real-time monitoring and alerting for model performance metrics
- 🐳 **Containerization**: Docker-based deployment for portability and scalability
- ☁️ **Cloud-Native**: Leverages OCI Data Science services for enterprise-grade ML operations

## 🏗️ Architecture

The pipeline follows a modular architecture:

```
Data Ingestion → Preprocessing → Feature Engineering → Model Training → Hyperparameter Tuning → Model Evaluation → Deployment → Monitoring
```

**Key Components:**
- **OCI Data Science**: Model development and training environment
- **OCI Object Storage**: Data lake for raw and processed data
- **OCI Container Registry**: Docker image storage
- **OCI Functions**: Serverless inference endpoints
- **OCI Monitoring**: Performance tracking and alerting

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Oracle Cloud Infrastructure (OCI)**: Cloud platform and infrastructure
- **OCI Data Science**: Managed ML platform
- **Scikit-learn**: Machine learning library
- **XGBoost**: Gradient boosting framework
- **Docker**: Containerization platform

### ML Frameworks & Libraries
- **Random Forest**: Ensemble learning algorithm
- **XGBoost**: Extreme gradient boosting
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization

### DevOps & MLOps
- **Git/GitHub**: Version control
- **Docker**: Container orchestration
- **OCI DevOps**: CI/CD automation
- **Jupyter Notebooks**: Interactive development

## 📁 Folder Structure

```
OCI-End-to-End-ML-Pipeline/
│
├── data/
│   ├── raw/                    # Raw dataset files
│   ├── processed/              # Cleaned and processed data
│   └── features/               # Engineered features
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   ├── utils/
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── helpers.py
│   │
│   └── deployment/
│       ├── inference.py
│       └── monitor.py
│
├── models/
│   ├── trained_models/         # Saved model artifacts
│   └── checkpoints/            # Training checkpoints
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
├── config/
│   ├── model_config.yaml
│   └── oci_config.yaml
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_inference.py
│
├── scripts/
│   ├── train_model.sh
│   ├── deploy_model.sh
│   └── run_pipeline.sh
│
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

## 📋 Prerequisites

- **Oracle Cloud Account** with access to:
  - OCI Data Science
  - OCI Object Storage
  - OCI Container Registry (optional)
- **Python 3.8+**
- **Docker** (for containerized deployment)
- **Git**
- **OCI CLI** (optional but recommended)

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/saishanmukh24/OCI-End-to-End-ML-Pipeline.git
cd OCI-End-to-End-ML-Pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure OCI Credentials

Set up your OCI configuration file (`~/.oci/config`):

```ini
[DEFAULT]
user=ocid1.user.oc1...
fingerprint=your_fingerprint
tenancy=ocid1.tenancy.oc1...
region=us-ashburn-1
key_file=~/.oci/oci_api_key.pem
```

### 5. Update Configuration Files

Edit `config/oci_config.yaml` and `config/model_config.yaml` with your specific settings:

```yaml
# Example: config/oci_config.yaml
compartment_id: "ocid1.compartment.oc1..."
project_id: "ocid1.datascienceproject.oc1..."
bucket_name: "ml-pipeline-data"
```

### 6. Prepare Data

Place your dataset in the `data/raw/` directory or configure the data loader to fetch from OCI Object Storage.

## 💻 Usage

### Training the Model

```bash
# Run the complete pipeline
python src/models/train.py --config config/model_config.yaml

# Or use the shell script
bash scripts/train_model.sh
```

### Making Predictions

```python
from src.models.predict import ModelPredictor

# Load the trained model
predictor = ModelPredictor(model_path="models/trained_models/best_model.pkl")

# Make predictions
predictions = predictor.predict(new_data)
```

### Running in Docker

```bash
# Build the Docker image
docker build -t oci-ml-pipeline:latest -f docker/Dockerfile .

# Run the container
docker run -v $(pwd)/data:/app/data oci-ml-pipeline:latest
```

### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Explore the notebooks in sequential order for a guided walkthrough of the pipeline.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 88% |
| **Precision** | 0.86 |
| **Recall** | 0.84 |
| **F1-Score** | 0.85 |
| **Training Time Reduction** | 50% (with AutoML) |

### Algorithm Comparison

| Algorithm | Accuracy | Training Time |
|-----------|----------|---------------|
| Random Forest | 85% | 45 min |
| XGBoost | 88% | 30 min |
| Logistic Regression | 78% | 10 min |

## 🔁 CI/CD Pipeline

The project includes automated CI/CD workflows:

1. **Continuous Integration**:
   - Automated testing on every commit
   - Code quality checks
   - Model validation

2. **Continuous Deployment**:
   - Automated model retraining on new data
   - Model versioning and registry
   - Blue-green deployment strategy
   - Performance monitoring and rollback

3. **Monitoring**:
   - Real-time model performance tracking
   - Data drift detection
   - Automated alerting

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Sai Shanmukh**
- GitHub: [@saishanmukh24](https://github.com/saishanmukh24)

## 📞 Contact

For questions or collaboration opportunities, please open an issue or reach out through GitHub.

---

**Project Date**: October 2025

⭐ If you find this project helpful, please consider giving it a star!
