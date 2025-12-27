# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-01

### Added

- **Core Training Pipeline**
  - QLoRA (4-bit quantization + LoRA) fine-tuning for memory-efficient training
  - Support for NousResearch/Hermes-3-Llama-3.1-8B base model
  - Configurable training parameters via `ModelConfig` and `TrainingConfig`

- **Data Harvesting**
  - `DefenderAPIHarvester` for generating training data from Microsoft Graph Security API
  - Support for 26+ security-focused API endpoints
  - 18 diverse prompt templates for training data generation
  - Randomized example argument generation to prevent overfitting

- **Inference**
  - `DefenderApiAgent` class for generating tool calls from natural language
  - Support for loading from local adapters or Hugging Face Hub
  - Common security tool definitions included (alerts, incidents, hunting, etc.)

- **Evaluation**
  - `DefenderToolEvaluator` with comprehensive metrics
  - JSON validity rate, tool name accuracy, argument accuracy
  - Batch evaluation and result export

- **Hugging Face Integration**
  - Model upload with auto-generated model cards
  - Dataset upload with dataset cards
  - YAML frontmatter for Hugging Face Hub compatibility

- **CLI Commands**
  - `secgraph-harvest`: Generate training data
  - `secgraph-train`: Fine-tune the model
  - `secgraph-evaluate`: Evaluate model performance
  - `secgraph-agent`: Interactive CLI
  - `secgraph-upload`: Upload model to Hub
  - `secgraph-upload-dataset`: Upload dataset to Hub

- **Gradio Demo**
  - Web-based interface for testing tool generation
  - Example prompts for common security operations

- **Documentation**
  - Comprehensive README with quick start guide
  - MODEL_CARD.md for Hugging Face Hub
  - DATASET_CARD.md for dataset documentation
  - CONTRIBUTING.md with development guidelines
  - SECURITY.md with vulnerability reporting procedures
  - CODE_OF_CONDUCT.md for community guidelines

- **Development Tools**
  - Pre-commit hooks (Black, Ruff, MyPy)
  - GitHub Actions CI/CD pipeline
  - Dev container configuration with GPU support
  - Dependabot for automated dependency updates

### Security Operations Supported

- Alert Management (list, get, update security alerts)
- Incident Response (manage and investigate incidents)
- Threat Hunting (run advanced hunting queries with KQL)
- Identity Protection (monitor risky users and sign-in events)
- Security Posture (check secure scores and compliance)
- Threat Intelligence (query threat indicators/IOCs)
- Audit & Compliance (access audit logs and directory events)

## [Unreleased]

### Planned

- Support for additional base models
- Enhanced evaluation metrics
- SOAR platform integration adapters
- Real-world alert dataset integration
- Multi-language prompt support
