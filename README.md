# DefenderApi-Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Enterprise-Grade Security Tool-Calling Agent for Microsoft Defender XDR**

A specialized LLM fine-tuning pipeline that trains models to convert natural language security requests into precise, schema-validated JSON tool calls for the Microsoft Defender XDR API. Designed for Security Operations Centers (SOC), incident response, and threat hunting workflows.

## Key Features

- **Security-First Design**: Optimized for Microsoft Defender XDR, Security Graph API, and SOC operations
- **Schema-First Training**: Training data generated from official Microsoft Security API specifications
- **Strict JSON Output**: Produces valid, schema-compliant JSON tool calls every time
- **Efficient Training**: Uses QLoRA (4-bit quantization + LoRA) for training on consumer GPUs
- **Production Ready**: Comprehensive evaluation metrics, logging, and Hugging Face Hub integration
- **HuggingFace Integration**: Easy model and dataset sharing through Hugging Face Hub

## Security Operations Supported

| Category | Operations |
|----------|------------|
| **Alert Management** | List, get, update security alerts |
| **Incident Response** | Manage and investigate security incidents |
| **Threat Hunting** | Run advanced hunting queries with KQL |
| **Identity Protection** | Monitor risky users and sign-in events |
| **Security Posture** | Check secure scores and compliance |
| **Threat Intelligence** | Query threat indicators (IOCs) |
| **Audit & Compliance** | Access audit logs and directory events |

## Quick Start

### Installation

```bash
# With pip
pip install defender-api-tool

# With demo UI
pip install 'defender-api-tool[demo]'

# For development
pip install 'defender-api-tool[dev]'
```

### Docker (Recommended for Training)

```bash
# Clone the repository
git clone https://github.com/ftrout/defender-api-tool.git
cd defender-api-tool

# Start dev container with GPU support
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.08-py3 \
  pip install -e '.[dev]'
```

### Basic Usage

```python
from defender_api_tool import DefenderApiAgent

# Load a trained agent
agent = DefenderApiAgent.from_pretrained("./defender-api-tool")

# Or from Hugging Face Hub
agent = DefenderApiAgent.from_hub("ftrout/defender-api-tool")

# Define a security tool
alert_tool = {
    "type": "function",
    "function": {
        "name": "security_alerts_list",
        "description": "List security alerts from Microsoft Defender XDR.",
        "parameters": {
            "type": "object",
            "properties": {
                "$filter": {"type": "string", "description": "OData filter"},
                "$top": {"type": "integer", "description": "Number of results"}
            }
        }
    }
}

# Generate a tool call from natural language
result = agent.generate(
    "Get all high severity security alerts from the last 24 hours",
    tool=alert_tool
)
print(result)
# {"name": "security_alerts_list", "arguments": {"$filter": "severity eq 'high'", "$top": 50}}
```

## Training Pipeline

### Stage 1: Generate Training Data

```bash
# Generate security-focused training data from Microsoft Graph Security API
defender-harvest --output-dir ./data
```

### Stage 2: Fine-tune the Model

```bash
# Train with QLoRA (requires ~16GB VRAM)
defender-train \
  --data-file ./data/defender_tool_dataset.jsonl \
  --output-name defender-api-tool \
  --epochs 3
```

### Stage 3: Evaluate

```bash
# Run evaluation
defender-evaluate \
  --adapter-path ./defender-api-tool \
  --data-file ./data/test_dataset.jsonl
```

### Stage 4: Interactive Demo

```bash
# CLI interface
defender-agent --adapter-path ./defender-api-tool

# Gradio web UI
python demo.py --adapter-path ./defender-api-tool
```

## Model Architecture

### Base Model
- **Name**: Hermes-3-Llama-3.1-8B
- **Publisher**: NousResearch
- **Why Hermes 3?**: Purpose-built for function calling, trained on tool-use datasets

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Quantization | 4-bit NF4 (Normal Float) |
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning Rate | 1e-4 |
| Batch Size | 16 (effective) |
| Epochs | 3 |
| Max Sequence Length | 2048 |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **JSON Validity Rate** | Percentage of outputs that are valid JSON |
| **Tool Name Accuracy** | Percentage of correct tool names |
| **Argument Accuracy** | Percentage of correct argument structures |

## CLI Commands

| Command | Description |
|---------|-------------|
| `defender-harvest` | Generate training data from Security API spec |
| `defender-train` | Fine-tune the model with QLoRA |
| `defender-evaluate` | Evaluate model performance |
| `defender-agent` | Interactive CLI for testing |
| `defender-upload` | Upload model to Hugging Face Hub |
| `defender-upload-dataset` | Upload dataset to Hugging Face Hub |

## Project Structure

```
defender-api-tool/
├── src/
│   └── defender_api_tool/
│       ├── __init__.py           # Package exports
│       ├── hub.py                # Hugging Face Hub integration
│       ├── data/
│       │   └── harvester.py      # Security API data harvester
│       ├── training/
│       │   └── trainer.py        # QLoRA training pipeline
│       ├── evaluation/
│       │   └── evaluator.py      # Comprehensive evaluation
│       ├── inference/
│       │   └── agent.py          # DefenderApiAgent class
│       └── utils/
│           ├── config.py         # Configuration classes
│           └── logging.py        # Logging utilities
├── tests/                        # Unit tests
├── demo.py                       # Gradio web demo
├── MODEL_CARD.md                 # HuggingFace model card
├── DATASET_CARD.md               # HuggingFace dataset card
├── pyproject.toml                # Package configuration
└── README.md                     # This file
```

## Development

### Setup

```bash
# Clone and install
git clone https://github.com/ftrout/defender-api-tool.git
cd defender-api-tool
pip install -e '.[dev]'

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Format code
black src tests

# Lint
ruff check src tests

# Type check
mypy src

# Run tests
pytest tests -v
```

## Hardware Requirements

### Training
- **GPU**: 16GB+ VRAM (RTX 3090, 4090, A10G recommended)
- **OS**: Linux (Ubuntu 22.04 LTS) or WSL2
- **Python**: 3.10+
- **CUDA**: 11.8+

### Inference
- **GPU**: 8GB+ VRAM (RTX 3070+ or equivalent)
- **Python**: 3.10+

## API Reference

### DefenderApiAgent

```python
class DefenderApiAgent:
    """Microsoft Defender XDR API tool-calling agent."""

    @classmethod
    def from_pretrained(cls, adapter_path: str, ...) -> "DefenderApiAgent":
        """Load a trained agent from a saved adapter."""

    @classmethod
    def from_hub(cls, repo_id: str, ...) -> "DefenderApiAgent":
        """Load a trained agent from Hugging Face Hub."""

    def generate(
        self,
        instruction: str,
        tool: Dict[str, Any],
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        return_dict: bool = True
    ) -> Union[Dict[str, Any], str]:
        """Generate a tool call for the given instruction."""
```

### DefenderAPIHarvester

```python
class DefenderAPIHarvester:
    """Harvests Microsoft Defender XDR API specifications for training."""

    def harvest(
        self,
        output_dir: str = "./data",
        output_filename: str = "defender_tool_dataset.jsonl"
    ) -> str:
        """Download and process the API spec into training data."""
```

### DefenderToolTrainer

```python
class DefenderToolTrainer:
    """Trainer for fine-tuning LLMs on Defender XDR tool calling."""

    def train(
        self,
        data_file: str,
        output_name: str,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None
    ) -> str:
        """Train the model and save the adapter."""
```

### DefenderToolEvaluator

```python
class DefenderToolEvaluator:
    """Evaluator for Defender XDR tool-calling models."""

    def evaluate_dataset(
        self,
        data_file: str,
        max_samples: Optional[int] = None,
        save_results: Optional[str] = None
    ) -> EvaluationMetrics:
        """Evaluate the model on a dataset."""
```

## Related Projects

- [kodiak-secops-1](https://github.com/ftrout/kodiak-secops-1) - SOC alert triage model for security operations

## Disclaimer

This tool is intended for authorized security operations only. While the model is trained to be accurate, always validate AI-generated API calls before executing them in a production environment. The authors are not responsible for unintended actions performed by the agent against live Microsoft Defender XDR tenants.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- Additional security API endpoints
- Improved prompt templates
- Evaluation benchmarks
- Documentation improvements
- Bug fixes

## Security

For security concerns, please see [SECURITY.md](SECURITY.md).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@misc{defender-api-tool,
  title={DefenderApi-Tool: Enterprise Security Tool-Calling Agent for Microsoft Defender XDR},
  author={ftrout},
  year={2025},
  url={https://github.com/ftrout/defender-api-tool}
}
```

## Acknowledgments

- [NousResearch](https://huggingface.co/NousResearch) for Hermes-3-Llama-3.1-8B
- [Microsoft](https://github.com/microsoftgraph) for Graph Security API specifications
- [Hugging Face](https://huggingface.co) for transformers, PEFT, and TRL libraries
