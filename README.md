# msgraph-tool-agent-8b

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Enterprise-Grade Tool-Calling LLM for Microsoft Graph API**

msgraph-tool-agent-8b is a specialized fine-tuning pipeline that trains language models to convert natural language into precise, schema-validated JSON tool calls for the Microsoft Graph API. Unlike generic chatbots, this model uses **Schema-First Learning** to reliably translate user intent into valid API operations.

## Features

- **Schema-First Training**: Automatically generates training data from the official Microsoft Graph OpenAPI specification
- **Strict JSON Compliance**: Fine-tuned to output valid JSON objects that strictly adhere to API tool definitions
- **State-of-the-Art Foundation**: Built on **Hermes 3 Llama 3.1 8B** - purpose-built for function calling with superior structured output
- **Efficient Fine-Tuning**: Uses **QLoRA** (4-bit quantization + LoRA) to train on consumer-grade hardware
- **Production Ready**: Comprehensive evaluation metrics, logging, and Hugging Face Hub integration
- **Modular Design**: Clean package structure with reusable components

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ftrout/msgraph-tool-agent-8b
cd msgraph-tool-agent-8b

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from msgraph_tool_agent_8b import MSGraphAgent

# Load a trained agent
agent = MSGraphAgent.from_pretrained("./msgraph-tool-agent-8b")

# Define a tool
email_tool = {
    "type": "function",
    "function": {
        "name": "me_sendMail",
        "description": "Send a new message.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "toRecipients": {"type": "array"}
            }
        }
    }
}

# Generate a tool call
result = agent.generate(
    "Send an email to john@example.com about the project update",
    tool=email_tool
)
print(result)
# {"name": "me_sendMail", "arguments": {"subject": "Project Update", "toRecipients": [...]}}
```

## Training Pipeline

The complete pipeline consists of four stages:

### Stage 1: Data Harvesting

Downloads the Microsoft Graph OpenAPI specification and generates training data:

```bash
# Using CLI
msgraph-harvest --output-dir ./data

# Or using Python
from msgraph_tool_agent_8b import GraphAPIHarvester

harvester = GraphAPIHarvester()
harvester.harvest(output_dir="./data")
```

**Output**: `./data/graph_tool_dataset.jsonl`

### Stage 2: Fine-Tuning

Trains the model using QLoRA:

```bash
# Using CLI
msgraph-train \
    --data-file ./data/graph_tool_dataset.jsonl \
    --output-name msgraph-tool-agent-8b \
    --epochs 3 \
    --batch-size 4

# Or using Python
from msgraph_tool_agent_8b import GraphToolTrainer

trainer = GraphToolTrainer()
trainer.train(
    data_file="./data/graph_tool_dataset.jsonl",
    output_name="msgraph-tool-agent-8b"
)
```

**Output**: `./msgraph-tool-agent-8b/` (LoRA adapter)

### Stage 3: Evaluation

Evaluates the trained model:

```bash
# Single query evaluation
msgraph-evaluate \
    --adapter-path ./msgraph-tool-agent-8b \
    --query "List all users with displayName starting with 'A'"

# Dataset evaluation with metrics
msgraph-evaluate \
    --adapter-path ./msgraph-tool-agent-8b \
    --data-file ./data/test_dataset.jsonl \
    --save-results ./evaluation_results.json
```

### Stage 4: Interactive Demo

Launch the interactive agent:

```bash
msgraph-agent --adapter-path ./msgraph-tool-agent-8b
```

## Uploading to Hugging Face Hub

Upload your trained model to share with the community:

```bash
msgraph-upload ./msgraph-tool-agent-8b username/msgraph-tool-agent-8b
```

Or programmatically:

```python
from msgraph_tool_agent_8b.hub import upload_to_hub

upload_to_hub(
    adapter_path="./msgraph-tool-agent-8b",
    repo_id="username/msgraph-tool-agent-8b",
    private=False
)
```

## Model Architecture

| Component | Specification |
|-----------|---------------|
| **Base Model** | NousResearch/Hermes-3-Llama-3.1-8B |
| **Format** | ChatML (Hermes format) |
| **Quantization** | 4-bit NF4 (Normal Float 4) |
| **Adapter Rank (r)** | 32 |
| **Adapter Alpha** | 64 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Context Window** | 2048 tokens |
| **Attention** | Flash Attention 2 (with SDPA fallback) |

### Why Hermes 3?

- **Purpose-built for function calling**: Trained specifically on tool-use datasets
- **Reliable JSON output**: Less fine-tuning needed for structured output
- **Strong base performance**: Llama 3.1 foundation with enhanced instruction following
- **Active development**: NousResearch actively maintains and improves the model

## Project Structure

```
msgraph-tool-agent-8b/
├── src/
│   └── msgraph_tool_agent_8b/
│       ├── __init__.py           # Package exports
│       ├── hub.py                # Hugging Face Hub integration
│       ├── data/
│       │   ├── __init__.py
│       │   └── harvester.py      # OpenAPI data harvester
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py        # QLoRA training pipeline
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── evaluator.py      # Comprehensive evaluation
│       ├── inference/
│       │   ├── __init__.py
│       │   └── agent.py          # MSGraphAgent class
│       └── utils/
│           ├── __init__.py
│           ├── config.py         # Configuration classes
│           └── logging.py        # Logging utilities
├── tests/                        # Unit tests
├── pyproject.toml               # Package configuration
├── README.md                    # This file
└── LICENSE                      # MIT License
```

## Requirements

### Hardware

| Use Case | GPU VRAM | Notes |
|----------|----------|-------|
| Training | 16GB+ | RTX 3090, 4090, A10G recommended |
| Inference | 8GB+ | RTX 3070+ or equivalent |
| Disk Space | ~20GB | Model weights and datasets |

### Software

- **OS**: Linux (Ubuntu 22.04 LTS recommended) or WSL2
- **Python**: 3.10+
- **CUDA**: 11.8 or newer

## Configuration

### Model Configuration

```python
from msgraph_tool_agent_8b.utils.config import ModelConfig

config = ModelConfig(
    base_model_id="NousResearch/Hermes-3-Llama-3.1-8B",
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    max_seq_length=2048
)
```

### Training Configuration

```python
from msgraph_tool_agent_8b.utils.config import TrainingConfig

config = TrainingConfig(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    bf16=True
)
```

## Evaluation Metrics

The evaluation framework provides comprehensive metrics:

```
============================================================
EVALUATION METRICS
============================================================
Total Samples:        1000
────────────────────────────────────────────────────────────
JSON Validity Rate:   98.50% (985/1000)
Tool Name Accuracy:   97.20% (972/1000)
Argument Accuracy:    95.80% (958/1000)
────────────────────────────────────────────────────────────
Overall Accuracy:     95.80%
────────────────────────────────────────────────────────────
Avg Inference Time:   0.245s
Tokens/Second:        156.3
============================================================
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/msgraph_tool_agent_8b --cov-report=html
```

### Code Style

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

## API Reference

### MSGraphAgent

```python
class MSGraphAgent:
    """Microsoft Graph API tool-calling agent."""

    @classmethod
    def from_pretrained(cls, adapter_path: str, ...) -> "MSGraphAgent":
        """Load a trained agent from a saved adapter."""

    @classmethod
    def from_hub(cls, repo_id: str, ...) -> "MSGraphAgent":
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

### GraphAPIHarvester

```python
class GraphAPIHarvester:
    """Harvests Microsoft Graph API specifications for training."""

    def harvest(
        self,
        output_dir: str = "./data",
        output_filename: str = "graph_tool_dataset.jsonl"
    ) -> str:
        """Download and process the API spec into training data."""
```

### GraphToolTrainer

```python
class GraphToolTrainer:
    """Trainer for fine-tuning LLMs on Microsoft Graph tool calling."""

    def train(
        self,
        data_file: str,
        output_name: str,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None
    ) -> str:
        """Train the model and save the adapter."""
```

### GraphToolEvaluator

```python
class GraphToolEvaluator:
    """Evaluator for Microsoft Graph tool-calling models."""

    def evaluate_dataset(
        self,
        data_file: str,
        max_samples: Optional[int] = None,
        save_results: Optional[str] = None
    ) -> EvaluationMetrics:
        """Evaluate the model on a dataset."""
```

## Disclaimer

This tool is intended for research and development purposes. While the model is trained to be secure, always validate AI-generated API calls before executing them in a production environment. The authors are not responsible for unintended actions performed by the agent against live Microsoft Graph tenants.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@misc{msgraph-tool-agent-8b,
  title={msgraph-tool-agent-8b: Enterprise Tool-Calling Agent for Microsoft Graph},
  author={ftrout},
  year={2024},
  url={https://github.com/ftrout/msgraph-tool-agent-8b}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
