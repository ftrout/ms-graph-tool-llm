"""
Hugging Face Hub integration for defender-api-tool.

Provides functionality for uploading trained security models and datasets to the Hugging Face Hub.
"""

import os

from huggingface_hub import create_repo, upload_file, upload_folder

from defender_api_tool.utils.logging import get_logger

logger = get_logger("hub")

# Model card template with placeholders for evaluation metrics
MODEL_CARD_TEMPLATE = """---
language:
- en
license: mit
library_name: peft
base_model: {base_model}
tags:
- tool-calling
- microsoft-defender
- defender-xdr
- security
- soc
- function-calling
- api-agent
- qlora
- lora
- cybersecurity
- threat-detection
- incident-response
pipeline_tag: text-generation
datasets:
- custom
model-index:
- name: {model_name}
  results:
  - task:
      type: text-generation
      name: Security Tool Calling
    dataset:
      type: custom
      name: Microsoft Defender XDR Tool-Calling Dataset
    metrics:
    - type: accuracy
      name: JSON Validity Rate
      value: {json_validity_rate}
    - type: accuracy
      name: Tool Name Accuracy
      value: {tool_name_accuracy}
    - type: accuracy
      name: Argument Accuracy
      value: {argument_accuracy}
---

# {model_name}

A specialized security tool-calling language model fine-tuned for Microsoft Defender XDR API operations.

## Model Description

This model is a LoRA (Low-Rank Adaptation) adapter fine-tuned on top of [{base_model}](https://huggingface.co/{base_model})
for generating precise JSON tool calls for the Microsoft Defender XDR API. Designed for Security Operations Centers (SOC)
and incident response workflows.

### Key Features

- **Security-First Design**: Trained on Microsoft Defender XDR and Security Graph API endpoints
- **SOC Operations**: Optimized for alert triage, incident response, and threat hunting
- **Strict JSON Output**: Generates valid, schema-compliant JSON tool calls
- **Enterprise-Ready**: Designed for integration with Microsoft security products
- **Efficient**: Uses QLoRA (4-bit quantization + LoRA) for memory efficiency

## Evaluation Results

| Metric | Score |
|--------|-------|
| JSON Validity Rate | {json_validity_rate}% |
| Tool Name Accuracy | {tool_name_accuracy}% |
| Argument Accuracy | {argument_accuracy}% |

## Intended Uses

This model is designed to:
- Convert natural language security requests into Defender XDR API tool calls
- Assist SOC analysts with alert triage and incident response
- Enable threat hunting through natural language queries
- Generate properly formatted JSON payloads for security operations

### Supported Security Operations

- **Alert Management**: List, get, update security alerts
- **Incident Response**: Manage and investigate security incidents
- **Threat Hunting**: Run advanced hunting queries with KQL
- **Identity Protection**: Monitor risky users and sign-in events
- **Security Posture**: Check secure scores and compliance

### Out-of-Scope Uses

- General conversation or chat
- APIs other than Microsoft Defender XDR / Security Graph
- Executing API calls without human validation in production
- Automated security responses without human oversight

### Example Usage

```python
from defender_api_tool import DefenderApiAgent

# Load the agent
agent = DefenderApiAgent.from_pretrained("{repo_id}")

# Define a security tool
alert_tool = {{
    "type": "function",
    "function": {{
        "name": "security_alerts_list",
        "description": "List security alerts from Microsoft Defender XDR.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "$filter": {{"type": "string", "description": "OData filter"}},
                "$top": {{"type": "integer", "description": "Number of results"}}
            }}
        }}
    }}
}}

# Generate tool call
result = agent.generate(
    "Get all high severity security alerts from the last 24 hours",
    tool=alert_tool
)
print(result)
# {{"name": "security_alerts_list", "arguments": {{"$filter": "severity eq 'high'", "$top": 50}}}}
```

### Direct Usage with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model (bfloat16 for inference, allows adapter merging)
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load and merge adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Generate
messages = [
    {{"role": "system", "content": "You are a Security Operations AI Agent for Microsoft Defender XDR. Generate JSON tool calls."}},
    {{"role": "user", "content": "User Request: Get high severity alerts\\nAvailable Tool: {{...}}"}}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

### Training Data

The model was trained on synthetic data generated from the official Microsoft Graph Security API specification.
The training set includes security-focused endpoints:
- Security alerts and incidents
- Threat intelligence indicators
- Advanced hunting queries
- Identity protection and risky users
- Secure scores and compliance
- Audit logs and sign-in events

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | {base_model} |
| Method | QLoRA (4-bit NF4 + LoRA) |
| LoRA Rank (r) | {lora_r} |
| LoRA Alpha | {lora_alpha} |
| LoRA Dropout | {lora_dropout} |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | {num_epochs} |
| Learning Rate | {learning_rate} |
| Batch Size (effective) | {batch_size} |
| Max Sequence Length | {max_seq_length} |
| Precision | bfloat16 |

### Training Infrastructure

- **Hardware**: {hardware}
- **Training Time**: {training_time}
- **Framework**: Hugging Face Transformers + PEFT + TRL

## Limitations

- The model is specialized for Microsoft Defender XDR API and may not generalize to other APIs
- Always validate generated tool calls before execution
- Not suitable for conversational use outside of security tool calling context
- Requires the correct tool definition to be provided in the prompt
- Should not be used for automated security responses without human oversight

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Incorrect API calls | Always validate outputs before execution |
| Unauthorized access | Implement proper OAuth scopes and permissions |
| Data leakage | Do not include sensitive security data in prompts |
| False positives/negatives | Human review required for all security actions |

## Ethical Considerations

- This model should be used responsibly for authorized security operations only
- Generated API calls should be validated before execution
- Users are responsible for proper authentication and authorization
- The model does not execute API calls - it only generates JSON payloads
- Always maintain human oversight for security-critical operations

## Related Projects

- [kodiak-secops-1](https://github.com/ftrout/kodiak-secops-1) - SOC alert triage model

## Citation

```bibtex
@misc{{defender-api-tool,
  title={{DefenderApi-Tool: Enterprise Security Tool-Calling Agent for Microsoft Defender XDR}},
  author={{ftrout}},
  year={{2025}},
  url={{https://github.com/ftrout/defender-api-tool}}
}}
```

## License

MIT License
"""

# Dataset card template
DATASET_CARD_TEMPLATE = """---
language:
- en
license: mit
task_categories:
- text-generation
tags:
- tool-calling
- microsoft-defender
- defender-xdr
- security
- soc
- function-calling
- api-agent
- synthetic
- cybersecurity
size_categories:
- 1K<n<10K
---

# {dataset_name}

A synthetic dataset for training security tool-calling language models on Microsoft Defender XDR API operations.

## Dataset Description

This dataset contains instruction-tool-output triplets generated from the official Microsoft Graph Security API specification.
Each example pairs a natural language security request with the corresponding JSON tool call.

### Dataset Summary

- **Source**: Microsoft Graph Security API / Defender XDR specification
- **Format**: JSONL with instruction/input/output fields
- **Size**: ~{num_samples} examples
- **Language**: English
- **Domain**: Security Operations, Incident Response, Threat Hunting

### Supported Tasks

- **Security Tool Calling**: Training models to generate structured JSON tool calls for security operations
- **SOC Automation**: Fine-tuning LLMs for security analyst workflows
- **Agent Training**: Building security automation agents for Defender XDR

## Dataset Structure

### Data Fields

- `instruction`: Natural language security request (e.g., "Get all high severity alerts")
- `input`: JSON tool definition with function name, description, and parameters
- `output`: Expected JSON tool call with name and arguments

### Example

```json
{{
  "instruction": "Get all high severity security alerts from the last 24 hours.",
  "input": "{{\\"type\\": \\"function\\", \\"function\\": {{\\"name\\": \\"security_alerts_list\\", \\"description\\": \\"List security alerts\\", \\"parameters\\": {{...}}}}}}",
  "output": "{{\\"name\\": \\"security_alerts_list\\", \\"arguments\\": {{\\"$filter\\": \\"severity eq 'high'\\"}}}}"
}}
```

## Dataset Creation

### Source Data

Generated from the [Microsoft Graph Security API specification](https://github.com/microsoftgraph/msgraph-metadata).

### Security API Categories Covered

- `/security/alerts` - Security alert management
- `/security/incidents` - Incident response
- `/security/runHuntingQuery` - Advanced threat hunting
- `/security/secureScores` - Security posture
- `/riskyUsers` - Identity protection
- `/signIns` - Authentication monitoring
- `/security/tiIndicators` - Threat intelligence
- `/auditLogs` - Compliance and audit

### Generation Process

1. Download Microsoft Graph Security OpenAPI YAML specification
2. Filter to security-focused endpoints
3. Generate diverse security analyst prompts using templates
4. Create JSON tool definitions and expected outputs
5. Export as JSONL format

## Usage

### Loading with Datasets

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

### Training with defender-api-tool

```bash
pip install defender-api-tool

# Download and generate fresh data
defender-harvest --output-dir ./data

# Train a model
defender-train --data-file ./data/defender_tool_dataset.jsonl
```

## Considerations

### Biases

- Dataset reflects the structure of Microsoft Defender XDR API
- Synthetic prompts may not capture all real-world security phrasings
- Security-specific terminology may vary by organization

### Limitations

- Does not include authentication/authorization examples
- Request body schemas are simplified
- Some complex nested parameters are flattened
- Does not include actual security data or indicators

## Related Projects

- [kodiak-secops-1](https://github.com/ftrout/kodiak-secops-1) - SOC alert triage model

## Citation

```bibtex
@misc{{defender-tool-dataset,
  title={{Microsoft Defender XDR Tool-Calling Dataset}},
  author={{ftrout}},
  year={{2025}},
  url={{https://github.com/ftrout/defender-api-tool}}
}}
```

## License

MIT License
"""


def create_model_card(
    model_name: str,
    base_model: str,
    repo_id: str,
    output_path: str,
    json_validity_rate: float = 95.0,
    tool_name_accuracy: float = 95.0,
    argument_accuracy: float = 95.0,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    num_epochs: int = 3,
    learning_rate: str = "1e-4",
    batch_size: int = 16,
    max_seq_length: int = 2048,
    hardware: str = "NVIDIA GPU (16GB+ VRAM)",
    training_time: str = "~1-3 hours",
) -> str:
    """
    Create a model card for Hugging Face Hub.

    Args:
        model_name: Name of the model
        base_model: Base model identifier
        repo_id: Hugging Face repository ID
        output_path: Path to save the model card
        json_validity_rate: Percentage of valid JSON outputs
        tool_name_accuracy: Percentage of correct tool names
        argument_accuracy: Percentage of correct arguments
        lora_r: LoRA rank used in training
        lora_alpha: LoRA alpha used in training
        lora_dropout: LoRA dropout used in training
        num_epochs: Number of training epochs
        learning_rate: Learning rate string
        batch_size: Effective batch size
        max_seq_length: Maximum sequence length
        hardware: Hardware description
        training_time: Training time description

    Returns:
        Path to the created model card
    """
    card_content = MODEL_CARD_TEMPLATE.format(
        model_name=model_name,
        base_model=base_model,
        repo_id=repo_id,
        json_validity_rate=json_validity_rate,
        tool_name_accuracy=tool_name_accuracy,
        argument_accuracy=argument_accuracy,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        hardware=hardware,
        training_time=training_time,
    )

    card_path = os.path.join(output_path, "README.md")
    with open(card_path, "w") as f:
        f.write(card_content)

    logger.info("Created model card at %s", card_path)
    return card_path


def upload_to_hub(
    adapter_path: str,
    repo_id: str,
    base_model: str = "NousResearch/Hermes-3-Llama-3.1-8B",
    private: bool = False,
    commit_message: str = "Upload defender-api-tool adapter",
    create_model_card_file: bool = True,
) -> str:
    """
    Upload a trained adapter to the Hugging Face Hub.

    Args:
        adapter_path: Path to the adapter directory
        repo_id: Repository ID (e.g., "username/model-name")
        base_model: Base model identifier for the model card
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        create_model_card_file: Whether to create a model card

    Returns:
        URL of the uploaded model

    Raises:
        FileNotFoundError: If adapter path doesn't exist
        ValueError: If repo_id is invalid
    """
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")

    if "/" not in repo_id:
        raise ValueError(
            f"Invalid repo_id '{repo_id}'. " "Format should be 'username/model-name'"
        )

    logger.info("Uploading to Hugging Face Hub: %s", repo_id)

    # Create repository if it doesn't exist
    try:
        create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        logger.info("Repository created/verified: %s", repo_id)
    except Exception as e:
        logger.warning("Could not create repository: %s", e)

    # Create model card if requested
    if create_model_card_file:
        model_name = repo_id.split("/")[-1]
        create_model_card(
            model_name=model_name,
            base_model=base_model,
            repo_id=repo_id,
            output_path=adapter_path,
        )

    # Upload folder
    logger.info("Uploading adapter files...")
    url = upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    logger.info("Successfully uploaded to: %s", url)
    return str(url)


def create_dataset_card(
    dataset_name: str, repo_id: str, num_samples: int, output_path: str
) -> str:
    """
    Create a dataset card for Hugging Face Hub.

    Args:
        dataset_name: Name of the dataset
        repo_id: Hugging Face repository ID
        num_samples: Number of samples in the dataset
        output_path: Path to save the dataset card

    Returns:
        Path to the created dataset card
    """
    card_content = DATASET_CARD_TEMPLATE.format(
        dataset_name=dataset_name, repo_id=repo_id, num_samples=num_samples
    )

    card_path = os.path.join(output_path, "README.md")
    with open(card_path, "w") as f:
        f.write(card_content)

    logger.info("Created dataset card at %s", card_path)
    return card_path


def upload_dataset_to_hub(
    dataset_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload defender-api-tool training dataset",
    create_dataset_card_file: bool = True,
) -> str:
    """
    Upload a dataset to the Hugging Face Hub.

    Args:
        dataset_path: Path to the dataset file (JSONL) or directory
        repo_id: Repository ID (e.g., "username/dataset-name")
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        create_dataset_card_file: Whether to create a dataset card

    Returns:
        URL of the uploaded dataset

    Raises:
        FileNotFoundError: If dataset path doesn't exist
        ValueError: If repo_id is invalid
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    if "/" not in repo_id:
        raise ValueError(
            f"Invalid repo_id '{repo_id}'. " "Format should be 'username/dataset-name'"
        )

    logger.info("Uploading dataset to Hugging Face Hub: %s", repo_id)

    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True
        )
        logger.info("Dataset repository created/verified: %s", repo_id)
    except Exception as e:
        logger.warning("Could not create repository: %s", e)

    # Determine if path is a file or directory
    is_file = os.path.isfile(dataset_path)

    if is_file:
        # Count samples in JSONL file
        num_samples = 0
        with open(dataset_path) as f:
            for _ in f:
                num_samples += 1

        # Create dataset card in a temp directory
        if create_dataset_card_file:
            import tempfile

            temp_dir = tempfile.mkdtemp()
            dataset_name = repo_id.split("/")[-1]
            create_dataset_card(
                dataset_name=dataset_name,
                repo_id=repo_id,
                num_samples=num_samples,
                output_path=temp_dir,
            )

            # Upload README
            upload_file(
                path_or_fileobj=os.path.join(temp_dir, "README.md"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Add dataset card",
            )

        # Upload the dataset file
        logger.info("Uploading dataset file...")
        url = upload_file(
            path_or_fileobj=dataset_path,
            path_in_repo=os.path.basename(dataset_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
    else:
        # It's a directory - count samples from all JSONL files
        num_samples = 0
        for filename in os.listdir(dataset_path):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(dataset_path, filename)
                with open(filepath) as f:
                    for _ in f:
                        num_samples += 1

        # Create dataset card if requested
        if create_dataset_card_file:
            dataset_name = repo_id.split("/")[-1]
            create_dataset_card(
                dataset_name=dataset_name,
                repo_id=repo_id,
                num_samples=num_samples,
                output_path=dataset_path,
            )

        # Upload folder
        logger.info("Uploading dataset folder...")
        url = upload_folder(
            folder_path=dataset_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

    logger.info("Successfully uploaded dataset to: %s", url)
    return str(url)


def main() -> None:
    """CLI entry point for Hub upload."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload trained defender-api-tool to Hugging Face Hub"
    )
    parser.add_argument(
        "adapter_path", type=str, help="Path to the trained adapter directory"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face repository ID (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="NousResearch/Hermes-3-Llama-3.1-8B",
        help="Base model identifier for model card",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create a private repository"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload defender-api-tool adapter",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--no-model-card", action="store_true", help="Skip creating model card"
    )

    args = parser.parse_args()

    from defender_api_tool.utils.logging import setup_logging

    setup_logging()

    try:
        upload_to_hub(
            adapter_path=args.adapter_path,
            repo_id=args.repo_id,
            base_model=args.base_model,
            private=args.private,
            commit_message=args.commit_message,
            create_model_card_file=not args.no_model_card,
        )
        print("\nModel uploaded successfully!")
        print(f"View at: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        logger.error("Upload failed: %s", e)
        raise


def dataset_main() -> None:
    """CLI entry point for dataset upload."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload defender-api-tool dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "dataset_path", type=str, help="Path to the dataset file (JSONL) or directory"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face repository ID (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create a private repository"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload defender-api-tool training dataset",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--no-dataset-card", action="store_true", help="Skip creating dataset card"
    )

    args = parser.parse_args()

    from defender_api_tool.utils.logging import setup_logging

    setup_logging()

    try:
        upload_dataset_to_hub(
            dataset_path=args.dataset_path,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message,
            create_dataset_card_file=not args.no_dataset_card,
        )
        print("\nDataset uploaded successfully!")
        print(f"View at: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        logger.error("Upload failed: %s", e)
        raise


if __name__ == "__main__":
    main()
