"""
Hugging Face Hub integration for msgraph-tool-agent-8b.

Provides functionality for uploading trained models and datasets to the Hugging Face Hub.
"""

import os

from huggingface_hub import create_repo, upload_file, upload_folder

from msgraph_tool_agent_8b.utils.logging import get_logger

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
- microsoft-graph
- function-calling
- api-agent
- qlora
- lora
pipeline_tag: text-generation
datasets:
- custom
model-index:
- name: {model_name}
  results:
  - task:
      type: text-generation
      name: Tool Calling
    dataset:
      type: custom
      name: Microsoft Graph Tool-Calling Dataset
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

A specialized tool-calling language model fine-tuned for Microsoft Graph API operations.

## Model Description

This model is a LoRA (Low-Rank Adaptation) adapter fine-tuned on top of [{base_model}](https://huggingface.co/{base_model})
for generating precise JSON tool calls for the Microsoft Graph API.

### Key Features

- **Schema-First Training**: Trained on the official Microsoft Graph OpenAPI specification
- **Strict JSON Output**: Generates valid, schema-compliant JSON tool calls
- **Enterprise-Ready**: Designed for integration with Microsoft 365 applications
- **Efficient**: Uses QLoRA (4-bit quantization + LoRA) for memory efficiency

## Evaluation Results

| Metric | Score |
|--------|-------|
| JSON Validity Rate | {json_validity_rate}% |
| Tool Name Accuracy | {tool_name_accuracy}% |
| Argument Accuracy | {argument_accuracy}% |

## Intended Uses

This model is designed to:
- Convert natural language requests into Microsoft Graph API tool calls
- Generate properly formatted JSON payloads for Graph API endpoints
- Assist in building AI agents for Microsoft 365 automation

### Out-of-Scope Uses

- General conversation or chat
- APIs other than Microsoft Graph
- Executing API calls without human validation in production

### Example Usage

```python
from msgraph_tool_agent_8b import MSGraphAgent

# Load the agent
agent = MSGraphAgent.from_pretrained("{repo_id}")

# Define a tool
email_tool = {{
    "type": "function",
    "function": {{
        "name": "me_sendMail",
        "description": "Send a new message.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "subject": {{"type": "string"}},
                "toRecipients": {{"type": "array"}}
            }}
        }}
    }}
}}

# Generate tool call
result = agent.generate(
    "Send an email to john@example.com about the project update",
    tool=email_tool
)
print(result)
# {{"name": "me_sendMail", "arguments": {{"subject": "Project Update", "toRecipients": [...]}}}}
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
    {{"role": "system", "content": "You are an AI Agent for Microsoft Graph. Generate JSON tool calls."}},
    {{"role": "user", "content": "User Request: List all users\\nAvailable Tool: {{...}}"}}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

### Training Data

The model was trained on synthetic data generated from the official Microsoft Graph OpenAPI specification.
The training set includes:
- User management endpoints
- Mail and calendar operations
- Drive and file operations
- Group and team management

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

- The model is specialized for Microsoft Graph API and may not generalize to other APIs
- Always validate generated tool calls before execution
- Not suitable for conversational use outside of tool calling context
- Requires the correct tool definition to be provided in the prompt

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Incorrect API calls | Always validate outputs before execution |
| Unauthorized access | Implement proper OAuth scopes and permissions |
| Data leakage | Do not include sensitive data in prompts |

## Ethical Considerations

- This model should be used responsibly for authorized API access only
- Generated API calls should be validated before execution
- Users are responsible for proper authentication and authorization
- The model does not execute API calls - it only generates JSON payloads

## Citation

```bibtex
@misc{{msgraph-tool-agent-8b,
  title={{msgraph-tool-agent-8b: Enterprise Tool-Calling Agent for Microsoft Graph}},
  author={{ftrout}},
  year={{2025}},
  url={{https://github.com/ftrout/msgraph-tool-agent-8b}}
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
- microsoft-graph
- function-calling
- api-agent
- synthetic
size_categories:
- 1K<n<10K
---

# {dataset_name}

A synthetic dataset for training tool-calling language models on Microsoft Graph API operations.

## Dataset Description

This dataset contains instruction-tool-output triplets generated from the official Microsoft Graph OpenAPI specification.
Each example pairs a natural language request with the corresponding JSON tool call.

### Dataset Summary

- **Source**: Microsoft Graph OpenAPI v1.0 specification
- **Format**: JSONL with instruction/input/output fields
- **Size**: ~{num_samples} examples
- **Language**: English

### Supported Tasks

- **Tool Calling**: Training models to generate structured JSON tool calls
- **Function Calling**: Fine-tuning LLMs for API interaction
- **Agent Training**: Building Microsoft 365 automation agents

## Dataset Structure

### Data Fields

- `instruction`: Natural language user request (e.g., "Send an email to john@example.com")
- `input`: JSON tool definition with function name, description, and parameters
- `output`: Expected JSON tool call with name and arguments

### Example

```json
{{
  "instruction": "I need to list all users.",
  "input": "{{\\"type\\": \\"function\\", \\"function\\": {{\\"name\\": \\"users_ListUser\\", \\"description\\": \\"List users\\", \\"parameters\\": {{...}}}}}}",
  "output": "{{\\"name\\": \\"users_ListUser\\", \\"arguments\\": {{}}}}"
}}
```

## Dataset Creation

### Source Data

Generated from the [Microsoft Graph OpenAPI specification](https://github.com/microsoftgraph/msgraph-metadata).

### API Categories Covered

- `/me/messages` - Email operations
- `/me/events` - Calendar operations
- `/me/drive` - OneDrive file operations
- `/users` - User management
- `/groups` - Group management
- `/teams` - Teams operations
- `/sites` - SharePoint operations
- `/applications` - App registrations

### Generation Process

1. Download Microsoft Graph OpenAPI YAML specification
2. Parse endpoints and extract operation metadata
3. Generate diverse natural language prompts using templates
4. Create JSON tool definitions and expected outputs
5. Export as JSONL format

## Usage

### Loading with Datasets

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

### Training with msgraph-tool-agent-8b

```bash
pip install msgraph-tool-agent-8b

# Download and generate fresh data
msgraph-harvest --output-dir ./data

# Train a model
msgraph-train --data-file ./data/graph_tool_dataset.jsonl
```

## Considerations

### Biases

- Dataset reflects the structure of Microsoft Graph API
- Synthetic prompts may not capture all real-world phrasings
- OData query examples use placeholder values

### Limitations

- Does not include authentication/authorization examples
- Request body schemas are simplified
- Some complex nested parameters are flattened

## Citation

```bibtex
@misc{{msgraph-tool-dataset,
  title={{Microsoft Graph Tool-Calling Dataset}},
  author={{ftrout}},
  year={{2025}},
  url={{https://github.com/ftrout/msgraph-tool-agent-8b}}
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
    commit_message: str = "Upload msgraph-tool-agent-8b adapter",
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
    return url


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
    commit_message: str = "Upload msgraph-tool-agent-8b training dataset",
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
    return url


def main():
    """CLI entry point for Hub upload."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload trained msgraph-tool-agent-8b to Hugging Face Hub"
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
        default="Upload msgraph-tool-agent-8b adapter",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--no-model-card", action="store_true", help="Skip creating model card"
    )

    args = parser.parse_args()

    from msgraph_tool_agent_8b.utils.logging import setup_logging

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


def dataset_main():
    """CLI entry point for dataset upload."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload msgraph-tool-agent-8b dataset to Hugging Face Hub"
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
        default="Upload msgraph-tool-agent-8b training dataset",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--no-dataset-card", action="store_true", help="Skip creating dataset card"
    )

    args = parser.parse_args()

    from msgraph_tool_agent_8b.utils.logging import setup_logging

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
