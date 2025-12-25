"""
Hugging Face Hub integration for msgraph-tool-agent-8b.

Provides functionality for uploading trained models to the Hugging Face Hub.
"""

import os
from typing import Optional

from huggingface_hub import HfApi, create_repo, upload_folder

from msgraph_tool_llm.utils.logging import get_logger

logger = get_logger("hub")

# Model card template
MODEL_CARD_TEMPLATE = '''---
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
  results: []
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

## Intended Uses

This model is designed to:
- Convert natural language requests into Microsoft Graph API tool calls
- Generate properly formatted JSON payloads for Graph API endpoints
- Assist in building AI agents for Microsoft 365 automation

### Example Usage

```python
from msgraph_tool_llm import MSGraphAgent

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

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
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

### Training Procedure

- **Base Model**: {base_model}
- **Method**: QLoRA (4-bit quantization + LoRA)
- **LoRA Rank**: 32
- **LoRA Alpha**: 64
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Epochs**: 3
- **Learning Rate**: 1e-4
- **Batch Size**: 16 (effective)

### Hardware

- GPU: NVIDIA RTX 3090/4090 or equivalent (16GB+ VRAM)
- Training Time: ~1-3 hours

## Limitations

- The model is specialized for Microsoft Graph API and may not generalize to other APIs
- Always validate generated tool calls before execution
- Not suitable for conversational use outside of tool calling context
- Requires the correct tool definition to be provided in the prompt

## Ethical Considerations

- This model should be used responsibly for authorized API access only
- Generated API calls should be validated before execution
- Users are responsible for proper authentication and authorization

## Citation

```bibtex
@misc{{msgraph-tool-agent-8b,
  title={{msgraph-tool-agent-8b: Enterprise Tool-Calling Agent for Microsoft Graph}},
  author={{ftrout}},
  year={{2024}},
  url={{https://github.com/ftrout/msgraph-tool-agent-8b}}
}}
```

## License

MIT License
'''


def create_model_card(
    model_name: str,
    base_model: str,
    repo_id: str,
    output_path: str
) -> str:
    """
    Create a model card for Hugging Face Hub.

    Args:
        model_name: Name of the model
        base_model: Base model identifier
        repo_id: Hugging Face repository ID
        output_path: Path to save the model card

    Returns:
        Path to the created model card
    """
    card_content = MODEL_CARD_TEMPLATE.format(
        model_name=model_name,
        base_model=base_model,
        repo_id=repo_id
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
    create_model_card_file: bool = True
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
            f"Invalid repo_id '{repo_id}'. "
            "Format should be 'username/model-name'"
        )

    logger.info("Uploading to Hugging Face Hub: %s", repo_id)

    # Initialize API
    api = HfApi()

    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
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
            output_path=adapter_path
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


def main():
    """CLI entry point for Hub upload."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload trained msgraph-tool-agent-8b to Hugging Face Hub"
    )
    parser.add_argument(
        "adapter_path",
        type=str,
        help="Path to the trained adapter directory"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face repository ID (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="NousResearch/Hermes-3-Llama-3.1-8B",
        help="Base model identifier for model card"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload msgraph-tool-agent-8b adapter",
        help="Commit message for the upload"
    )
    parser.add_argument(
        "--no-model-card",
        action="store_true",
        help="Skip creating model card"
    )

    args = parser.parse_args()

    from msgraph_tool_llm.utils.logging import setup_logging
    setup_logging()

    try:
        url = upload_to_hub(
            adapter_path=args.adapter_path,
            repo_id=args.repo_id,
            base_model=args.base_model,
            private=args.private,
            commit_message=args.commit_message,
            create_model_card_file=not args.no_model_card
        )
        print(f"\nModel uploaded successfully!")
        print(f"View at: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        logger.error("Upload failed: %s", e)
        raise


if __name__ == "__main__":
    main()
