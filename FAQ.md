# Frequently Asked Questions (FAQ)

## General Questions

### What is SecurityGraph-Agent?

SecurityGraph-Agent is an enterprise-grade security tool-calling language model specifically fine-tuned for Microsoft Security Graph API operations. It converts natural language security requests into precise, schema-validated JSON tool calls for Security Operations Center (SOC) workflows, including alert management, incident response, and threat hunting.

### What can I use SecurityGraph-Agent for?

Common use cases include:
- **Alert Triage**: Generate API calls to list, filter, and update security alerts
- **Incident Response**: Query and manage security incidents programmatically
- **Threat Hunting**: Create advanced hunting queries using natural language
- **Security Monitoring**: Monitor risky users, sign-in events, and security posture
- **Automation**: Integrate with SOAR platforms for automated security workflows

### Is SecurityGraph-Agent production-ready?

Yes, SecurityGraph-Agent is designed for production use. However, you should:
- Always validate AI-generated outputs before execution
- Implement proper authentication and authorization
- Maintain human oversight for security-critical operations
- Test in non-production environments first

---

## Installation & Setup

### What are the system requirements?

**For Training:**
- GPU: 16GB+ VRAM (NVIDIA RTX 3090, 4090, or A10G recommended)
- OS: Linux (Ubuntu 22.04 LTS) or WSL2
- Python: 3.10+
- CUDA: 11.8+

**For Inference:**
- GPU: 8GB+ VRAM (RTX 3070+ or equivalent)
- Python: 3.10+

### How do I install SecurityGraph-Agent?

```bash
# Basic installation
pip install securitygraph-agent

# With Gradio demo UI
pip install 'securitygraph-agent[demo]'

# For development
pip install 'securitygraph-agent[dev]'
```

### Can I use SecurityGraph-Agent without a GPU?

While possible, inference without a GPU will be significantly slower. For production use, a CUDA-capable GPU is strongly recommended. Training requires a GPU with at least 16GB VRAM.

### How do I use the model from Hugging Face Hub?

```python
from defender_api_tool import DefenderApiAgent

# Load from Hugging Face Hub
agent = DefenderApiAgent.from_hub("fmt0816/securitygraph-agent")

# Generate a tool call
result = agent.generate(
    "Get all high severity security alerts",
    tool=alert_tool_definition
)
```

---

## Training & Fine-tuning

### What base model does SecurityGraph-Agent use?

SecurityGraph-Agent uses [NousResearch/Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B) as the base model. This model was chosen because it's specifically designed for function calling and tool use tasks.

### What training method is used?

SecurityGraph-Agent uses QLoRA (Quantized Low-Rank Adaptation):
- 4-bit NF4 quantization for memory efficiency
- LoRA rank of 32 with alpha of 64
- Targets attention and MLP layers
- Enables training on consumer GPUs (16GB VRAM)

### How do I generate training data?

```bash
# Generate training data from Microsoft Graph Security API specification
secgraph-harvest --output-dir ./data
```

This command downloads the official Microsoft Graph Security API specification and generates synthetic training examples covering all supported security endpoints.

### Can I train on custom data?

Yes. The training data format is JSONL with three fields:
- `instruction`: Natural language security request
- `input`: JSON tool definition
- `output`: Expected JSON tool call

You can add your own examples following this format.

### How long does training take?

With default settings on an NVIDIA RTX 4090:
- Training time: ~1-3 hours for 3 epochs
- Dataset size: ~1,000+ synthetic examples

---

## Model Usage

### What security operations are supported?

| Operation | Description |
|-----------|-------------|
| Alert Management | List, get, update security alerts |
| Incident Response | Manage and investigate incidents |
| Threat Hunting | Run advanced hunting queries with KQL |
| Identity Protection | Monitor risky users and sign-in events |
| Security Posture | Check secure scores and compliance |
| Threat Intelligence | Query threat indicators (IOCs) |
| Audit & Compliance | Access audit logs and directory events |

### How do I define custom tools?

```python
custom_tool = {
    "type": "function",
    "function": {
        "name": "my_security_function",
        "description": "Description of what this function does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Parameter description"},
                "param2": {"type": "integer", "description": "Another parameter"}
            },
            "required": ["param1"]
        }
    }
}

result = agent.generate("Your natural language request", tool=custom_tool)
```

### Why am I getting invalid JSON output?

Common causes:
1. **Prompt too complex**: Simplify your request
2. **Missing tool definition**: Ensure you provide a valid tool schema
3. **Temperature too high**: Lower the temperature (default is 0.1)
4. **Max tokens too low**: Increase `max_new_tokens` parameter

### Can I use multiple tools in one request?

Yes, use the `tools` parameter:

```python
result = agent.generate(
    "Your request",
    tools=[tool1, tool2, tool3]
)
```

The model will select the most appropriate tool.

---

## Hugging Face Integration

### How do I upload my trained model to Hugging Face Hub?

```bash
secgraph-upload ./my-adapter username/model-name
```

Or with Python:

```python
from defender_api_tool.hub import upload_to_hub

upload_to_hub(
    adapter_path="./my-adapter",
    repo_id="username/model-name",
    private=False
)
```

### How do I upload my dataset to Hugging Face Hub?

```bash
secgraph-upload-dataset ./data/defender_tool_dataset.jsonl username/dataset-name
```

### What metadata should I include for Hugging Face?

The upload commands automatically generate:
- Model cards with training details and evaluation metrics
- Dataset cards with schema information
- YAML frontmatter with proper tags and categories

---

## Security & Privacy

### Does the model execute API calls?

No. SecurityGraph-Agent only generates JSON payloads representing API calls. It does not execute any calls against live systems. You must implement the actual API execution in your application.

### What data is sent to the model?

The model processes:
- Your natural language security request
- Tool definitions you provide

Do NOT include:
- Actual API keys or tokens
- Sensitive security data or indicators
- Personal identifiable information (PII)
- Real incident details or alert data

### How should I handle the generated outputs?

1. **Validate** all generated JSON before execution
2. **Review** API parameters for correctness
3. **Test** in non-production environments first
4. **Log** all executed API calls for audit
5. **Implement** proper OAuth scopes with least privilege

---

## Troubleshooting

### "CUDA out of memory" error during training

Solutions:
- Reduce batch size with `--batch-size 1`
- Use gradient accumulation to maintain effective batch size
- Ensure no other GPU processes are running
- Consider using a GPU with more VRAM

### "Adapter not found" error

Ensure you've trained a model first:
```bash
secgraph-train --data-file ./data/defender_tool_dataset.jsonl --output-name my-adapter
```

Or download from Hub:
```python
agent = DefenderApiAgent.from_hub("fmt0816/securitygraph-agent")
```

### Model outputs look wrong or nonsensical

- Ensure you're using a compatible base model
- Check that the adapter was trained correctly
- Verify the tool definition format is correct
- Try lowering temperature to 0.0 for deterministic output

### Gradio demo not loading

Install the demo dependencies:
```bash
pip install 'securitygraph-agent[demo]'
```

---

## Performance & Optimization

### How can I speed up inference?

1. **Use GPU**: Ensure CUDA is properly configured
2. **Merge adapter**: The agent automatically merges LoRA weights for faster inference
3. **Batch requests**: Process multiple requests together when possible
4. **Reduce max tokens**: Lower `max_new_tokens` if outputs are shorter

### What's the expected accuracy?

On the synthetic evaluation dataset:
- **JSON Validity Rate**: >95%
- **Tool Name Accuracy**: >90%
- **Argument Accuracy**: >85%

Real-world performance may vary based on prompt complexity and domain specificity.

---

## Licensing & Usage

### What license is SecurityGraph-Agent under?

SecurityGraph-Agent is released under the MIT License, allowing free use, modification, and distribution for both commercial and non-commercial purposes.

### Can I use this for commercial purposes?

Yes, the MIT License permits commercial use. However, you are responsible for:
- Validating outputs before production use
- Ensuring proper Microsoft Graph API authorization
- Compliance with your organization's security policies

### How should I cite SecurityGraph-Agent?

```bibtex
@misc{securitygraph-agent,
  title={SecurityGraph-Agent: Enterprise Security Tool-Calling Agent for Microsoft Security Graph},
  author={ftrout},
  year={2025},
  url={https://github.com/ftrout/SecurityGraph-Agent}
}
```

---

## Getting Help

### Where can I report issues?

Open an issue on GitHub: [github.com/ftrout/SecurityGraph-Agent/issues](https://github.com/ftrout/SecurityGraph-Agent/issues)

### Where can I find the source code?

GitHub: [github.com/ftrout/SecurityGraph-Agent](https://github.com/ftrout/SecurityGraph-Agent)

### Where is the model hosted?

Hugging Face Hub: [huggingface.co/fmt0816/securitygraph-agent](https://huggingface.co/fmt0816/securitygraph-agent)
