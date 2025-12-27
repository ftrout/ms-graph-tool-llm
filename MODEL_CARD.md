# SecurityGraph-Agent Model Card

## Model Details

### Model Description

**SecurityGraph-Agent** is an enterprise-grade security tool-calling language model fine-tuned for Microsoft Security Graph API operations. It converts natural language security requests into precise, schema-validated JSON tool calls for Security Operations Center (SOC) workflows.

- **Model Type**: LoRA (Low-Rank Adaptation) adapter
- **Base Model**: [NousResearch/Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)
- **Fine-tuning Method**: QLoRA (4-bit quantization + LoRA)
- **Language**: English
- **License**: MIT

### Model Sources

- **Repository**: [github.com/ftrout/SecurityGraph-Agent](https://github.com/ftrout/SecurityGraph-Agent)
- **Related Projects**: [kodiak-secops-1](https://github.com/ftrout/kodiak-secops-1)

## Uses

### Direct Use

This model is designed for:
- Converting natural language security requests into Security Graph API tool calls
- Assisting SOC analysts with alert triage and incident response
- Enabling threat hunting through natural language queries
- Generating properly formatted JSON payloads for security operations

### Supported Security Operations

| Category | Operations |
|----------|------------|
| Alert Management | List, get, update security alerts |
| Incident Response | Manage and investigate security incidents |
| Threat Hunting | Run advanced hunting queries with KQL |
| Identity Protection | Monitor risky users and sign-in events |
| Security Posture | Check secure scores and compliance |
| Threat Intelligence | Query threat indicators (IOCs) |

### Downstream Use

The model can be integrated into:
- Security orchestration and automation platforms (SOAR)
- Custom SOC chatbots and assistants
- Incident response workflows
- Security analytics dashboards

### Out-of-Scope Use

- General conversation or chat
- Non-security Microsoft Graph API operations
- Executing API calls without human validation
- Automated security responses without human oversight

## Bias, Risks, and Limitations

### Technical Limitations

- Specialized for Microsoft Security Graph API endpoints only
- Requires correct tool definition to be provided in the prompt
- Not suitable for conversational use outside of tool calling context
- May not generalize to other security APIs

### Risks

| Risk | Mitigation |
|------|------------|
| Incorrect API calls | Always validate outputs before execution |
| Unauthorized access | Implement proper OAuth scopes and permissions |
| Data leakage | Do not include sensitive security data in prompts |
| False positives/negatives | Human review required for all security actions |

### Recommendations

- Always validate generated tool calls before execution
- Maintain human oversight for all security-critical operations
- Use proper authentication and authorization with Security Graph
- Do not include sensitive security data or indicators in prompts

## Training Details

### Training Data

The model was trained on synthetic data generated from the official Microsoft Graph Security API specification. The training set includes:
- Security alerts and incidents
- Threat intelligence indicators
- Advanced hunting queries
- Identity protection and risky users
- Secure scores and compliance
- Audit logs and sign-in events

### Training Procedure

| Parameter | Value |
|-----------|-------|
| Method | QLoRA (4-bit NF4 + LoRA) |
| LoRA Rank (r) | 32 |
| LoRA Alpha | 64 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 3 |
| Learning Rate | 1e-4 |
| Batch Size | 16 (effective) |
| Max Sequence Length | 2048 |
| Precision | bfloat16 |

### Training Infrastructure

- **Hardware**: NVIDIA GPU (16GB+ VRAM recommended)
- **Framework**: Hugging Face Transformers + PEFT + TRL
- **Environment**: Docker with NVIDIA container

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| JSON Validity Rate | Percentage of outputs that are valid JSON |
| Tool Name Accuracy | Percentage of correct tool names |
| Argument Accuracy | Percentage of correct argument structures |

### Expected Performance

The model achieves high accuracy on:
- Generating valid JSON tool calls (>95%)
- Selecting the correct tool name (>90%)
- Providing appropriate arguments (>85%)

## Environmental Impact

Carbon emissions estimated using the [ML CO2 Impact calculator](https://mlco2.github.io/impact/):
- **Hardware Type**: NVIDIA RTX 4090
- **Training Duration**: ~2-3 hours
- **Cloud Provider**: Local/Self-hosted
- **Carbon Emitted**: Minimal (single GPU training)

## Citation

```bibtex
@misc{securitygraph-agent,
  title={SecurityGraph-Agent: Enterprise Security Tool-Calling Agent for Microsoft Security Graph},
  author={ftrout},
  year={2025},
  url={https://github.com/ftrout/SecurityGraph-Agent}
}
```

## Model Card Contact

For questions or issues, please visit:
- GitHub Issues: [github.com/ftrout/SecurityGraph-Agent/issues](https://github.com/ftrout/SecurityGraph-Agent/issues)
