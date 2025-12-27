# SecurityGraph Tool-Calling Dataset

## Dataset Description

A synthetic dataset for training security tool-calling language models on Microsoft Security Graph API operations. Each example pairs a natural language security request with the corresponding JSON tool call.

### Dataset Summary

- **Source**: Microsoft Graph Security API specification
- **Format**: JSONL with instruction/input/output fields
- **Language**: English
- **Domain**: Security Operations, Incident Response, Threat Hunting
- **License**: MIT

### Languages

English

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `instruction` | string | Natural language security request |
| `input` | string | JSON tool definition with function schema |
| `output` | string | Expected JSON tool call with arguments |

### Example Instance

```json
{
  "instruction": "Get all high severity security alerts from the last 24 hours.",
  "input": "{\"type\": \"function\", \"function\": {\"name\": \"security_alerts_list\", \"description\": \"List security alerts from Microsoft Defender XDR.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"$filter\": {\"type\": \"string\"}, \"$top\": {\"type\": \"integer\"}}}}}",
  "output": "{\"name\": \"security_alerts_list\", \"arguments\": {\"$filter\": \"severity eq 'high'\", \"$top\": 50}}"
}
```

### Data Splits

| Split | Description |
|-------|-------------|
| train | Full dataset for training |

## Dataset Creation

### Curation Rationale

This dataset was created to train language models for security-focused tool calling with the Microsoft Security Graph API. The synthetic approach ensures diverse prompt templates while maintaining consistent, valid JSON outputs.

### Source Data

Generated from the [Microsoft Graph Security API specification](https://github.com/microsoftgraph/msgraph-metadata).

#### Security API Categories Covered

| Endpoint | Description |
|----------|-------------|
| `/security/alerts` | Security alert management |
| `/security/incidents` | Incident response |
| `/security/runHuntingQuery` | Advanced threat hunting |
| `/security/secureScores` | Security posture |
| `/riskyUsers` | Identity protection |
| `/signIns` | Authentication monitoring |
| `/security/tiIndicators` | Threat intelligence |
| `/auditLogs` | Compliance and audit |

#### Generation Process

1. Download Microsoft Graph Security OpenAPI YAML specification
2. Filter to security-focused endpoints
3. Generate diverse security analyst prompts using templates
4. Create JSON tool definitions and expected outputs
5. Export as JSONL format

### Annotations

The dataset is automatically generated from OpenAPI specifications. No manual annotation was performed.

## Considerations for Using the Data

### Social Impact

This dataset enables:
- Faster security incident response
- More accessible security operations for SOC analysts
- Automation of routine security tasks
- Natural language interfaces for security tools

### Discussion of Biases

- Dataset reflects the structure of Microsoft Security Graph API
- Synthetic prompts may not capture all real-world security phrasings
- Security-specific terminology may vary by organization
- Focused on English language only

### Limitations

- Does not include authentication/authorization examples
- Request body schemas are simplified
- Some complex nested parameters are flattened
- Does not include actual security data or indicators

### Risks

- Model trained on this data should not execute actions without human validation
- Improper use could lead to incorrect security responses
- Should not be used for automated security decisions

## Additional Information

### Dataset Curators

ftrout

### Licensing Information

MIT License

### Citation

```bibtex
@misc{securitygraph-dataset,
  title={Microsoft Security Graph Tool-Calling Dataset},
  author={ftrout},
  year={2025},
  url={https://github.com/ftrout/SecurityGraph-Agent}
}
```

### Related Projects

- [kodiak-secops-1](https://github.com/ftrout/kodiak-secops-1) - SOC alert triage model
- [SecurityGraph-Agent](https://github.com/ftrout/SecurityGraph-Agent) - Security tool-calling agent
