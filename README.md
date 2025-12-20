# Microsoft Graph AI Agent Model

**An Enterprise-Grade Tool-Calling AI for Microsoft Graph**

This is a specialized AI pipeline that fine-tunes the **Qwen 2.5 (7B)** Large Language Model to function as an autonomous agent for the Microsoft Graph API. Unlike generic chatbots, this model is trained via **Schema-First Learning** to reliably translate natural language user intent into precise, schema-validated JSON tool calls.

---

### 🚀 Key Features

* **Schema-First Training:** Ingests the official Microsoft Graph OpenAPI specification to generate high-fidelity training data.
* **Strict JSON Compliance:** Fine-tuned to output valid JSON objects that strictly adhere to API tool definitions.
* **State-of-the-Art Foundation:** Built on **Qwen 2.5 7B Instruct**, leveraging its superior coding and logic capabilities over Llama 3 or Mistral.
* **Efficient Fine-Tuning:** Uses **QLoRA** (4-bit quantization + LoRA) to train on consumer-grade hardware (24GB VRAM recommended).
* **ChatML Optimized:** Utilizes the specific `<|im_start|>` prompt format for maximum instruction-following performance.

---

### 🛠️ Prerequisites

### Hardware Requirements

* **Training:** NVIDIA GPU with **≥16GB VRAM** (RTX 3090, 4090, or A10G recommended).
* **Inference:** NVIDIA GPU with **≥8GB VRAM**.
* **Disk Space:** ~20GB for model weights and datasets.

### Software Requirements

* **OS:** Linux (Ubuntu 22.04 LTS recommended) or WSL2.
* **Python:** 3.10+.
* **CUDA:** 11.8 or newer.

---

### 📦 Installation

1. **Clone the Repository**
```bash
git clone https://github.com/ftrout/ms-graph-ai-agent-model
cd ms-graph-ai-agent-model
```


2. **Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate
```


3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

### 🔄 The Pipeline

#### Step 1: Data Harvesting (`1_graph_api_harvester.py`)

This script downloads the official Microsoft Graph OpenAPI (Swagger) specification and converts it into a training dataset. It maps API definitions to synthetic user prompts (e.g., "I need to send an email" \rightarrow `me_sendMail` schema).

```bash
python 1_graph_api_harvester.py
```

* **Input:** `https://raw.githubusercontent.com.../openapi.yaml`
* **Output:** `./data/graph_tool_dataset.jsonl`

#### Step 2: Fine-Tuning (`2_train_graph_agent_model.py`)

Trains the Qwen 2.5 model using QLoRA. This step teaches the model to ignore conversational fluff and focus strictly on generating valid JSON tool calls.

```bash
python 2_train_graph_agent_model.py
```

* **Base Model:** `Qwen/Qwen2.5-7B-Instruct`
* **Output:** `./ms-graph-v1` (The trained LoRA adapter)
* **Duration:** ~1-3 hours on an RTX 3090.

#### Step 3: Evaluation (`3_evaluate_model.py`)

Runs the "Judge" script to validate the model's performance against a known tool definition. It checks for:

1. Valid JSON syntax.
2. Correct argument handling (e.g., ensuring OData filters like `$filter` are applied correctly).

```bash
python 3_evaluate_model.py
```

#### Step 4: Interactive Demo (`4_interactive_graph.py`)

Launch the CLI to interact with your agent in real-time. You can type commands like *"Find email from admin@contoso.com"* and see the generated JSON payload.

```bash
python 4_interactive_graph.py
```

---

### 📂 Project Structure

```text
ms-graph-ai-agent-model/
├── data/                      # Generated training datasets
│   └── graph_tool_dataset.jsonl
├── ms-graph-v1/               # Saved Model Adapters (after training)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── results/                   # Checkpoints during training
├── 1_graph_api_harvester.py   # Data Engine
├── 2_train_graph_agent_model.py  # Training Lab
├── 3_evaluate_model.py        # The Judge
├── 4_interactive_graph.py     # Interactive CLI
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

---

### 🧠 Model Architecture Details

| Component | Specification |
| --- | --- |
| **Base Model** | Qwen 2.5 7B Instruct |
| **Format** | ChatML |
| **Quantization** | 4-bit NF4 (Normal Float 4) |
| **Adapter Rank (r)** | 32 |
| **Target Modules** | `all-linear` (q, k, v, o, gate, up, down) |
| **Context Window** | 2048 tokens (optimized for tool calls) |

---

### ⚠️ Disclaimer

This tool is intended for research and development purposes. While the model is trained to be secure, always validate AI-generated API calls before executing them in a production environment. The authors are not responsible for unintended actions performed by the agent against live Microsoft Graph tenants.