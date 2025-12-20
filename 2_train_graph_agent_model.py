import torch
import os
import logging
from typing import Dict, List, Any
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NEW_MODEL_NAME = "ms-graph-v1"
DATA_FILE = "./data/graph_tool_dataset.jsonl"
OUTPUT_DIR = "./results"

def format_chat_template(examples: Dict[str, List[str]], tokenizer: Any) -> List[str]:
    """
    Applies the model's chat template to instruction/input/output triplets.

    Converts training examples into the ChatML format expected by Qwen models,
    with system prompt, user request (with tool definition), and assistant response.

    Args:
        examples: Batch of examples with 'instruction', 'input', 'output' keys
        tokenizer: Tokenizer with apply_chat_template method

    Returns:
        List[str]: Formatted chat strings with special tokens (<|im_start|>, etc.)
    """
    texts = []
    # Handle batch processing
    instructions = examples['instruction']
    inputs = examples['input']
    outputs = examples['output']

    for i in range(len(instructions)):
        messages = [
            {"role": "system", "content": "You are an AI Agent for Microsoft Graph. Given the user request and the available tool definition, generate the correct JSON tool call."},
            {"role": "user", "content": f"User Request: {instructions[i]}\nAvailable Tool: {inputs[i]}"},
            {"role": "assistant", "content": outputs[i]}
        ]
        # tokenize=False returns the raw string formatted with <|im_start|>, etc.
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    return texts

def train() -> None:
    """
    Fine-tunes Qwen 2.5 7B model for Microsoft Graph tool calling using QLoRA.

    This function:
    1. Validates and loads the training dataset
    2. Configures 4-bit quantization for memory efficiency
    3. Loads the base model with Flash Attention 2 (or SDPA fallback)
    4. Applies LoRA adapters to key layers
    5. Trains using supervised fine-tuning (SFT)
    6. Saves the trained adapter weights

    Raises:
        FileNotFoundError: If training dataset doesn't exist
        ValueError: If dataset is empty
    """
    logger.info("Starting training pipeline for %s", MODEL_ID)

    # 1. Validate Dataset Exists
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. "
            f"Please run 1_graph_api_harvester.py first to generate training data."
        )

    # 2. Load and Split Dataset
    # We use a simulated split strategy here. In production, split by 'tool_family'.
    logger.info("Loading dataset from %s...", DATA_FILE)
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    if len(dataset) == 0:
        raise ValueError(
            f"Dataset is empty! Check {DATA_FILE} and regenerate if necessary."
        )

    logger.info("Loaded %d training examples", len(dataset))
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    logger.info("Split: %d train, %d test", len(dataset["train"]), len(dataset["test"]))

    # 3. Quantization Config (4-bit for memory efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 4. Load Model
    # Try Flash Attention 2, fallback to SDPA if unavailable
    try:
        logger.info("Attempting to load model with Flash Attention 2...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        logger.info("Successfully loaded with Flash Attention 2")
    except Exception as e:
        logger.warning("Flash Attention 2 not available, falling back to SDPA: %s", e)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa"
        )
        logger.info("Successfully loaded with SDPA attention")
    
    # Enable gradient checkpointing to save VRAM
    model = prepare_model_for_kbit_training(model)

    # 5. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # SFTTrainer requires right padding for completion

    # 6. LoRA Configuration
    peft_config = LoraConfig(
        r=32,                    # Rank: Higher r = more parameters to train (better for complex schemas)
        lora_alpha=64,           # Alpha: Scaling factor (usually 2x rank)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 7. Training Configuration (SFTConfig for TRL >= 0.15.0)
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,   # Adjust based on VRAM
        gradient_accumulation_steps=4,   # Effective batch size = 4*4 = 16
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=False,
        bf16=True,                       # Use BFloat16 for stability
        logging_steps=10,
        max_seq_length=2048,             # Context window size
        packing=False,                   # False is safer for instruction tuning with distinct samples
        report_to="none",                # Change to "wandb" for tracking
        dataset_text_field="text",       # Placeholder (formatting_func overrides this)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False} # Important for modern PyTorch
    )

    # 8. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        formatting_func=lambda x: format_chat_template(x, tokenizer),
        args=sft_config
    )

    logger.info("Starting training...")
    logger.info("Training configuration: %d epochs, batch size %d, gradient accumulation %d",
                sft_config.num_train_epochs,
                sft_config.per_device_train_batch_size,
                sft_config.gradient_accumulation_steps)

    trainer.train()

    logger.info("Training complete! Saving model to %s...", NEW_MODEL_NAME)
    trainer.model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)
    logger.info("Model saved successfully to %s", NEW_MODEL_NAME)

if __name__ == "__main__":
    train()