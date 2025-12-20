import torch
import os
import logging
import argparse
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

# --- DEFAULT CONFIGURATION ---
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL_NAME = "ms-graph-v1"
DEFAULT_DATA_FILE = "./data/graph_tool_dataset.jsonl"
DEFAULT_OUTPUT_DIR = "./results"

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

def train(
    model_id: str,
    output_name: str,
    data_file: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_rank: int
) -> None:
    """
    Fine-tunes Qwen 2.5 7B model for Microsoft Graph tool calling using QLoRA.

    Args:
        model_id: Base model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")
        output_name: Name for the output adapter directory
        data_file: Path to training dataset JSONL file
        output_dir: Directory for training checkpoints
        epochs: Number of training epochs
        batch_size: Per-device training batch size
        learning_rate: Learning rate for optimizer
        lora_rank: LoRA adapter rank

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
    logger.info("Starting training pipeline for %s", model_id)

    # 1. Validate Dataset Exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Dataset not found at {data_file}. "
            f"Please run 1_graph_api_harvester.py first to generate training data."
        )

    # 2. Load and Split Dataset
    # We use a simulated split strategy here. In production, split by 'tool_family'.
    logger.info("Loading dataset from %s...", data_file)
    dataset = load_dataset("json", data_files=data_file, split="train")

    if len(dataset) == 0:
        raise ValueError(
            f"Dataset is empty! Check {data_file} and regenerate if necessary."
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
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        logger.info("Successfully loaded with Flash Attention 2")
    except Exception as e:
        logger.warning("Flash Attention 2 not available, falling back to SDPA: %s", e)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa"
        )
        logger.info("Successfully loaded with SDPA attention")

    # Enable gradient checkpointing to save VRAM
    model = prepare_model_for_kbit_training(model)

    # 5. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # SFTTrainer requires right padding for completion

    # 6. LoRA Configuration
    peft_config = LoraConfig(
        r=lora_rank,             # Rank: Higher r = more parameters to train (better for complex schemas)
        lora_alpha=lora_rank * 2,  # Alpha: Scaling factor (usually 2x rank)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 7. Training Configuration (SFTConfig for TRL >= 0.15.0)
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,   # Adjust based on VRAM
        gradient_accumulation_steps=4,   # Effective batch size = batch_size * 4
        learning_rate=learning_rate,
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

    logger.info("Training complete! Saving model to %s...", output_name)
    trainer.model.save_pretrained(output_name)
    tokenizer.save_pretrained(output_name)
    logger.info("Model saved successfully to %s", output_name)

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen 2.5 7B for Microsoft Graph tool calling using QLoRA"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Base model identifier (default: {DEFAULT_MODEL_ID})"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Output adapter directory name (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=DEFAULT_DATA_FILE,
        help=f"Path to training dataset JSONL file (default: {DEFAULT_DATA_FILE})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for training checkpoints (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size (default: 4)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer (default: 1e-4)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA adapter rank (default: 32)"
    )

    args = parser.parse_args()

    logger.info("Configuration:")
    logger.info("  Base model: %s", args.model_id)
    logger.info("  Output name: %s", args.output_name)
    logger.info("  Data file: %s", args.data_file)
    logger.info("  Output directory: %s", args.output_dir)
    logger.info("  Epochs: %d", args.epochs)
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("  Learning rate: %.2e", args.learning_rate)
    logger.info("  LoRA rank: %d", args.lora_rank)

    train(
        model_id=args.model_id,
        output_name=args.output_name,
        data_file=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank
    )

if __name__ == "__main__":
    main()