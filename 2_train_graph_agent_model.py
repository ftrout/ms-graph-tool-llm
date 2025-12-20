import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NEW_MODEL_NAME = "ms-graph-v1"
DATA_FILE = "./data/graph_tool_dataset.jsonl"
OUTPUT_DIR = "./results"

def format_chat_template(examples, tokenizer):
    """
    Applies the model's specific chat template to the instruction/input/output triplets.
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

def train():
    print(f"Loading Qwen 2.5 from {MODEL_ID}...")
    
    # 1. Load and Split Dataset
    # We use a simulated split strategy here. In production, split by 'tool_family'.
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    # 2. Quantization Config (4-bit for memory efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2"  # Requires A100/H100 or Ampere GPUs. Use "sdpa" or "eager" otherwise.
    )
    
    # Enable gradient checkpointing to save VRAM
    model = prepare_model_for_kbit_training(model)

    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right" # SFTTrainer requires right padding for completion

    # 5. LoRA Configuration
    peft_config = LoraConfig(
        r=32,                    # Rank: Higher r = more parameters to train (better for complex schemas)
        lora_alpha=64,           # Alpha: Scaling factor (usually 2x rank)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 6. Training Configuration (SFTConfig for TRL >= 0.15.0)
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

    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        formatting_func=lambda x: format_chat_template(x, tokenizer),
        args=sft_config
    )

    print("Starting Training...")
    trainer.train()
    
    print(f"Saving Model to {NEW_MODEL_NAME}...")
    trainer.model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)
    print("Done.")

if __name__ == "__main__":
    train()