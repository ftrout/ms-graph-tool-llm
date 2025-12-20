import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NEW_MODEL_NAME = "ms-graph-v1"
DATA_FILE = "./data_graph/graph_tool_dataset.jsonl"

def train():
    print(f"Loading Qwen 2.5 from {MODEL_ID}...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right" # SFTTrainer usually prefers right padding for training

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules="all-linear"
    )

    # --- QWEN CHATML FORMATTER (ROBUST VERSION) ---
    def formatting_prompts_func(example):
        output_texts = []
        
        # 1. Normalize inputs: Ensure we always work with lists
        instructions = example['instruction']
        inputs = example['input']
        outputs = example['output']

        if not isinstance(instructions, list):
            instructions = [instructions]
            inputs = [inputs]
            outputs = [outputs]

        # 2. Process the list
        for i in range(len(instructions)):
            system_msg = "You are an AI Agent for Microsoft Graph. Given the user request and the available tool definition, generate the correct JSON tool call."
            user_msg = f"User Request: {instructions[i]}\nAvailable Tool: {inputs[i]}"
            assistant_msg = outputs[i]

            text = (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
            )
            output_texts.append(text)
            
        return output_texts

    # --- CONFIGURATION ---
    # Note: 'max_length' is the correct parameter name for modern TRL versions (replacing max_seq_length)
    sft_config = SFTConfig(
        output_dir="./results_graph",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        logging_steps=10,
        max_length=2048, 
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,             # <--- ADDED: Essential for processing!
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        args=sft_config
    )

    print("Starting Training...")
    trainer.train()
    print("Saving Model...")
    trainer.model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)

if __name__ == "__main__":
    train()