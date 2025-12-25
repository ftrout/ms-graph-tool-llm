"""
Training pipeline for msgraph-tool-agent-8b.

Fine-tunes LLMs using QLoRA for Microsoft Graph API tool calling.
"""

import os
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import SFTConfig, SFTTrainer

from msgraph_tool_agent_8b.utils.config import ModelConfig, TrainingConfig
from msgraph_tool_agent_8b.utils.logging import get_logger

logger = get_logger("training.trainer")

# System prompt for the agent
SYSTEM_PROMPT = (
    "You are an AI Agent for Microsoft Graph. Given the user request and "
    "the available tool definition, generate the correct JSON tool call."
)


class GraphToolTrainer:
    """
    Trainer for fine-tuning LLMs on Microsoft Graph tool calling.

    Uses QLoRA (Quantized LoRA) for efficient fine-tuning on consumer hardware.

    Attributes:
        model_config: Configuration for the base model and LoRA
        training_config: Configuration for training parameters

    Example:
        >>> trainer = GraphToolTrainer()
        >>> trainer.train(
        ...     data_file="./data/graph_tool_dataset.jsonl",
        ...     output_name="msgraph-tool-agent-7b"
        ... )
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize the trainer.

        Args:
            model_config: Model configuration (uses defaults if None)
            training_config: Training configuration (uses defaults if None)
        """
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        logger.info(
            "Initialized GraphToolTrainer with base model: %s",
            self.model_config.base_model_id
        )

    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """Create quantization configuration for 4-bit loading."""
        compute_dtype = (
            torch.bfloat16
            if self.model_config.bnb_4bit_compute_dtype == "bfloat16"
            else torch.float16
        )
        return BitsAndBytesConfig(
            load_in_4bit=self.model_config.load_in_4bit,
            bnb_4bit_quant_type=self.model_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.model_config.bnb_4bit_use_double_quant,
        )

    def _get_lora_config(self) -> LoraConfig:
        """Create LoRA configuration."""
        return LoraConfig(
            r=self.model_config.lora_r,
            lora_alpha=self.model_config.lora_alpha,
            lora_dropout=self.model_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.model_config.lora_target_modules,
        )

    def _load_model(self) -> PreTrainedModel:
        """
        Load the base model with quantization.

        Returns:
            Loaded and prepared model
        """
        bnb_config = self._get_quantization_config()

        # Try Flash Attention 2, fallback to SDPA
        try:
            logger.info("Loading model with Flash Attention 2...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            logger.info("Successfully loaded with Flash Attention 2")
        except Exception as e:
            logger.warning("Flash Attention 2 unavailable, using SDPA: %s", e)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
            logger.info("Successfully loaded with SDPA attention")

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and configure the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model_id,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def _format_chat_template(
        self,
        examples: Dict[str, List[str]]
    ) -> List[str]:
        """
        Format examples using the model's chat template.

        Args:
            examples: Batch of examples with instruction/input/output keys

        Returns:
            List of formatted chat strings
        """
        texts = []
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        for i in range(len(instructions)):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"User Request: {instructions[i]}\nAvailable Tool: {inputs[i]}"
                },
                {"role": "assistant", "content": outputs[i]}
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False
            )
            texts.append(text)

        return texts

    def _load_dataset(self, data_file: str) -> DatasetDict:
        """
        Load and split the training dataset.

        Args:
            data_file: Path to the JSONL dataset file

        Returns:
            DatasetDict with train and test splits

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset is empty
        """
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Dataset not found at {data_file}. "
                "Run data harvesting first to generate training data."
            )

        logger.info("Loading dataset from %s...", data_file)
        dataset = load_dataset("json", data_files=data_file, split="train")

        if len(dataset) == 0:
            raise ValueError(f"Dataset is empty: {data_file}")

        logger.info("Loaded %d training examples", len(dataset))

        # Split into train/test
        dataset = dataset.train_test_split(
            test_size=self.training_config.test_size,
            seed=self.training_config.seed
        )
        logger.info(
            "Split: %d train, %d test",
            len(dataset["train"]),
            len(dataset["test"])
        )

        return dataset

    def train(
        self,
        data_file: str = "./data/graph_tool_dataset.jsonl",
        output_name: str = "msgraph-tool-agent-7b",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None
    ) -> str:
        """
        Train the model on the Microsoft Graph tool calling dataset.

        Args:
            data_file: Path to the training dataset
            output_name: Name for the output adapter directory
            push_to_hub: Whether to push to Hugging Face Hub
            hub_model_id: Model ID for Hub upload (e.g., "username/model-name")

        Returns:
            Path to the saved adapter

        Raises:
            FileNotFoundError: If dataset doesn't exist
            ValueError: If dataset is empty
        """
        logger.info("Starting training pipeline...")

        # Load components
        dataset = self._load_dataset(data_file)
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

        # Configure LoRA
        peft_config = self._get_lora_config()

        # Configure training
        sft_config = SFTConfig(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            # Evaluation strategy
            eval_strategy=self.training_config.eval_strategy,
            eval_steps=self.training_config.eval_steps,
            # Best model selection
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            # Logging and saving
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            # Experiment tracking
            report_to=self.training_config.report_to,
            run_name=self.training_config.run_name,
            # Model config
            max_seq_length=self.model_config.max_seq_length,
            packing=False,
            dataset_text_field="text",
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            seed=self.training_config.seed,
        )

        # Setup callbacks
        callbacks = []
        if self.training_config.early_stopping:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience,
                    early_stopping_threshold=self.training_config.early_stopping_threshold,
                )
            )
            logger.info(
                "Early stopping enabled: patience=%d, threshold=%.4f",
                self.training_config.early_stopping_patience,
                self.training_config.early_stopping_threshold,
            )

        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            peft_config=peft_config,
            formatting_func=self._format_chat_template,
            args=sft_config,
            callbacks=callbacks if callbacks else None,
        )

        # Log training config
        effective_batch_size = (
            self.training_config.per_device_train_batch_size *
            self.training_config.gradient_accumulation_steps
        )
        logger.info("Training configuration:")
        logger.info("  Epochs: %d", self.training_config.num_train_epochs)
        logger.info("  Batch size: %d", self.training_config.per_device_train_batch_size)
        logger.info("  Gradient accumulation: %d", self.training_config.gradient_accumulation_steps)
        logger.info("  Effective batch size: %d", effective_batch_size)
        logger.info("  Learning rate: %.2e", self.training_config.learning_rate)
        logger.info("  LoRA rank: %d", self.model_config.lora_r)
        logger.info("  Eval strategy: %s", self.training_config.eval_strategy)
        logger.info("  Load best model: %s", self.training_config.load_best_model_at_end)
        logger.info("  Report to: %s", self.training_config.report_to)

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save
        logger.info("Saving model to %s...", output_name)
        trainer.model.save_pretrained(output_name)
        self.tokenizer.save_pretrained(output_name)

        # Save model config
        self.model_config.save(os.path.join(output_name, "msgraph_config.json"))

        logger.info("Training complete! Model saved to %s", output_name)
        return output_name


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune LLM for Microsoft Graph tool calling"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="NousResearch/Hermes-3-Llama-3.1-8B",
        help="Base model identifier"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="msgraph-tool-agent-8b",
        help="Output adapter directory name"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="./data/graph_tool_dataset.jsonl",
        help="Path to training dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA adapter rank"
    )
    # Early stopping options
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience (number of eval steps)"
    )
    # Experiment tracking
    parser.add_argument(
        "--report-to",
        type=str,
        choices=["none", "wandb", "tensorboard", "all"],
        default="none",
        help="Experiment tracking: none, wandb, tensorboard, or all"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the training run (for W&B/TensorBoard)"
    )
    # Hub options
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Model ID for Hub upload"
    )

    args = parser.parse_args()

    from msgraph_tool_agent_8b.utils.logging import setup_logging
    setup_logging()

    # Create configs
    model_config = ModelConfig(
        base_model_id=args.model_id,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
    )
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        report_to=args.report_to,
        run_name=args.run_name,
    )

    trainer = GraphToolTrainer(
        model_config=model_config,
        training_config=training_config
    )
    trainer.train(
        data_file=args.data_file,
        output_name=args.output_name,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


if __name__ == "__main__":
    main()
