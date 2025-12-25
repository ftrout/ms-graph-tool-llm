"""Configuration classes for msgraph-tool-agent-8b."""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for the base model and LoRA adapter."""

    # Base model settings
    base_model_id: str = "NousResearch/Hermes-3-Llama-3.1-8B"
    model_type: str = "causal_lm"

    # Quantization settings
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    # LoRA settings
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Model behavior
    max_seq_length: int = 2048
    attn_implementation: str = "flash_attention_2"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "base_model_id": self.base_model_id,
            "model_type": self.model_type,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "max_seq_length": self.max_seq_length,
            "attn_implementation": self.attn_implementation,
        }

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class TrainingConfig:
    """Configuration for training the model."""

    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03

    # Precision and optimization
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    # Evaluation strategy
    eval_strategy: str = "steps"  # "steps", "epoch", or "no"
    eval_steps: int = 100

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # Best model selection
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3

    # Experiment tracking: "none", "wandb", "tensorboard", or "all"
    report_to: str = "none"
    run_name: str | None = None

    # Dataset
    test_size: float = 0.1
    seed: int = 42

    # Output
    output_dir: str = "./results"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_threshold": self.early_stopping_threshold,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "report_to": self.report_to,
            "run_name": self.run_name,
            "test_size": self.test_size,
            "seed": self.seed,
            "output_dir": self.output_dir,
        }


# Default configuration
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
