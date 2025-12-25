"""Tests for configuration classes."""

import json
import os
import tempfile

import pytest

from msgraph_tool_llm.utils.config import (
    ModelConfig,
    TrainingConfig,
    DEFAULT_MODEL_CONFIG,
)


class TestModelConfig:
    """Test suite for ModelConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ModelConfig()

        assert config.base_model_id == "NousResearch/Hermes-3-Llama-3.1-8B"
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.lora_r == 32
        assert config.lora_alpha == 64
        assert config.max_seq_length == 2048

    def test_custom_values(self):
        """Test initialization with custom values."""
        config = ModelConfig(
            base_model_id="custom/model",
            lora_r=64,
            lora_alpha=128,
            max_seq_length=4096
        )

        assert config.base_model_id == "custom/model"
        assert config.lora_r == 64
        assert config.lora_alpha == 128
        assert config.max_seq_length == 4096

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ModelConfig()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["base_model_id"] == "NousResearch/Hermes-3-Llama-3.1-8B"
        assert result["lora_r"] == 32
        assert "lora_target_modules" in result

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "base_model_id": "test/model",
            "model_type": "causal_lm",
            "load_in_4bit": False,
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_compute_dtype": "float16",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "lora_target_modules": ["q_proj", "v_proj"],
            "max_seq_length": 1024,
            "attn_implementation": "sdpa"
        }

        config = ModelConfig.from_dict(config_dict)

        assert config.base_model_id == "test/model"
        assert config.lora_r == 16
        assert config.load_in_4bit is False

    def test_save_and_load(self):
        """Test saving and loading configuration."""
        config = ModelConfig(lora_r=48, lora_alpha=96)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            config.save(f.name)
            loaded_config = ModelConfig.load(f.name)

            assert loaded_config.lora_r == 48
            assert loaded_config.lora_alpha == 96

            os.unlink(f.name)

    def test_target_modules_default(self):
        """Test that target modules have correct defaults."""
        config = ModelConfig()

        expected_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        assert config.lora_target_modules == expected_modules


class TestTrainingConfig:
    """Test suite for TrainingConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = TrainingConfig()

        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 4
        assert config.learning_rate == 1e-4
        assert config.bf16 is True
        assert config.gradient_checkpointing is True

    def test_custom_values(self):
        """Test initialization with custom values."""
        config = TrainingConfig(
            num_train_epochs=5,
            per_device_train_batch_size=8,
            learning_rate=2e-4
        )

        assert config.num_train_epochs == 5
        assert config.per_device_train_batch_size == 8
        assert config.learning_rate == 2e-4

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingConfig()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["num_train_epochs"] == 3
        assert result["learning_rate"] == 1e-4
        assert "gradient_checkpointing" in result

    def test_effective_batch_size(self):
        """Test that effective batch size can be computed."""
        config = TrainingConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4
        )

        effective_batch_size = (
            config.per_device_train_batch_size *
            config.gradient_accumulation_steps
        )
        assert effective_batch_size == 16


class TestDefaultConfig:
    """Test suite for default configuration."""

    def test_default_config_exists(self):
        """Test that default config is available."""
        assert DEFAULT_MODEL_CONFIG is not None
        assert isinstance(DEFAULT_MODEL_CONFIG, ModelConfig)

    def test_default_config_immutability(self):
        """Test that modifying a config doesn't affect the default."""
        config = ModelConfig()
        config.lora_r = 128

        # Default should still have original value
        assert DEFAULT_MODEL_CONFIG.lora_r == 32
