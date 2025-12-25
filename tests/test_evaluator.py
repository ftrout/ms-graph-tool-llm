"""Tests for the evaluation module."""

import json
import pytest

from msgraph_tool_agent_8b.evaluation.evaluator import (
    EvaluationMetrics,
    GraphToolEvaluator,
)


class TestEvaluationMetrics:
    """Test suite for EvaluationMetrics."""

    def test_init_defaults(self):
        """Test that metrics initialize with correct defaults."""
        metrics = EvaluationMetrics()
        assert metrics.total_samples == 0
        assert metrics.valid_json_count == 0
        assert metrics.correct_tool_name_count == 0
        assert metrics.correct_arguments_count == 0
        assert metrics.json_validity_rate == 0.0
        assert len(metrics.errors) == 0

    def test_compute_rates_with_samples(self):
        """Test rate computation with samples."""
        metrics = EvaluationMetrics(
            total_samples=100,
            valid_json_count=90,
            correct_tool_name_count=85,
            correct_arguments_count=80,
            total_inference_time=10.0,
        )

        metrics.compute_rates()

        assert metrics.json_validity_rate == 0.9
        assert metrics.tool_name_accuracy == 0.85
        assert metrics.argument_accuracy == 0.8
        assert metrics.avg_inference_time == 0.1

    def test_compute_rates_zero_samples(self):
        """Test rate computation with zero samples."""
        metrics = EvaluationMetrics(total_samples=0)
        metrics.compute_rates()

        assert metrics.json_validity_rate == 0.0
        assert metrics.tool_name_accuracy == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = EvaluationMetrics(total_samples=100, valid_json_count=90)
        metrics.compute_rates()

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["total_samples"] == 100
        assert result["valid_json_count"] == 90
        assert "json_validity_rate" in result

    def test_str_representation(self):
        """Test string representation."""
        metrics = EvaluationMetrics(total_samples=100, valid_json_count=90)

        result = str(metrics)

        assert "EVALUATION METRICS" in result
        assert "100" in result
        assert "90" in result


class TestGraphToolEvaluator:
    """Test suite for GraphToolEvaluator."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        evaluator = GraphToolEvaluator()
        assert evaluator.base_model_id == "NousResearch/Hermes-3-Llama-3.1-8B"
        assert evaluator.adapter_path is None
        assert evaluator.model is None

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        evaluator = GraphToolEvaluator(
            base_model_id="custom/model", adapter_path="./custom-adapter"
        )
        assert evaluator.base_model_id == "custom/model"
        assert evaluator.adapter_path == "./custom-adapter"

    def test_validate_json_valid(self):
        """Test JSON validation with valid JSON."""
        valid_json = '{"name": "test", "value": 123}'
        is_valid, parsed = GraphToolEvaluator.validate_json(valid_json)

        assert is_valid is True
        assert parsed == {"name": "test", "value": 123}

    def test_validate_json_invalid(self):
        """Test JSON validation with invalid JSON."""
        invalid_json = '{"name": "test", value: 123}'  # Missing quotes
        is_valid, parsed = GraphToolEvaluator.validate_json(invalid_json)

        assert is_valid is False
        assert parsed is None

    def test_validate_json_empty(self):
        """Test JSON validation with empty string."""
        is_valid, parsed = GraphToolEvaluator.validate_json("")
        assert is_valid is False
        assert parsed is None

    def test_compare_tool_calls_matching(self):
        """Test tool call comparison with matching calls."""
        generated = {
            "name": "users_ListUsers",
            "arguments": {"$filter": "test", "$select": "id"},
        }
        expected = {
            "name": "users_ListUsers",
            "arguments": {"$filter": "test", "$select": "id"},
        }

        name_match, args_match = GraphToolEvaluator.compare_tool_calls(
            generated, expected
        )

        assert name_match is True
        assert args_match is True

    def test_compare_tool_calls_different_name(self):
        """Test tool call comparison with different names."""
        generated = {"name": "users_GetUser", "arguments": {}}
        expected = {"name": "users_ListUsers", "arguments": {}}

        name_match, args_match = GraphToolEvaluator.compare_tool_calls(
            generated, expected
        )

        assert name_match is False

    def test_compare_tool_calls_missing_argument(self):
        """Test tool call comparison with missing argument."""
        generated = {"name": "users_ListUsers", "arguments": {"$filter": "test"}}
        expected = {
            "name": "users_ListUsers",
            "arguments": {"$filter": "test", "$select": "id"},
        }

        name_match, args_match = GraphToolEvaluator.compare_tool_calls(
            generated, expected
        )

        assert name_match is True
        assert args_match is False

    def test_compare_tool_calls_type_mismatch(self):
        """Test tool call comparison with type mismatch."""
        generated = {
            "name": "users_ListUsers",
            "arguments": {"items": "string"},  # Should be array
        }
        expected = {"name": "users_ListUsers", "arguments": {"items": ["a", "b"]}}

        name_match, args_match = GraphToolEvaluator.compare_tool_calls(
            generated, expected
        )

        assert name_match is True
        assert args_match is False


class TestJSONValidation:
    """Additional tests for JSON validation edge cases."""

    @pytest.mark.parametrize(
        "json_str,expected_valid",
        [
            ('{"name": "test"}', True),
            ('{"name": "test", "nested": {"key": "value"}}', True),
            ('{"array": [1, 2, 3]}', True),
            ("null", True),
            ("true", True),
            ("123", True),
            ('"string"', True),
            ('{"incomplete": ', False),
            ("not json at all", False),
            ('{"trailing": "comma",}', False),
        ],
    )
    def test_various_json_inputs(self, json_str, expected_valid):
        """Test JSON validation with various inputs."""
        is_valid, _ = GraphToolEvaluator.validate_json(json_str)
        assert is_valid == expected_valid
