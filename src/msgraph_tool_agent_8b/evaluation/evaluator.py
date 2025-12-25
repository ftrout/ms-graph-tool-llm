"""
Comprehensive evaluation framework for msgraph-tool-agent-8b.

Provides metrics for evaluating tool-calling accuracy, JSON validity,
and schema compliance.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from msgraph_tool_agent_8b.utils.logging import get_logger

logger = get_logger("evaluation.evaluator")

# System prompt for evaluation
SYSTEM_PROMPT = (
    "You are an AI Agent for Microsoft Graph. Given the user request and "
    "the available tool definition, generate the correct JSON tool call."
)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    # Core metrics
    total_samples: int = 0
    valid_json_count: int = 0
    correct_tool_name_count: int = 0
    correct_arguments_count: int = 0

    # Derived metrics (computed)
    json_validity_rate: float = 0.0
    tool_name_accuracy: float = 0.0
    argument_accuracy: float = 0.0
    overall_accuracy: float = 0.0

    # Timing
    total_inference_time: float = 0.0
    avg_inference_time: float = 0.0
    tokens_per_second: float = 0.0

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def compute_rates(self) -> None:
        """Compute derived metric rates."""
        if self.total_samples > 0:
            self.json_validity_rate = self.valid_json_count / self.total_samples
            self.tool_name_accuracy = self.correct_tool_name_count / self.total_samples
            self.argument_accuracy = self.correct_arguments_count / self.total_samples
            # Overall = valid JSON + correct name + correct args
            self.overall_accuracy = (
                self.correct_arguments_count / self.total_samples
            )
            if self.total_inference_time > 0:
                self.avg_inference_time = self.total_inference_time / self.total_samples

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_samples": self.total_samples,
            "valid_json_count": self.valid_json_count,
            "correct_tool_name_count": self.correct_tool_name_count,
            "correct_arguments_count": self.correct_arguments_count,
            "json_validity_rate": round(self.json_validity_rate, 4),
            "tool_name_accuracy": round(self.tool_name_accuracy, 4),
            "argument_accuracy": round(self.argument_accuracy, 4),
            "overall_accuracy": round(self.overall_accuracy, 4),
            "total_inference_time_seconds": round(self.total_inference_time, 2),
            "avg_inference_time_seconds": round(self.avg_inference_time, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "error_count": len(self.errors),
        }

    def __str__(self) -> str:
        """Human-readable metrics summary."""
        self.compute_rates()
        return (
            f"\n{'='*60}\n"
            f"EVALUATION METRICS\n"
            f"{'='*60}\n"
            f"Total Samples:        {self.total_samples}\n"
            f"{'─'*60}\n"
            f"JSON Validity Rate:   {self.json_validity_rate:.2%} ({self.valid_json_count}/{self.total_samples})\n"
            f"Tool Name Accuracy:   {self.tool_name_accuracy:.2%} ({self.correct_tool_name_count}/{self.total_samples})\n"
            f"Argument Accuracy:    {self.argument_accuracy:.2%} ({self.correct_arguments_count}/{self.total_samples})\n"
            f"{'─'*60}\n"
            f"Overall Accuracy:     {self.overall_accuracy:.2%}\n"
            f"{'─'*60}\n"
            f"Avg Inference Time:   {self.avg_inference_time:.3f}s\n"
            f"Tokens/Second:        {self.tokens_per_second:.1f}\n"
            f"Total Errors:         {len(self.errors)}\n"
            f"{'='*60}\n"
        )


class GraphToolEvaluator:
    """
    Comprehensive evaluator for Microsoft Graph tool-calling models.

    Evaluates models on JSON validity, tool name accuracy, and
    argument correctness.

    Attributes:
        base_model_id: Base model identifier
        adapter_path: Path to the trained LoRA adapter
        device: Inference device (cuda/cpu)

    Example:
        >>> evaluator = GraphToolEvaluator(
        ...     adapter_path="./msgraph-tool-agent-8b"
        ... )
        >>> metrics = evaluator.evaluate_dataset("./data/test.jsonl")
        >>> print(metrics)
    """

    def __init__(
        self,
        base_model_id: str = "NousResearch/Hermes-3-Llama-3.1-8B",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the evaluator.

        Args:
            base_model_id: Base model identifier
            adapter_path: Path to LoRA adapter (None for base model only)
            device: Device for inference (auto-detected if None)
        """
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        if device:
            self.device = device
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        self.model = None
        self.tokenizer = None
        logger.info(
            "Initialized evaluator with base model: %s, adapter: %s",
            base_model_id, adapter_path
        )

    def load_model(self) -> None:
        """Load the model and adapter."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model...")

        # Load base model in bfloat16 (not quantized) to allow adapter merging
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Load and merge adapter if provided
        if self.adapter_path and os.path.exists(self.adapter_path):
            logger.info("Loading adapter from %s...", self.adapter_path)
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            logger.info("Merging adapter weights...")
            self.model = self.model.merge_and_unload()
            logger.info("Adapter merged successfully")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            trust_remote_code=True
        )

        self.model.eval()
        logger.info("Model loaded successfully on %s", self.device)

    def generate(
        self,
        instruction: str,
        tool_definition: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1
    ) -> Tuple[str, float, int]:
        """
        Generate a tool call for the given instruction.

        Args:
            instruction: User instruction/request
            tool_definition: JSON tool definition string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (generated_text, inference_time, num_tokens)
        """
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"User Request: {instruction}\nAvailable Tool: {tool_definition}"
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        inference_time = time.time() - start_time

        # Decode only new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        num_tokens = len(new_tokens)
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip(), inference_time, num_tokens

    @staticmethod
    def validate_json(text: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate that text is valid JSON.

        Args:
            text: Text to validate

        Returns:
            Tuple of (is_valid, parsed_json)
        """
        try:
            parsed = json.loads(text)
            return True, parsed
        except json.JSONDecodeError:
            return False, None

    @staticmethod
    def compare_tool_calls(
        generated: Dict[str, Any],
        expected: Dict[str, Any]
    ) -> Tuple[bool, bool]:
        """
        Compare generated tool call with expected.

        Args:
            generated: Generated tool call dict
            expected: Expected tool call dict

        Returns:
            Tuple of (name_matches, arguments_match)
        """
        # Check tool name
        gen_name = generated.get("name", "")
        exp_name = expected.get("name", "")
        name_matches = gen_name == exp_name

        # Check arguments (flexible comparison)
        gen_args = generated.get("arguments", {})
        exp_args = expected.get("arguments", {})

        # Arguments match if all expected keys are present
        # and have compatible values
        args_match = True
        for key in exp_args:
            if key not in gen_args:
                args_match = False
                break
            # Type-aware comparison
            if isinstance(exp_args[key], (list, dict)):
                if type(gen_args[key]) != type(exp_args[key]):
                    args_match = False
                    break

        return name_matches, args_match

    def evaluate_sample(
        self,
        instruction: str,
        tool_definition: str,
        expected_output: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample.

        Args:
            instruction: User instruction
            tool_definition: Tool definition JSON string
            expected_output: Expected output JSON string

        Returns:
            Dictionary with evaluation results
        """
        result = {
            "instruction": instruction,
            "expected": expected_output,
            "generated": None,
            "is_valid_json": False,
            "name_correct": False,
            "arguments_correct": False,
            "inference_time": 0.0,
            "num_tokens": 0,
            "error": None,
        }

        try:
            # Generate response
            generated, inference_time, num_tokens = self.generate(
                instruction, tool_definition
            )
            result["generated"] = generated
            result["inference_time"] = inference_time
            result["num_tokens"] = num_tokens

            # Validate JSON
            is_valid, parsed_gen = self.validate_json(generated)
            result["is_valid_json"] = is_valid

            if is_valid:
                _, parsed_exp = self.validate_json(expected_output)
                if parsed_exp:
                    name_match, args_match = self.compare_tool_calls(
                        parsed_gen, parsed_exp
                    )
                    result["name_correct"] = name_match
                    result["arguments_correct"] = name_match and args_match

        except Exception as e:
            result["error"] = str(e)
            logger.warning("Error evaluating sample: %s", e)

        return result

    def evaluate_dataset(
        self,
        data_file: str,
        max_samples: Optional[int] = None,
        save_results: Optional[str] = None
    ) -> EvaluationMetrics:
        """
        Evaluate the model on a dataset.

        Args:
            data_file: Path to JSONL dataset file
            max_samples: Maximum samples to evaluate (None for all)
            save_results: Path to save detailed results (optional)

        Returns:
            EvaluationMetrics with computed metrics
        """
        from datasets import load_dataset

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Load dataset
        logger.info("Loading evaluation dataset from %s...", data_file)
        dataset = load_dataset("json", data_files=data_file, split="train")

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        logger.info("Evaluating %d samples...", len(dataset))

        metrics = EvaluationMetrics()
        metrics.total_samples = len(dataset)
        results = []
        total_tokens = 0

        for i, sample in enumerate(dataset):
            if (i + 1) % 10 == 0:
                logger.info("Progress: %d/%d", i + 1, len(dataset))

            result = self.evaluate_sample(
                instruction=sample["instruction"],
                tool_definition=sample["input"],
                expected_output=sample["output"]
            )
            results.append(result)

            # Update counts
            if result["is_valid_json"]:
                metrics.valid_json_count += 1
            if result["name_correct"]:
                metrics.correct_tool_name_count += 1
            if result["arguments_correct"]:
                metrics.correct_arguments_count += 1
            if result["error"]:
                metrics.errors.append({
                    "index": i,
                    "instruction": sample["instruction"],
                    "error": result["error"]
                })

            metrics.total_inference_time += result["inference_time"]
            total_tokens += result["num_tokens"]

        # Compute derived metrics
        metrics.compute_rates()
        if metrics.total_inference_time > 0:
            metrics.tokens_per_second = total_tokens / metrics.total_inference_time

        # Save detailed results
        if save_results:
            with open(save_results, "w") as f:
                json.dump({
                    "metrics": metrics.to_dict(),
                    "results": results
                }, f, indent=2)
            logger.info("Detailed results saved to %s", save_results)

        logger.info(str(metrics))
        return metrics

    def evaluate_single(self, query: str, tool: Dict[str, Any]) -> None:
        """
        Evaluate a single query interactively.

        Args:
            query: Natural language query
            tool: Tool definition dictionary
        """
        if self.model is None:
            self.load_model()

        tool_json = json.dumps(tool)
        response, inference_time, num_tokens = self.generate(query, tool_json)

        print(f"\nQuery: {query}")
        print(f"Tool: {tool['function']['name']}")
        print(f"\nGenerated Response:")

        is_valid, parsed = self.validate_json(response)
        if is_valid:
            print(json.dumps(parsed, indent=2))
            print(f"\n[Valid JSON] Inference time: {inference_time:.3f}s")
        else:
            print(response)
            print(f"\n[Invalid JSON] Inference time: {inference_time:.3f}s")


# Default test tool for quick evaluation
DEFAULT_TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "users_ListUsers",
        "description": "Retrieve a list of user objects.",
        "parameters": {
            "type": "object",
            "properties": {
                "$filter": {"type": "string", "description": "Filter items"},
                "$select": {"type": "string", "description": "Select properties"}
            }
        }
    }
}


def main():
    """CLI entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate msgraph-tool-agent-8b model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="NousResearch/Hermes-3-Llama-3.1-8B",
        help="Base model identifier"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="msgraph-tool-agent-8b",
        help="Path to trained LoRA adapter"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to evaluation dataset (JSONL)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Find the user with email 'admin@contoso.com' and select their id.",
        help="Single query to evaluate"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save detailed results"
    )

    args = parser.parse_args()

    from msgraph_tool_agent_8b.utils.logging import setup_logging
    setup_logging()

    evaluator = GraphToolEvaluator(
        base_model_id=args.base_model,
        adapter_path=args.adapter_path
    )

    if args.data_file:
        # Evaluate on dataset
        evaluator.evaluate_dataset(
            data_file=args.data_file,
            max_samples=args.max_samples,
            save_results=args.save_results
        )
    else:
        # Single query evaluation
        evaluator.evaluate_single(args.query, DEFAULT_TEST_TOOL)


if __name__ == "__main__":
    main()
