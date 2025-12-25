"""
Microsoft Graph API Data Harvester.

This module downloads the Microsoft Graph OpenAPI specification and generates
training data for fine-tuning tool-calling language models.
"""

import argparse
import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default OpenAPI spec URL
DEFAULT_OPENAPI_URL = (
    "https://raw.githubusercontent.com/microsoftgraph/msgraph-metadata"
    "/master/openapi/v1.0/openapi.yaml"
)

# Prompt templates for generating diverse training samples
PROMPT_TEMPLATES = [
    "I need to {action}.",
    "Please help me {action}.",
    "Can you {action}?",
    "How do I {action}?",
    "{action}",
]

# Target API namespaces to include
TARGET_NAMESPACES = [
    "/me/messages",
    "/me/events",
    "/me/drive",
    "/teams",
    "/users",
    "/sites",
    "/groups",
    "/applications",
]

# Valid HTTP methods to process
VALID_METHODS = ["get", "post", "patch", "delete", "put"]


class GraphAPIHarvester:
    """Harvests Microsoft Graph API endpoints and generates training data."""

    def __init__(self, openapi_url: str = DEFAULT_OPENAPI_URL) -> None:
        """
        Initialize the harvester.

        Args:
            openapi_url: URL to the OpenAPI specification YAML file.
        """
        self.openapi_url = openapi_url

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """
        Clean text by removing HTML tags and normalizing whitespace.

        Args:
            text: Input text that may contain HTML tags.

        Returns:
            Cleaned text with HTML removed and whitespace normalized.
        """
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r"<[^<]+?>", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _format_tool(
        self, path: str, method: str, operation: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Format an OpenAPI operation as a tool definition.

        Args:
            path: API endpoint path.
            method: HTTP method.
            operation: OpenAPI operation object.

        Returns:
            Tool definition in function-calling format.
        """
        tool_name = operation.get(
            "operationId", f"{method}_{path.replace('/', '_')}"
        )
        description = self.clean_text(
            operation.get("summary", operation.get("description", ""))
        )

        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        # Process parameters
        for param in operation.get("parameters", []):
            if param.get("in") in ["path", "query"]:
                param_name = param["name"]
                param_desc = self.clean_text(param.get("description", ""))
                param_type = param.get("schema", {}).get("type", "string")

                parameters["properties"][param_name] = {
                    "type": param_type,
                    "description": param_desc,
                }

                if param.get("required", False):
                    parameters["required"].append(param_name)

        # Process request body
        if "requestBody" in operation:
            content = operation["requestBody"].get("content", {})
            if "application/json" in content:
                parameters["properties"]["request_body"] = {
                    "type": "object",
                    "description": "JSON body required for this action",
                }

        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": parameters,
            },
        }

    def _generate_example_args(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate example arguments based on parameter definitions.

        Args:
            params: Parameter schema with properties.

        Returns:
            Dictionary of example argument values.
        """
        args: dict[str, Any] = {}
        properties = params.get("properties", {})

        for name, prop in properties.items():
            param_type = prop.get("type", "string")

            # Handle OData parameters specially
            if name == "$filter":
                args[name] = "startswith(displayName, 'A')"
            elif name == "$select":
                args[name] = "id,displayName,mail"
            elif name == "$top":
                args[name] = 10
            elif name == "$orderby":
                args[name] = "displayName asc"
            elif name == "$expand":
                args[name] = "members"
            elif name == "$search":
                args[name] = '"displayName:John"'
            # Handle by type
            elif param_type == "integer":
                args[name] = 10
            elif param_type == "boolean":
                args[name] = True
            elif param_type == "array":
                args[name] = ["item1", "item2"]
            elif param_type == "object":
                args[name] = {}
            else:
                # Default to string with example value
                args[name] = f"example_{name}"

        return args

    def download_spec(self) -> dict[str, Any]:
        """
        Download the OpenAPI specification.

        Returns:
            Parsed OpenAPI specification as a dictionary.

        Raises:
            Exception: If download fails.
        """
        import requests
        import yaml

        logger.info(f"Downloading OpenAPI spec from {self.openapi_url}")
        response = requests.get(self.openapi_url, timeout=120)

        if response.status_code != 200:
            raise Exception(
                f"Failed to download spec: HTTP {response.status_code}"
            )

        return yaml.safe_load(response.content)

    def process_spec(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Process the OpenAPI spec and generate training samples.

        Args:
            spec: Parsed OpenAPI specification.

        Returns:
            List of training samples with instruction, input, and output.
        """
        samples: list[dict[str, Any]] = []
        paths = spec.get("paths", {})

        logger.info(f"Processing {len(paths)} endpoints")

        for path, methods in paths.items():
            # Filter to target namespaces if specified
            if TARGET_NAMESPACES:
                if not any(ns in path for ns in TARGET_NAMESPACES):
                    continue

            for method, operation in methods.items():
                # Skip non-HTTP method keys
                if method not in VALID_METHODS:
                    continue

                # Skip if no summary
                summary = self.clean_text(
                    operation.get("summary", "")
                )
                if not summary:
                    continue

                # Format tool definition
                tool_def = self._format_tool(path, method, operation)

                # Generate example arguments
                example_args = self._generate_example_args(
                    tool_def["function"]["parameters"]
                )

                # Create training samples with different prompt templates
                for template in PROMPT_TEMPLATES:
                    instruction = template.format(action=summary.lower())
                    output = {
                        "name": tool_def["function"]["name"],
                        "arguments": example_args,
                    }

                    samples.append({
                        "instruction": instruction,
                        "input": json.dumps(tool_def),
                        "output": json.dumps(output),
                    })

        logger.info(f"Generated {len(samples)} training samples")
        return samples

    def harvest(
        self,
        output_dir: str = "./data_graph",
        output_filename: str = "graph_tool_dataset.jsonl",
    ) -> str:
        """
        Download spec and generate training dataset.

        Args:
            output_dir: Directory to write output file.
            output_filename: Name of the output JSONL file.

        Returns:
            Path to the generated dataset file.
        """
        # Download specification
        spec = self.download_spec()

        # Process and generate samples
        samples = self.process_spec(spec)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        # Write samples to JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        logger.info(f"Wrote {len(samples)} samples to {output_path}")
        return output_path


def main() -> None:
    """CLI entry point for the harvester."""
    parser = argparse.ArgumentParser(
        description="Harvest Microsoft Graph API for LLM training data"
    )
    parser.add_argument(
        "--output-dir",
        default="./data_graph",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--output-file",
        default="graph_tool_dataset.jsonl",
        help="Output filename",
    )
    parser.add_argument(
        "--openapi-url",
        default=DEFAULT_OPENAPI_URL,
        help="URL to OpenAPI specification",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run harvester
    harvester = GraphAPIHarvester(openapi_url=args.openapi_url)
    output_path = harvester.harvest(
        output_dir=args.output_dir,
        output_filename=args.output_file,
    )
    print(f"Dataset generated: {output_path}")


if __name__ == "__main__":
    main()
