"""
Microsoft Defender XDR API Data Harvester.

This module downloads the Microsoft Security Graph API specification and generates
training data for fine-tuning security-focused tool-calling language models.
Focused on Defender XDR, Sentinel, and security operations endpoints.
"""

import argparse
import json
import os
import random
import re
import uuid
from typing import Any

from defender_api_tool.utils.logging import get_logger

logger = get_logger("data.harvester")

# Default OpenAPI spec URL - Microsoft Graph Security API
DEFAULT_OPENAPI_URL = (
    "https://raw.githubusercontent.com/microsoftgraph/msgraph-metadata"
    "/master/openapi/v1.0/openapi.yaml"
)

# Security-focused prompt templates for generating diverse training samples
PROMPT_TEMPLATES = [
    # Direct security requests
    "I need to {action}.",
    "I want to {action}.",
    "I'd like to {action}.",
    # Security analyst questions
    "Can you {action}?",
    "How do I {action}?",
    "Help me {action}.",
    # Incident response
    "Investigate and {action}.",
    "For this security incident, {action}.",
    "Urgently {action}.",
    # SOC operations
    "As part of threat hunting, {action}.",
    "For compliance purposes, {action}.",
    "During this investigation, {action}.",
    # Imperatives
    "{action}.",
    "{action}",
    # Alert triage
    "Triage this alert by {action}.",
    "Analyze the threat and {action}.",
    "Check if we need to {action}.",
    # Contextual security
    "The SOC team needs to {action}.",
    "For incident response, {action}.",
]

# Microsoft Defender XDR and Security API namespaces
TARGET_NAMESPACES = [
    # Defender XDR - Alerts and Incidents
    "/security/alerts",
    "/security/alerts_v2",
    "/security/incidents",
    # Defender XDR - Threat Intelligence
    "/security/threatIntelligence",
    "/security/tiIndicators",
    # Defender XDR - Hunting
    "/security/runHuntingQuery",
    "/security/cases",
    # Defender for Endpoint
    "/security/attackSimulation",
    # Identity Security
    "/security/identities",
    "/identity",
    "/identityProtection",
    "/riskDetections",
    # Secure Score
    "/security/secureScores",
    "/security/secureScoreControlProfiles",
    # Audit and Compliance
    "/auditLogs",
    "/security/auditLog",
    # User Risk and Sign-in
    "/riskyUsers",
    "/signIns",
    # eDiscovery and Compliance
    "/security/cases/ediscoveryCases",
    # Device Management (Intune Security)
    "/deviceManagement",
    # Application Security
    "/applications",
    "/servicePrincipals",
    # Conditional Access
    "/identity/conditionalAccess",
    # Directory Audit
    "/directoryAudits",
]

# Valid HTTP methods to process
VALID_METHODS = ["get", "post", "patch", "delete", "put"]


class DefenderAPIHarvester:
    """Harvests Microsoft Defender XDR API endpoints and generates training data."""

    def __init__(self, openapi_url: str = DEFAULT_OPENAPI_URL) -> None:
        """
        Initialize the harvester.

        Args:
            openapi_url: URL to the OpenAPI specification YAML file.
        """
        self.openapi_url = openapi_url

    @staticmethod
    def clean_text(text: str | None) -> str:
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
        tool_name = operation.get("operationId", f"{method}_{path.replace('/', '_')}")
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

    def _generate_example_args(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Generate randomized example arguments based on parameter definitions.

        Now uses random values for IDs, integers, and booleans to prevent overfitting.

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
                filters = [
                    "severity eq 'high'",
                    "severity eq 'medium'",
                    "status eq 'new'",
                    "status eq 'inProgress'",
                    "vendorInformation/provider eq 'ASC'",
                ]
                # Combine 1 or 2 random filters
                args[name] = " and ".join(
                    random.sample(filters, k=random.randint(1, 2))
                )
            elif name == "$select":
                fields = [
                    "id",
                    "title",
                    "severity",
                    "status",
                    "assignedTo",
                    "createdDateTime",
                ]
                # Select random subset of fields
                args[name] = ",".join(random.sample(fields, k=random.randint(2, 4)))
            elif name == "$top":
                args[name] = random.choice([10, 25, 50, 100])
            elif name == "$orderby":
                fields = ["createdDateTime", "lastModifiedDateTime", "severity"]
                direction = random.choice(["asc", "desc"])
                args[name] = f"{random.choice(fields)} {direction}"
            elif name == "$expand":
                args[name] = random.choice(["alerts", "evidence", "relatedIncidents"])
            elif name == "$search":
                terms = ["malware", "ransomware", "phishing", "backdoor", "mimikatz"]
                args[name] = f'"{random.choice(terms)}"'

            # Handle security-specific parameters with random values
            elif name == "alertId":
                # Generate a random alert ID format (mix of ID and hash)
                args[name] = f"{uuid.uuid4()}_{random.randint(10000000, 99999999)}"
            elif name == "incidentId":
                args[name] = str(random.randint(1000, 999999))
            elif name == "userId":
                users = ["alice", "bob", "charlie", "admin", "service_account"]
                domain = random.choice(["contoso.com", "fabrikam.com", "woodgrove.com"])
                args[name] = f"{random.choice(users)}@{domain}"
            elif name == "hostName":
                prefixes = ["desktop", "laptop", "server", "workstation"]
                args[name] = f"{random.choice(prefixes)}-{random.randint(1, 99):02d}"
            elif name == "query":
                # KQL Query templates
                kql_queries = [
                    "DeviceProcessEvents | where FileName == 'powershell.exe' | take 100",
                    "SecurityAlert | where Severity == 'High' | sort by TimeGenerated desc",
                    "DeviceLogonEvents | where ActionType == 'LogonFailed' | count",
                    "IdentityLogonEvents | where Application == 'Office 365' | take 50",
                ]
                args[name] = random.choice(kql_queries)
            elif name == "timeRange":
                args[name] = random.choice(["P1D", "P7D", "P30D", "PT1H", "PT24H"])

            # Handle standard types
            elif param_type == "integer":
                args[name] = random.randint(1, 100)
            elif param_type == "boolean":
                args[name] = random.choice([True, False])
            elif param_type == "array":
                args[name] = [f"item{i}" for i in range(random.randint(1, 3))]
            elif param_type == "object":
                args[name] = {}
            else:
                # Default fallback for strings
                args[name] = f"example_{name}_{random.randint(100, 999)}"

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
            raise Exception(f"Failed to download spec: HTTP {response.status_code}")

        result = yaml.safe_load(response.content)
        if not isinstance(result, dict):
            raise ValueError("OpenAPI spec must be a YAML/JSON object")
        return result

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
                summary = self.clean_text(operation.get("summary", ""))
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

                    samples.append(
                        {
                            "instruction": instruction,
                            "input": json.dumps(tool_def),
                            "output": json.dumps(output),
                        }
                    )

        logger.info(f"Generated {len(samples)} training samples")
        return samples

    def harvest(
        self,
        output_dir: str = "./data",
        output_filename: str = "defender_tool_dataset.jsonl",
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


# Backward compatibility alias
GraphAPIHarvester = DefenderAPIHarvester


def main() -> None:
    """CLI entry point for the harvester."""
    parser = argparse.ArgumentParser(
        description="Harvest Microsoft Defender XDR API for LLM training data"
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--output-file",
        default="defender_tool_dataset.jsonl",
        help="Output filename",
    )
    parser.add_argument(
        "--openapi-url",
        default=DEFAULT_OPENAPI_URL,
        help="URL to OpenAPI specification",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    import logging

    from defender_api_tool.utils.logging import setup_logging

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    # Run harvester
    harvester = DefenderAPIHarvester(openapi_url=args.openapi_url)
    output_path = harvester.harvest(
        output_dir=args.output_dir,
        output_filename=args.output_file,
    )
    print(f"Dataset generated: {output_path}")


if __name__ == "__main__":
    main()
