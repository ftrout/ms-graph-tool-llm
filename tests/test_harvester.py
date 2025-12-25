"""Tests for the data harvester module."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from msgraph_tool_agent_8b.data.harvester import GraphAPIHarvester, PROMPT_TEMPLATES


class TestGraphAPIHarvester:
    """Test suite for GraphAPIHarvester."""

    def test_init_default_url(self):
        """Test initialization with default URL."""
        harvester = GraphAPIHarvester()
        assert "githubusercontent.com" in harvester.openapi_url
        assert "msgraph-metadata" in harvester.openapi_url

    def test_init_custom_url(self):
        """Test initialization with custom URL."""
        custom_url = "https://example.com/openapi.yaml"
        harvester = GraphAPIHarvester(openapi_url=custom_url)
        assert harvester.openapi_url == custom_url

    def test_clean_text_removes_html(self):
        """Test that HTML tags are removed from text."""
        text = "<p>Hello <b>World</b></p>"
        result = GraphAPIHarvester.clean_text(text)
        assert result == "Hello World"

    def test_clean_text_normalizes_whitespace(self):
        """Test that whitespace is normalized."""
        text = "Hello    World\n\nTest"
        result = GraphAPIHarvester.clean_text(text)
        assert result == "Hello World Test"

    def test_clean_text_handles_none(self):
        """Test that None input returns empty string."""
        result = GraphAPIHarvester.clean_text(None)
        assert result == ""

    def test_clean_text_handles_empty(self):
        """Test that empty string returns empty string."""
        result = GraphAPIHarvester.clean_text("")
        assert result == ""

    def test_format_tool_basic(self):
        """Test basic tool formatting."""
        harvester = GraphAPIHarvester()
        operation = {
            "operationId": "users_ListUsers",
            "summary": "List all users",
            "parameters": [
                {
                    "name": "$filter",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter results",
                }
            ],
        }

        tool = harvester._format_tool("/users", "get", operation)

        assert tool["type"] == "function"
        assert tool["function"]["name"] == "users_ListUsers"
        assert tool["function"]["description"] == "List all users"
        assert "$filter" in tool["function"]["parameters"]["properties"]

    def test_format_tool_with_path_params(self):
        """Test tool formatting with path parameters."""
        harvester = GraphAPIHarvester()
        operation = {
            "operationId": "users_GetUser",
            "summary": "Get user by ID",
            "parameters": [
                {
                    "name": "user-id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "User ID",
                }
            ],
        }

        tool = harvester._format_tool("/users/{user-id}", "get", operation)

        assert "user-id" in tool["function"]["parameters"]["properties"]
        assert "user-id" in tool["function"]["parameters"]["required"]

    def test_generate_example_args_string(self):
        """Test example argument generation for string types."""
        harvester = GraphAPIHarvester()
        params = {
            "properties": {"name": {"type": "string", "description": "User name"}}
        }

        args = harvester._generate_example_args(params)
        assert args["name"] == "example_name"

    def test_generate_example_args_integer(self):
        """Test example argument generation for integer types."""
        harvester = GraphAPIHarvester()
        params = {"properties": {"count": {"type": "integer", "description": "Count"}}}

        args = harvester._generate_example_args(params)
        assert args["count"] == 10

    def test_generate_example_args_boolean(self):
        """Test example argument generation for boolean types."""
        harvester = GraphAPIHarvester()
        params = {
            "properties": {"enabled": {"type": "boolean", "description": "Is enabled"}}
        }

        args = harvester._generate_example_args(params)
        assert args["enabled"] is True

    def test_generate_example_args_odata_filter(self):
        """Test example argument generation for OData $filter."""
        harvester = GraphAPIHarvester()
        params = {
            "properties": {"$filter": {"type": "string", "description": "Filter"}}
        }

        args = harvester._generate_example_args(params)
        assert "startswith" in args["$filter"]

    def test_generate_example_args_odata_select(self):
        """Test example argument generation for OData $select."""
        harvester = GraphAPIHarvester()
        params = {
            "properties": {"$select": {"type": "string", "description": "Select"}}
        }

        args = harvester._generate_example_args(params)
        assert "id" in args["$select"]
        assert "displayName" in args["$select"]

    def test_process_spec_generates_samples(self):
        """Test that process_spec generates training samples."""
        harvester = GraphAPIHarvester()
        spec = {
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "users_ListUsers",
                        "summary": "List all users",
                        "parameters": [],
                    }
                }
            }
        }

        samples = harvester.process_spec(spec)

        # Should generate one sample per template
        assert len(samples) == len(PROMPT_TEMPLATES)
        assert all("instruction" in s for s in samples)
        assert all("input" in s for s in samples)
        assert all("output" in s for s in samples)

    def test_process_spec_skips_invalid_methods(self):
        """Test that non-HTTP methods are skipped."""
        harvester = GraphAPIHarvester()
        spec = {
            "paths": {
                "/users": {
                    "parameters": [],  # Not an HTTP method
                    "get": {
                        "operationId": "users_ListUsers",
                        "summary": "List users",
                        "parameters": [],
                    },
                }
            }
        }

        samples = harvester.process_spec(spec)
        # Should only process 'get', not 'parameters'
        assert len(samples) == len(PROMPT_TEMPLATES)

    def test_process_spec_skips_missing_summary(self):
        """Test that endpoints without summary are skipped."""
        harvester = GraphAPIHarvester()
        spec = {
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "users_ListUsers",
                        "parameters": [],
                        # No summary
                    }
                }
            }
        }

        samples = harvester.process_spec(spec)
        assert len(samples) == 0

    def test_harvest_creates_output_file(self):
        """Test that harvest creates the output file."""
        harvester = GraphAPIHarvester()

        # Mock the download to avoid network call
        mock_spec = {
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "users_ListUsers",
                        "summary": "List all users",
                        "parameters": [],
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(harvester, "download_spec", return_value=mock_spec):
                output_path = harvester.harvest(
                    output_dir=tmpdir, output_filename="test_dataset.jsonl"
                )

                assert os.path.exists(output_path)

                # Verify content
                with open(output_path, "r") as f:
                    lines = f.readlines()
                    assert len(lines) == len(PROMPT_TEMPLATES)

                    # Verify each line is valid JSON
                    for line in lines:
                        sample = json.loads(line)
                        assert "instruction" in sample
                        assert "input" in sample
                        assert "output" in sample


class TestPromptTemplates:
    """Test suite for prompt templates."""

    def test_all_templates_have_action_placeholder(self):
        """Test that all templates have the {action} placeholder."""
        for template in PROMPT_TEMPLATES:
            assert "{action}" in template

    def test_templates_format_correctly(self):
        """Test that templates format correctly with action."""
        action = "list all users"
        for template in PROMPT_TEMPLATES:
            result = template.format(action=action)
            assert action in result
            assert "{action}" not in result
