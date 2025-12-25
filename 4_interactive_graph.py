#!/usr/bin/env python3
"""
Interactive Agent Script - Backward Compatibility Wrapper.

This script wraps the msgraph_tool_llm.inference.agent module for backward
compatibility with the original pipeline interface.

Usage:
    python 4_interactive_graph.py [options]

For the full package, use:
    msgraph-agent [options]
"""

import sys
import os

# Add src to path for development installs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from msgraph_tool_llm.inference.agent import main

if __name__ == "__main__":
    main()
