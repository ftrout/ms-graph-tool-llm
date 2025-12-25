#!/usr/bin/env python3
"""
Data Harvesting Script - Backward Compatibility Wrapper.

This script wraps the msgraph_tool_llm.data.harvester module for backward
compatibility with the original pipeline interface.

Usage:
    python 1_graph_api_harvester.py [options]

For the full package, use:
    msgraph-harvest [options]
"""

import sys
import os

# Add src to path for development installs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from msgraph_tool_llm.data.harvester import main

if __name__ == "__main__":
    main()
