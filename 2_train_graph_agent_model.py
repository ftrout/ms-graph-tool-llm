#!/usr/bin/env python3
"""
Training Script - Backward Compatibility Wrapper.

This script wraps the msgraph_tool_llm.training.trainer module for backward
compatibility with the original pipeline interface.

Usage:
    python 2_train_graph_agent_model.py [options]

For the full package, use:
    msgraph-train [options]
"""

import sys
import os

# Add src to path for development installs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from msgraph_tool_llm.training.trainer import main

if __name__ == "__main__":
    main()
