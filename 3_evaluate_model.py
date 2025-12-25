#!/usr/bin/env python3
"""
Evaluation Script - Backward Compatibility Wrapper.

This script wraps the msgraph_tool_llm.evaluation.evaluator module for backward
compatibility with the original pipeline interface.

Usage:
    python 3_evaluate_model.py [options]

For the full package, use:
    msgraph-evaluate [options]
"""

import sys
import os

# Add src to path for development installs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from msgraph_tool_llm.evaluation.evaluator import main

if __name__ == "__main__":
    main()
