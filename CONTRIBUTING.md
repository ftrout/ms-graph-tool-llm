# Contributing to SecurityGraph-Agent

Thank you for your interest in contributing to SecurityGraph-Agent! This document provides guidelines and information for contributors.

## Getting Started

### Development Environment

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/securitygraph-agent
   cd securitygraph-agent
   ```

2. **Set up the development environment**

   **Option A: Dev Container (Recommended)**
   - Open the project in VS Code
   - Click "Reopen in Container" when prompted
   - All dependencies will be installed automatically

   **Option B: Local Installation**
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking

Run these locally before committing:

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/ --fix

# Type check
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/defender_api_tool --cov-report=html

# Run specific test file
pytest tests/test_harvester.py -v
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run them manually:

```bash
pre-commit run --all-files
```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-new-security-tool`
- `fix/json-parsing-error`
- `docs/update-readme`
- `refactor/improve-training-loop`

### Commit Messages

Write clear, concise commit messages:

- Use the imperative mood ("Add feature" not "Added feature")
- First line should be 50 characters or less
- Add details in the body if needed

Example:
```
Add support for batch inference

- Implement batch processing in DefenderApiAgent.generate()
- Add max_batch_size parameter
- Update documentation with batch usage examples
```

### Pull Requests

1. **Create a feature branch** from `main`
2. **Make your changes** with clear, atomic commits
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Run the full test suite** locally
6. **Open a Pull Request** with:
   - Clear description of changes
   - Link to any related issues
   - Screenshots/examples if applicable

## Areas for Contribution

### Good First Issues

Look for issues labeled `good first issue` for beginner-friendly tasks.

### High-Impact Areas

- **New Security API Coverage**: Add support for more Defender XDR endpoints
- **Improved Prompts**: Better security-focused prompt templates
- **Evaluation Metrics**: Additional metrics for security model evaluation
- **Documentation**: Improve examples and tutorials
- **Testing**: Increase test coverage

### Adding New Security Tools

When adding new Microsoft Defender XDR API tools:

1. Add the endpoint to `TARGET_NAMESPACES` in `harvester.py` if needed
2. Create example tool definitions in `agent.py` under `COMMON_TOOLS`
3. Add tests for the new functionality
4. Update documentation with usage examples

## Code Review Process

1. All changes require review before merging
2. Address reviewer feedback promptly
3. Keep discussions constructive and professional
4. Squash commits when requested for cleaner history

## Reporting Issues

### Bug Reports

Include:
- Python version and OS
- Package version
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Include:
- Clear description of the feature
- Security use case / motivation
- Example API or usage pattern
- Any implementation ideas

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to:
- Open a GitHub Discussion
- Comment on relevant issues
- Reach out to maintainers

Thank you for contributing!
