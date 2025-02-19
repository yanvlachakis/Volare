# Contributing to Volare

First off, thank you for considering contributing to Volare! It's people like you that make Volare such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by the [Volare Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check [this list](https://github.com/volare-trading/volare-framework/issues) as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include details about your configuration and environment

### Suggesting Enhancements

Enhancement suggestions are tracked as [GitHub issues](https://github.com/volare-trading/volare-framework/issues). When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful

### Pull Requests

* Fill in [the required template](PULL_REQUEST_TEMPLATE.md)
* Do not include issue numbers in the PR title
* Follow the [styleguides](#styleguides)
* Include appropriate test coverage
* End all files with a newline

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Python Styleguide

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* Use type hints for function arguments and return values
* Use docstrings for all public methods and functions
* Keep functions focused and small
* Use meaningful variable names

### Documentation Styleguide

* Use [Markdown](https://daringfireball.net/projects/markdown/) for documentation
* Reference function and variable names using backticks
* Include code examples when relevant
* Keep documentation up to date with code changes

## Development Process

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Set up development environment:
   ```bash
   poetry install --with dev
   pre-commit install
   ```
4. Make your changes
5. Run tests and linting:
   ```bash
   poetry run pytest
   poetry run flake8
   poetry run mypy .
   ```
6. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```
7. Push to the branch:
   ```bash
   git push origin feature/amazing-feature
   ```
8. Open a Pull Request

## Setting Up Development Environment

1. Install dependencies:
   ```bash
   poetry install --with dev
   ```

2. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

3. Create test environment file:
   ```bash
   cp .env.example .env.test
   ```

4. Run tests:
   ```bash
   poetry run pytest
   ```

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=volare

# Run specific test file
poetry run pytest tests/test_specific_file.py

# Run tests with specific marker
poetry run pytest -m "integration"
```

### Writing Tests

* Write tests for all new features
* Follow the Arrange-Act-Assert pattern
* Use meaningful test names that describe the behavior being tested
* Use fixtures for common setup
* Mock external dependencies

## Code Review Process

The core team looks at Pull Requests on a regular basis. After feedback has been given we expect responses within two weeks. After two weeks we may close the PR if it isn't showing any activity.

## Community

* Join our [Discord server](https://discord.gg/volare-trading)
* Follow us on [Twitter](https://twitter.com/volare_trading)
* Read our [Blog](https://blog.volare-trading.com)

## Recognition

Contributors who have made significant improvements will be recognized in our [CONTRIBUTORS.md](CONTRIBUTORS.md) file.

## Questions?

Feel free to contact the core team at dev@volare-trading.com 