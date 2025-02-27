# ClaudeWoW Project Guidelines

## Build/Test Commands
- Start the application: `python main.py`
- Run code formatter: `black src/`
- Lint code: `pylint src/`
- Type checking: `mypy src/`
- Run tests: `pytest tests/`
- Run specific test: `pytest tests/path/to/test_file.py::test_function_name`

## Code Style Guidelines
- **Formatting**: Use Black with default settings
- **Imports**: Group standard library, third-party, and local imports with blank lines between groups
- **Type Hints**: Always use type hints (typing module) for function parameters and return values
- **Docstrings**: All modules, classes, and functions should have docstrings following Google style
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPER_CASE for constants
- **Error Handling**: Use try/except blocks with specific exceptions, log all errors
- **Logging**: Use Python's logging module with proper levels (DEBUG, INFO, WARNING, ERROR)
- **Configuration**: Store configuration in JSON files, access via utils.config module
- **Model Structure**: Maintain clear separation between perception, decision, and action components

For developing new features, follow existing patterns in similar modules.