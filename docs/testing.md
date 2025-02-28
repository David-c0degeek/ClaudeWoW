# Testing Guide for ClaudeWoW

This document provides comprehensive guidance on testing ClaudeWoW. It covers unit testing, integration testing, and performance testing approaches.

## Testing Philosophy

ClaudeWoW follows these testing principles:

1. **Test-Driven Development** - Write tests before implementing features
2. **Comprehensive Coverage** - Aim for high test coverage, especially for critical components
3. **Realistic Scenarios** - Test with realistic game scenarios
4. **Automated Testing** - Prioritize automated tests over manual testing
5. **Performance Validation** - Include performance benchmarks in test suite

## Test Organization

Tests are organized in a structure that mirrors the source code:

```
tests/
  combat/            # Tests for combat system components
  decision/          # Tests for decision-making components
  perception/        # Tests for perception components
  action/            # Tests for action execution components
  knowledge/         # Tests for game knowledge components
  learning/          # Tests for learning components
  social/            # Tests for social components
  economic/          # Tests for economic components
  utils/             # Tests for utility functions
  integration/       # Integration tests across components
  conftest.py        # Test fixtures and configuration
  pytest.ini         # PyTest configuration
```

## Setting Up the Test Environment

### Prerequisites

- Python 3.9+
- All dependencies from requirements.txt
- pytest and pytest-cov

### Installation

```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

### Configuration

The test suite uses the configuration in `tests/conftest.py`. This file provides fixtures like mock game state, mock game knowledge, and test configuration.

## Running Tests

### Running All Tests

```bash
pytest
```

### Running Specific Test Categories

```bash
# Run combat system tests
pytest tests/combat/

# Run tests for a specific file
pytest tests/combat/test_warrior_combat.py

# Run a specific test
pytest tests/combat/test_warrior_combat.py::TestWarriorCombatModule::test_arms_rotation
```

### Test Tags

Tests are organized with markers to allow selective execution:

```bash
# Run only unit tests
pytest -m "not integration"

# Run only fast tests
pytest -m "not slow"

# Run only combat-related tests
pytest -m combat
```

Available markers:
- `slow`: Tests that take a long time to execute
- `integration`: Integration tests between components
- `combat`, `learning`, `social`, etc.: Component-specific markers

## Writing Tests

### Unit Tests

Here's how to write a basic unit test:

```python
import unittest
from src.module_to_test import FunctionToTest

class TestModuleName(unittest.TestCase):
    
    def setUp(self):
        # Setup code runs before each test
        self.test_instance = FunctionToTest()
    
    def test_specific_functionality(self):
        # Arrange
        input_data = "test input"
        expected_output = "expected result"
        
        # Act
        actual_output = self.test_instance.process(input_data)
        
        # Assert
        self.assertEqual(actual_output, expected_output)
```

### Using Test Fixtures

The `conftest.py` file provides common fixtures:

```python
def test_function_with_fixtures(mock_config, mock_game_state):
    # mock_config and mock_game_state are automatically 
    # injected fixtures from conftest.py
    assert mock_game_state.player_class == "warrior"
```

### Mocking

For components that interact with external systems, use mocking:

```python
from unittest.mock import MagicMock, patch

def test_with_mocks():
    with patch('src.module.ExternalDependency') as mock_dependency:
        mock_dependency.return_value.process.return_value = "mocked result"
        
        # Test code that uses ExternalDependency
```

## Test Data

### Sample Game State

Tests use a mock game state defined in `conftest.py` that simulates a real-world game state.

### Recorded Game Data

For integration tests, recorded game data is stored in `tests/data/recordings/`.

## Performance Testing

Performance tests measure system performance:

1. **Benchmark Tests**: Measure execution time for critical operations
2. **Resource Usage Tests**: Monitor memory and CPU usage
3. **Latency Tests**: Measure time between perception and action

Example performance test:

```python
def test_pathfinding_performance():
    start_point = (100, 100)
    end_point = (1000, 1000)
    
    start_time = time.time()
    path = navigator.find_path(start_point, end_point)
    execution_time = time.time() - start_time
    
    assert execution_time < 0.5  # Should complete in under 500ms
    assert len(path) > 0  # Should find a valid path
```

## Integration Tests

Integration tests verify that components work together correctly:

```python
def test_navigation_with_perception():
    # Test that perception component correctly identifies obstacles
    # and navigation component correctly plans paths around them
    perception = PerceptionSystem()
    navigation = NavigationSystem()
    
    game_state = perception.analyze_screen(test_screen_image)
    path = navigation.plan_path(game_state, destination)
    
    assert path is not None
    assert len(path) > 0
```

## Code Coverage

Track test coverage using pytest-cov:

```bash
pytest --cov=src --cov-report=html
```

This generates an HTML report in the `htmlcov` directory, showing which lines of code are tested.

## Continuous Integration

Tests run automatically on GitHub Actions:

1. **Pull Request Checks**: Run tests on every pull request
2. **Main Branch Validation**: Run tests after merges to main
3. **Nightly Builds**: Run full test suite including slow and integration tests

## Best Practices

1. **Test Isolation**: Tests should not depend on each other
2. **Clean Up**: Tests clean up any resources they create
3. **Realistic Testing**: Use realistic game scenarios
4. **Clear Purpose**: Each test has a single, clear purpose
5. **Self-Documenting**: Test names clearly describe what they test

## Troubleshooting Tests

Common test issues:

1. **Flaky Tests**: Tests that sometimes pass and sometimes fail
   - Solution: Use deterministic inputs and mock randomness

2. **Slow Tests**:
   - Solution: Mark as slow and optimize when possible

3. **Dependencies**:
   - Solution: Use dependency injection and mocking

## Adding New Tests

When adding new features:

1. Write tests first (TDD approach)
2. Ensure tests cover basic functionality, edge cases, and error cases
3. Add corresponding integration tests if needed
4. Add to CI pipeline if needed

For bug fixes:

1. Write a test that reproduces the bug
2. Fix the code until the test passes
3. Ensure the fix doesn't break existing functionality