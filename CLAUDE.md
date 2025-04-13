# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Setup**: `pip install -r requirements.txt`
- **Run**: `python main.py [--api-key YOUR_KEY] [--debug]`
- **Lint**: `pylint agents tools managers utils config.py main.py`

## Code Style Guidelines

- **Imports**: Group by standard lib → third-party → local modules
- **Type Annotations**: Use typing module (Optional, List, Dict, etc.) for all functions
- **Documentation**: Google-style docstrings with Args/Returns/Raises sections
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPER_CASE for constants
- **Error Handling**: Use specific exceptions, include error context, use fallback mechanisms
- **Formatting**: Follow PEP 8, 4-space indentation, max 100 chars per line
- **Async Pattern**: Use async/await consistently, proper exception handling in async context

## Project Structure

- `agents/`: Main communication components
- `managers/`: State and resource management
- `tools/`: API-facing functionality
- `utils/`: Helper functions and utilities

All new code should maintain the existing patterns of modular architecture with clear separation of concerns.