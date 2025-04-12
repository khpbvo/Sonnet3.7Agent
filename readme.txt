# ETMSonnet

A modular, asynchronous code assistant that leverages Claude API for code editing, analysis, and generation.

## Features

- **Dual Agent Architecture**: Uses a router agent to interpret commands and a chat agent for conversations
- **Fully Asynchronous**: Built with Python's asyncio for better performance
- **Claude Tool Integration**: Leverages Claude's function calling capabilities
- **Multi-line Input Support**: Submit multi-line code with the 'END' marker
- **Rich Code Operations**:
  - Read and analyze Python files
  - Generate and modify code
  - Apply changes with diff previews
  - Find and list files
  - Analyze code structure and quality

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ETMSonnet.git
cd ETMSonnet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

4. Run the application:
```bash
python main.py
```

## Usage

### Commands

ETMSonnet supports both slash commands and code-specific commands:

#### Slash Commands
- `/help` - Show help information
- `/exit` - Exit the program
- `/clear` - Clear the conversation history
- `/status` - Show token usage and session information

#### Code Commands
- `code:workdir:/path/to/dir` - Set working directory
- `code:read:path/to/file.py` - Read a file
- `code:read:file1.py,file2.py,file3.py` - Read multiple files
- `code:find:directory` - Find Python files in directory
- `code:find:recursive:directory` - Find Python files recursively
- `code:list` - Show loaded files

#### Code Analysis Commands
- `code:analyze:path/to/file.py` - Analyze Python file
- `code:structure:path/to/file.py` - Show file structure
- `code:pylint:path/to/file.py` - Run Pylint analysis
- `code:fullanalysis:path/to/file.py` - Run full analysis

#### Code Generation Commands
- `code:generate:path/to/file.py:prompt` - Generate code with prompt
- `code:change:path/to/file.py:prompt` - Change existing code with prompt
- `code:diff:path/to/file.py` - Show diff for changes
- `code:apply:path/to/file.py` - Apply changes to file

### Natural Language Command Support

ETMSonnet uses Claude to understand natural language commands, so you can also say things like:
- "Please read the file main.py"
- "Find all Python files in the src directory"
- "Generate a utility function for parsing JSON and save it to utils/json_parser.py"

## Project Structure

```
/ETMSonnet/
│
├── main.py                 # Entry point, minimal orchestration
├── config.py               # Configuration settings
│
├── agents/
│   ├── chat_agent.py       # Claude chat agent for conversation
│   └── router_agent.py     # Agent for interpreting commands
│
├── managers/
│   ├── conversation_manager.py  # Manages conversation history
│   └── file_manager.py          # Manages file operations
│
├── tools/
│   ├── file_tools.py       # File reading/writing tools
│   └── code_tools.py       # Code generation and modification
│
└── utils/
    ├── terminal_utils.py   # Terminal handling utilities
    └── diff_utils.py       # Diff generation and visualization
```

## Dependencies

- Python 3.8+
- anthropic
- tiktoken
- colorama
- pylint (optional for code analysis)

## License

MIT
