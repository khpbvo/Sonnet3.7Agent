"""
ETMSonnet - Fully async, modular assistant for code editing and analysis.
Uses a single chat agent with tool capabilities.
"""

import os
import sys
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import argparse

# Import configuration
from config import Config

# Import managers
from managers.conversation_manager import ConversationManager
from managers.file_manager import FileManager

# Import agents
from agents.chat_agent import ChatAgent

# Import tools
from tools.file_tools import FileTools, register_file_tools, Tool
from tools.code_tools import CodeTools, register_code_tools
from direct_command_handler import DirectCommandHandler

# Import utilities
from utils.terminal_utils import get_multiline_input, print_colored, create_stream_callback

# Import Anthropic SDK
import anthropic


async def setup_system_message(app_context: Dict[str, Any]) -> None:
    """
    Set up the initial system message.
    
    Args:
        app_context: Application context
    """
    conversation_manager = app_context['conversation_manager']
    file_manager = app_context['file_manager']
    config = app_context['config']
    
    system_message = f"""You are Claude, an intelligent code assistant specialized in Python programming.

# ROLE AND CAPABILITIES
- You are an expert in Python code analysis, optimization, and debugging
- You can view and understand Python file contents
- You can suggest specific code changes with precise line and code references
- You can explain code improvements in an educational way
- You can handle file operations and commands through your available tools

# COMMANDS AND TOOLS
- You have access to tools for reading files, listing directories, and finding files
- You can analyze code, generate diffs, and suggest changes
- You should recognize natural language requests for file operations and direct code commands
  - Examples: "read file.py", "find Python files in directory", "set working directory to path"
  - Examples: "code:read:app.py", "code:workdir:/path/to/dir", "code:list", "code:find:directory"
- When the user asks to perform these operations, ALWAYS use the appropriate tool
- ALWAYS respond to the user with the results of your tool operations
- When a file is read, make sure to remember its contents for future reference

## IMPORTANT: USE YOUR TOOLS
- When you see "code:read:" ALWAYS use the read_file tool with the specified file path
- When you see "code:workdir:" ALWAYS use the set_working_directory tool with the path
- When you see a request to change or set working directory, ALWAYS use the set_working_directory tool
- When you see "code:list" ALWAYS use the list_loaded_files tool
- When you see "code:find:" ALWAYS use the list_directory or find_files tool
- When you see "code:generate:" ALWAYS use the generate_code tool
- When you see "code:change:" ALWAYS use the modify_code tool
- When you need to read a file, ALWAYS use the read_file tool
- When you need to find files, ALWAYS use the list_directory or find_files tool
- When users use direct code commands, ALWAYS treat them as explicit tool use instructions

## CRUCIAL BEHAVIOR:
- For changing working directory: ALWAYS use the set_working_directory tool, NEVER just list directory contents
- For any request mentioning "change directory", "set directory", "working directory", or "cd to", use set_working_directory
- After setting a working directory, confirm success and then list the directory contents to show the user what's available
- For generating or modifying code: ALWAYS use the proper code tools (generate_code, modify_code, analyze_code)
- For analyzing code: ALWAYS use analyze_code tool when users ask for code review or analysis
- When diffing code changes: ALWAYS use the generate_diff tool

# INSTRUCTIONS FOR CODE EVALUATION
- Analyze code systematically for readability, effectiveness, and correctness
- Pay attention to type hints, docstrings, naming according to PEP 8, and code organization
- Identify potential bugs, inefficiencies, or security risks
- Be specific about where and why improvements are needed

# INSTRUCTIONS FOR CODE SUGGESTIONS
- Give clear, implementable suggestions for improving code
- Use one of these notations for specific changes:
  1. "Line X: replace 'old_code' with 'new_code'"
  2. Code blocks with - for lines to remove and + for new lines
  3. "Replace this: ```python\\nold\\n``` With this: ```python\\nnew\\n```"
- Explain why each suggested change is an improvement
- Use modern Python conventions (f-strings, type hints, etc.)

# UNDERSTANDING CODE CONTEXT
- When a file is loaded, study it thoroughly before discussing it
- Understand the function and architecture of code before suggesting changes
- Consider existing patterns and coding style
- IMPORTANT: Base your knowledge about a file on the analysis and not on assumptions

# AGENTIC BEHAVIOR
- Be proactive in using your tools to fulfill user requests
- If you need to read a file to answer a question, ALWAYS use the read_file tool
- If you need to find files, ALWAYS use the list_directory or find_files tool
- If you need to analyze code, ALWAYS use the analyze_code tool
- If you need to generate code, ALWAYS use the generate_code tool
- If you need to modify code, ALWAYS use the modify_code tool
- When you make changes to code, remember to explain what you did and why
- ALWAYS use the proper tool for each task - never try to handle file or code operations manually

For file operations and code tasks, ALWAYS use your available tools rather than saying you can't do something.

Context window is set to maximum {config.max_context_tokens} tokens.
"""

    conversation_manager.add_message("system", system_message)


async def main():
    """Main application entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="ETMSonnet Assistant")
    parser.add_argument("--api-key", help="Anthropic API key")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Get API key from arguments, environment, or prompt
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = input("Enter your Anthropic API key: ").strip()
    
    # Set the API key in environment as well for Anthropic SDK
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    # Print debug info about the API key (masked for security)
    if api_key:
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "****"
        print(f"Using API key: {masked_key}")
    else:
        print("Warning: No API key provided")
    
    # Test the API connection directly
    try:
        print("Testing API connection...")
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=10,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ]
        )
        print("API connection successful!")
    except Exception as e:
        print(f"API connection test failed: {str(e)}")
        # Continue anyway to allow troubleshooting
    
    # Initialize configuration
    config = Config()
    
    # Initialize managers
    conversation_manager = ConversationManager(max_tokens=config.max_context_tokens)
    file_manager = FileManager(conversation_manager)
    
    # Initialize tools
    file_tools = FileTools(file_manager)
    code_tools = CodeTools(file_manager)
    
    # Register tools
    all_tools = register_file_tools() + register_code_tools()
    
    # Create a tool handler mapping
    tool_handlers = {
        # File tools
        'read_file': file_tools,
        'write_file': file_tools,
        'list_directory': file_tools,
        'find_files': file_tools,
        'generate_diff': file_tools,
        'list_loaded_files': file_tools,
        'set_working_directory': file_tools,
        
        # Code tools
        'generate_code': code_tools,
        'modify_code': code_tools,
        'parse_diff_suggestions': code_tools,
        'apply_changes': code_tools,
        'analyze_code': code_tools
    }
    # Print registered tools for debugging
    print("Registered tool handlers:")
    for tool_name, handler in tool_handlers.items():
        print(f"  - {tool_name}: {type(handler).__name__}")
    
    # Initialize the enhanced chat agent with direct access to file_manager and debug mode
    debug_mode = args.debug
    chat_agent = ChatAgent(api_key, config, conversation_manager, file_manager, debug_mode=debug_mode)
    chat_agent.register_tools(all_tools, tool_handlers)
    
    
    debug_mode = args.debug
    chat_agent = ChatAgent(api_key, config, conversation_manager, file_manager, debug_mode=debug_mode)
    chat_agent.register_tools(all_tools, tool_handlers)
    # Application context
    app_context = {
        "config": config,
        "conversation_manager": conversation_manager,
        "file_manager": file_manager,
        "chat_agent": chat_agent,
        "file_tools": file_tools,
        "code_tools": code_tools
    }
    
    # Setup initial system message
    await setup_system_message(app_context)
    
    # Print welcome message
    print_colored(f"ETMSonnet Assistant (Claude {config.model})", "cyan", bold=True)
    print_colored("Type /help for available commands or /exit to quit", "blue")
    print_colored("Type 'END' on a new line to finish multi-line input", "blue")
    print_colored(f"Debug mode is {'enabled' if debug_mode else 'disabled'} - use /debug to toggle", "blue")
    print()
    
    # Main loop
    try:
        while True:
            # Get user input
            user_input = get_multiline_input("\nYou: ")
            if not user_input.strip():
                continue
        
            # Enable direct command processing
            direct_result = await direct_command_handler.process_command(user_input)
        
            if direct_result:
                print_colored("\nAssistant: ", "green", bold=True)
                print(f"I've processed your command directly: {direct_result}")
            
                # Still send a version to Claude to maintain conversation context
                callback = create_stream_callback(config.typing_simulation_delay)
                print_colored("\nAdditional response from Claude: ", "cyan")
                await chat_agent.send_message(user_input, callback)
            else:
                # Create streaming callback
                callback = create_stream_callback(config.typing_simulation_delay)
            
                print_colored("\nAssistant: ", "green", bold=True)
                await chat_agent.send_message(user_input, callback)
        
            # Show token usage if it's high
            token_percentage = conversation_manager.get_token_percentage()
            if token_percentage > 50:
                print_colored(f"\n[Token usage: ~{conversation_manager.get_token_usage():,} tokens ({token_percentage:.1f}% of max)]", "yellow")

    except KeyboardInterrupt:
        print_colored("\nExiting...", "cyan")
        sys.exit(0)
    except Exception as e:
        print_colored(f"\nError: {str(e)}", "red")
        import traceback
        traceback.print_exc()
        sys.exit(1)