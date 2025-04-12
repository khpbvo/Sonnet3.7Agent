"""
ETMSonnet - Fully async, modular assistant for code editing and analysis.
Uses dual agent architecture for command routing and understanding.
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
from agents.router_agent import RouterAgent

# Import tools
from tools.file_tools import FileTools, register_file_tools
from tools.code_tools import CodeTools, register_code_tools

# Import utilities
from utils.terminal_utils import get_multiline_input, print_colored, create_stream_callback


async def process_command(command_data: Dict[str, Any], app_context: Dict[str, Any]) -> str:
    """
    Process a command based on its type.
    
    Args:
        command_data: Command data from router
        app_context: Application context
        
    Returns:
        Command result message
    """
    command_type = command_data.get('command_type')
    command = command_data.get('command', '')
    args = command_data.get('args', [])
    
    # Process slash commands
    if command_type == 'slash':
        if command == 'help':
            return await show_help_command(app_context)
        elif command == 'exit':
            print_colored("Exiting...", "cyan")
            sys.exit(0)
        elif command == 'clear':
            app_context['conversation_manager'].clear()
            return "Conversation cleared. New session started."
        elif command == 'status':
            return await show_status_command(app_context)
        else:
            return f"Unknown command: /{command}"
    
    # Process code commands
    elif command_type == 'code':
        file_manager = app_context['file_manager']
        conversation_manager = app_context['conversation_manager']
        
        if command == 'workdir':
            directory = args[0] if args else ""
            if not directory:
                return f"Current working directory: {file_manager.get_working_directory()}"
            else:
                return file_manager.set_working_directory(directory)
                
        elif command == 'read':
            if not args:
                return "Please specify a file to read: code:read:path/to/file.py"
            
            try:
                # Handle multiple files separated by commas
                file_paths = [path.strip() for path in args[0].split(',')] if args else []
                results = []
                success_count = 0
                
                for filepath in file_paths:
                    try:
                        content = await file_manager.read_file(filepath)
                        lines = content.splitlines()
                        results.append(f"File '{filepath}' loaded ({len(lines)} lines).")
                        success_count += 1
                    except Exception as e:
                        results.append(f"Error reading file '{filepath}': {str(e)}")
                
                if len(file_paths) == 1:
                    return results[0]
                else:
                    summary = f"{success_count} of {len(file_paths)} files successfully loaded."
                    return summary + "\n" + "\n".join(results)
                    
            except Exception as e:
                return f"Error reading file: {str(e)}"
                
        elif command == 'find':
            directory = args[0] if args else "."
            recursive = False
            
            # Check for recursive flag
            if directory.startswith("recursive:"):
                recursive = True
                directory = directory[len("recursive:"):]
            
            # Check for directory: prefix
            if directory.startswith("directory:"):
                directory = directory[len("directory:"):]
            
            try:
                files = await file_manager.find_python_files(directory, recursive)
                if not files:
                    return f"No Python files found in '{directory}'."
                
                result = f"Python files in '{directory}':"
                for f in files:
                    result += f"\n- {f}"
                return result
            except Exception as e:
                return f"Error finding files: {str(e)}"
                
        elif command == 'list':
            return conversation_manager.get_loaded_files_info()
            
        else:
            # For other commands like analyze, generate, pylint, etc.
            # Will be implemented by the tool framework
            return f"Command not implemented yet: code:{command}"
    
    # Process inferred commands
    elif command_type == 'inferred':
        # For inferred commands, construct a code: command and recurse
        if command and args:
            code_command = f"code:{command}:{':'.join(args)}"
            print_colored(f"Executing inferred command: {code_command}", "cyan")
            
            # Parse the command
            parts = code_command.split(':', 2)
            if len(parts) >= 2:
                subcommand = parts[1].lower()
                subargs = parts[2].split(':') if len(parts) > 2 else []
                
                # Recurse with the explicit command
                return await process_command({
                    'command_type': 'code',
                    'command': subcommand,
                    'args': subargs
                }, app_context)
        
        return "Could not process inferred command"
    
    return f"Unknown command type: {command_type}"


async def show_help_command(app_context: Dict[str, Any]) -> str:
    """
    Show help information.
    
    Args:
        app_context: Application context
        
    Returns:
        Help message
    """
    config = app_context['config']
    
    help_text = f"""
=== ETMSonnet Assistant ===

Slash Commands:
  /help - Show this help information
  /exit - Exit the program
  /clear - Clear the conversation history
  /status - Show token usage and session information

Code Commands:
  code:workdir:/path/to/dir - Set working directory
  code:read:path/to/file.py - Read a file
  code:read:file1.py,file2.py,file3.py - Read multiple files
  code:find:directory - Find Python files in directory
  code:find:recursive:directory - Find Python files recursively
  code:list - Show loaded files

Code Analysis Commands:
  code:analyze:path/to/file.py - Analyze Python file
  code:structure:path/to/file.py - Show file structure
  code:pylint:path/to/file.py - Run Pylint analysis
  code:fullanalysis:path/to/file.py - Run full analysis

Code Generation Commands:
  code:generate:path/to/file.py:prompt - Generate code with prompt
  code:change:path/to/file.py:prompt - Change existing code with prompt
  code:diff:path/to/file.py - Show diff for changes
  code:apply:path/to/file.py - Apply changes to file

System Information:
  - Model: {config.model}
  - Context tokens: {config.max_context_tokens}
  - Working directory: {app_context['file_manager'].get_working_directory()}

Tip: Type 'END' on a new line to finish multi-line input.
"""
    return help_text


async def show_status_command(app_context: Dict[str, Any]) -> str:
    """
    Show status information.
    
    Args:
        app_context: Application context
        
    Returns:
        Status message
    """
    chat_agent = app_context['chat_agent']
    session_info = await chat_agent.get_session_info()
    
    status_text = f"""
=== Session Status ===

Token usage: ~{session_info['token_count']:,} tokens ({session_info['token_percentage']:.1f}% of maximum)
Messages in history: {session_info['message_count']}
Model: {session_info['model']}
Working directory: {app_context['file_manager'].get_working_directory()}

{session_info['loaded_files_info']}
"""
    return status_text


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
- You can recognize natural language requests to open files, apply code changes, etc.
- Refer to loaded files and their content in your answers
- Be specific in your suggestions so they are easy to apply

Context window is set to maximum {config.max_context_tokens} tokens.
"""

    conversation_manager.add_message("system", system_message)


async def main():
    """Main application entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="ETMSonnet Assistant")
    parser.add_argument("--api-key", help="Anthropic API key")
    args = parser.parse_args()
    
    # Get API key from arguments, environment, or prompt
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = input("Enter your Anthropic API key: ").strip()
    
    # Initialize configuration
    config = Config()
    
    # Initialize managers
    conversation_manager = ConversationManager(max_tokens=config.max_context_tokens)
    file_manager = FileManager(conversation_manager)
    
    # Initialize agents
    chat_agent = ChatAgent(api_key, config, conversation_manager)
    router_agent = RouterAgent(api_key, config)
    
    # Initialize tools
    file_tools = FileTools(file_manager)
    code_tools = CodeTools(file_manager)
    
    # Application context
    app_context = {
        "config": config,
        "conversation_manager": conversation_manager,
        "file_manager": file_manager,
        "chat_agent": chat_agent,
        "router_agent": router_agent,
        "file_tools": file_tools,
        "code_tools": code_tools
    }
    
    # Setup initial system message
    await setup_system_message(app_context)
    
    # Print welcome message
    print_colored(f"ETMSonnet Assistant (Claude {config.model})", "cyan", bold=True)
    print_colored("Type /help for available commands or /exit to quit", "blue")
    print_colored("Type 'END' on a new line to finish multi-line input", "blue")
    print()
    
    # Main loop
    try:
        while True:
            # Get user input
            user_input = get_multiline_input("\nYou: ")
            if not user_input.strip():
                continue
            
            # Route the input
            route_type, command_data = await router_agent.route_input(user_input)
            
            if route_type == 'command':
                # Process command
                result = await process_command(command_data, app_context)
                print_colored("\nAssistant: ", "green", bold=True)
                print(result)
                
                # Add info about command to conversation if needed
                if 'original_query' in command_data:
                    # For inferred commands, send the original query to Claude
                    original_query = command_data['original_query']
                    if original_query:
                        print("\nProcessing remaining query:", original_query)
                        
                        # Create streaming callback
                        callback = create_stream_callback(config.typing_simulation_delay)
                        
                        print_colored("\nAssistant: ", "green", bold=True)
                        await chat_agent.send_message(original_query, callback)
                
            else:  # chat
                # Send to chat agent
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


if __name__ == "__main__":
    asyncio.run(main())
