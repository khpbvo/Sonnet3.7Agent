"""
Enhanced chat agent that handles interactions with Claude API,
including command recognition and tool calling.
"""

import os
import sys
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
import anthropic
import json
import re

from config import Config


class ChatAgent:
    """
    Chat agent that interacts with Claude API.
    Handles message sending, receiving, streaming, and tool calling.
    """
    
    def __init__(self, api_key: str, config: Config, conversation_manager, file_manager, debug_mode=True):
        """
        Initialize the chat agent.
        
        Args:
            api_key: Anthropic API key
            config: Application configuration
            conversation_manager: Manager for conversation history
            file_manager: Manager for file operations
            debug_mode: Whether to print debug information about tool calls
        """
        # Create the client with the API key directly
        if not api_key:
            print("Warning: No API key provided to ChatAgent")
            
        self.client = anthropic.Anthropic(api_key=api_key)
        self.config = config
        self.conversation_manager = conversation_manager
        self.file_manager = file_manager
        self.tools = []
        self.tool_handlers = {}
        self.debug_mode = debug_mode
        self.tool_call_history = []
        
    def register_tools(self, tools, tool_handlers):
        """
        Register tools with the chat agent.
        
        Args:
            tools: List of available tools
            tool_handlers: Dictionary mapping tool names to handler objects
        """
        self.tools = tools
        self.tool_handlers = tool_handlers

        if self.debug_mode:
            print(f"Registered {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")

    """
    This update focuses on the stream response handling in send_message to ensure
    tool execution works correctly.
    """

    async def send_message(
        self,
        message: str,
        stream_callback: Optional[Callable[[str], None]] = None,
        thinking_enabled: bool = True
    ) -> Optional[str]:
        """
        Send a message to Claude and get a response.
    
        Args:
            message: User message to send
            stream_callback: Callback function for streaming response
            thinking_enabled: Whether to enable Claude's thinking
        
        Returns:
            Claude's response (None if using streaming)
        """
        # Add user message to conversation
        self.conversation_manager.add_message("user", message)

        try:
            # Extract system message and regular messages
            system_message, message_objs = await self.conversation_manager.extract_system_message()
        
            # Format messages correctly for the Messages API
            formatted_messages = self.conversation_manager.format_messages_for_api(message_objs)
        
            # Prepare thinking parameters if enabled
            thinking = {"type": "enabled", "budget_tokens": 16000} if thinking_enabled else None
        
            complete_response = ""
        
            # Prepare tools for API call if we have any
            formatted_tools = [tool.to_dict() for tool in self.tools] if self.tools else None
        
            if self.debug_mode:
                print("\n[DEBUG] Sending message to Claude with:")
                print(f"[DEBUG] - Model: {self.config.model}")
                print(f"[DEBUG] - System message length: {len(system_message or '')}")
                print(f"[DEBUG] - Message history: {len(formatted_messages)} messages")
                print(f"[DEBUG] - Tools: {len(formatted_tools) if formatted_tools else 0} tools")
                if formatted_tools:
                    print(f"[DEBUG] - First tool: {json.dumps(formatted_tools[0], indent=2)}")
        
            # Stream the response if needed
            if stream_callback:
                with self.client.messages.stream(
                    model=self.config.model,
                    max_tokens=self.config.max_response_tokens,
                    system=system_message,
                    messages=formatted_messages,
                    tools=formatted_tools,
                    thinking=thinking
                ) as stream:
                    tool_calls = []
                    current_tool_call = None
                    current_tool_input = ""
                
                    if self.debug_mode:
                        print("[DEBUG] Stream started")
                
                    for event in stream:
                        event_type = getattr(event, 'type', None)
                        if self.debug_mode and event_type:
                            print(f"[DEBUG] Event type: {event_type}")
                        
                        # Regular content block event
                        if event_type == "content_block_delta":
                            if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                                chunk_text = event.delta.text
                                if chunk_text:
                                    complete_response += chunk_text
                                    stream_callback(chunk_text)
                        
                        # Tool call events handling - now checking for input_json events
                        elif event_type == "input_json":
                            if self.debug_mode:
                                print(f"\n[DEBUG] 🔧 Tool call detected via input_json event")
                                print(f"[DEBUG] Partial JSON: {getattr(event, 'partial_json', None)}")
                                print(f"[DEBUG] Snapshot: {getattr(event, 'snapshot', None)}")
                                
                            # Only process when we have a non-empty snapshot (meaning the JSON is complete)
                            if hasattr(event, 'snapshot') and event.snapshot:
                                snapshot = event.snapshot
                                
                                if self.debug_mode:
                                    print(f"[DEBUG] 📦 Complete snapshot received: {snapshot}")
                                
                                # Process the complete tool call
                                try:
                                    # Check if snapshot contains tool information
                                    if isinstance(snapshot, dict):
                                        # Case 1: We have a name in the snapshot
                                        if 'name' in snapshot and snapshot.get('name') in self.tool_handlers:
                                            tool_name = snapshot.get('name')
                                            tool_input = snapshot.get('input', {})
                                            tool_id = snapshot.get('id', f"tool-{len(self.tool_call_history)}")
                                        
                                        # Case 2: We don't have a name, but need to infer it from parameters
                                        else:
                                            # Infer tool based on parameters
                                            tool_name = None
                                            tool_input = snapshot
                                            tool_id = f"tool-{len(self.tool_call_history)}"
                                            
                                            # Infer tool based on parameters
                                            if 'path' in snapshot and len(snapshot) == 1:
                                                # If only path is provided, check if it's a directory or a file
                                                path = snapshot['path']
                                                if self.debug_mode:
                                                    print(f"[DEBUG] 🔍 Inferring tool from path: {path}")
                                                
                                                # Try to check if it's a directory
                                                if os.path.isdir(path):
                                                    tool_name = 'set_working_directory'
                                                    if self.debug_mode:
                                                        print(f"[DEBUG] 🔧 Inferred tool: {tool_name} (path is a directory)")
                                                elif os.path.isfile(path):
                                                    tool_name = 'read_file'
                                                    if self.debug_mode:
                                                        print(f"[DEBUG] 🔧 Inferred tool: {tool_name} (path is a file)")
                                                else:
                                                    # Assume it's a directory change if the path looks like a directory path
                                                    # (ends with / or doesn't have a file extension)
                                                    if path.endswith('/') or '.' not in os.path.basename(path):
                                                        tool_name = 'set_working_directory'
                                                        if self.debug_mode:
                                                            print(f"[DEBUG] 🔧 Inferred tool: {tool_name} (path looks like a directory)")
                                                    else:
                                                        # Default to read_file for any other path
                                                        tool_name = 'read_file'
                                                        if self.debug_mode:
                                                            print(f"[DEBUG] 🔧 Inferred tool: {tool_name} (default for path parameter)")
                                            elif 'path' in snapshot and 'content' in snapshot:
                                                tool_name = 'write_file'
                                                if self.debug_mode:
                                                    print(f"[DEBUG] 🔧 Inferred tool: {tool_name} (path and content parameters)")
                                            elif 'filepath' in snapshot and 'code' in snapshot:
                                                tool_name = 'generate_code'
                                                if self.debug_mode:
                                                    print(f"[DEBUG] 🔧 Inferred tool: {tool_name} (filepath and code parameters)")
                                            elif 'filepath' in snapshot and 'analysis_type' in snapshot:
                                                tool_name = 'analyze_code'
                                                if self.debug_mode:
                                                    print(f"[DEBUG] 🔧 Inferred tool: {tool_name} (filepath and analysis_type parameters)")
                                        
                                        # If we have a valid tool name
                                        if tool_name and tool_name in self.tool_handlers:
                                            if self.debug_mode:
                                                print(f"[DEBUG] 🔧 Using tool: {tool_name}")
                                                print(f"[DEBUG] 📝 Tool input: {tool_input}")
                                            
                                            # Create the tool call object
                                            tool_call = {
                                                "name": tool_name,
                                                "input": tool_input,
                                                "id": tool_id
                                            }
                                            
                                            # Add to tool call history
                                            self.tool_call_history.append(tool_call)
                                            
                                            # Execute the tool
                                            if self.debug_mode:
                                                print(f"[DEBUG] 🛠️ Executing tool: {tool_name}")
                                                
                                            result = await self._handle_tool_call(tool_call)
                                            
                                            # Add result to response - only show details in debug mode
                                            if self.debug_mode:
                                                # In debug mode, show the full tool result
                                                result_text = f"\n\nTool: {tool_name}\nResult: {json.dumps(result, indent=2)}\n"
                                            else:
                                                # In normal mode, just indicate a tool was used
                                                if 'error' in result:
                                                    # Show errors even in non-debug mode
                                                    result_text = f"\n[Tool error: {result['error']}]\n"
                                                else:
                                                    # Don't show successful tool results in regular mode
                                                    result_text = ""
                                            
                                            if result_text:
                                                stream_callback(result_text)
                                                complete_response += result_text
                                            
                                            # Submit tool output back to Claude
                                            try:
                                                # Don't try to submit directly as this SDK version doesn't support it
                                                if self.debug_mode:
                                                    print(f"[DEBUG] ✅ Tool execution complete, adding result to conversation")
                                                
                                                # Add a system message with the tool result for context
                                                self.conversation_manager.add_message(
                                                    "system", 
                                                    f"Tool '{tool_name}' was called with input: {json.dumps(tool_input)} " +
                                                    f"and returned result: {json.dumps(result)}"
                                                )
                                            except Exception as e:
                                                if self.debug_mode:
                                                    print(f"[DEBUG] ❌ Error handling tool result: {str(e)}")
                                        else:
                                            if self.debug_mode:
                                                print(f"[DEBUG] ⚠️ Could not identify a valid tool for snapshot: {snapshot}")
                                    else:
                                        if self.debug_mode:
                                            print(f"[DEBUG] ⚠️ Snapshot is not a dictionary: {snapshot}")
                                
                                except Exception as e:
                                    if self.debug_mode:
                                        print(f"[DEBUG] ❌ Error processing tool snapshot: {str(e)}")
                                        import traceback
                                        traceback.print_exc()
            else:
                # Non-streaming mode
                response = await self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_response_tokens,
                    system=system_message,
                    messages=formatted_messages,
                    tools=formatted_tools
                )
                
                # Extract text content
                complete_response = self._extract_text_from_response(response)
            
            # Add the complete response to conversation history
            if complete_response:
                self.conversation_manager.add_message("assistant", complete_response)
        
            return complete_response if not stream_callback else None
        
        except Exception as e:
            error_msg = f"Error in send_message: {str(e)}"
            print(error_msg, file=sys.stderr)
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    # This method is no longer needed with the new API
    async def _send_tool_results(self, tool_results, previous_messages):
        """
        This method is deprecated with the latest Anthropic API.
        Tool results are now submitted directly through the stream.
        """
        print("[DEBUG] _send_tool_results is deprecated, tool results should be submitted through the stream")
        return "Tool results were processed."

    async def _handle_tool_call(self, tool_call):
        """
        Handle a tool call from Claude.

        Args:
            tool_call: Tool call information

        Returns:
            Tool call result
        """
        tool_name = tool_call.get('name')
        tool_input = tool_call.get('input', {})

        # Enhanced debugging for directory-related commands
        if self.debug_mode and (tool_name == 'set_working_directory' or 
                                (isinstance(tool_input, str) and ('directory' in tool_input.lower() or 'workingdir' in tool_input.lower()))):
            print(f"[DEBUG] 🔍 Processing directory-related tool call: {tool_name}")
            print(f"[DEBUG] 📝 Tool input type: {type(tool_input)} - Content: {tool_input}")

        # Ensure tool_input is a dictionary
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                # Handle specific tools with string inputs
                if tool_name == "read_file":
                    # Try to convert string input to a path parameter
                    tool_input = {"path": tool_input.strip()}
                elif tool_name == "set_working_directory":
                    # Handle direct path inputs for directory changes
                    tool_input = {"path": tool_input.strip()}
                # Add other tool-specific string input handling as needed

        # Special handling for directory-related commands if they got misrouted
        if tool_name == "read_file" and isinstance(tool_input, dict) and "path" in tool_input:
            path = tool_input["path"]
            # Check if this is likely a directory path that should be handled by set_working_directory
            if self.debug_mode:
                print(f"[DEBUG] Checking if '{path}' is a misrouted directory command")
        
            if (path.startswith("/") or path.startswith("C:") or path.startswith("D:")) and not path.endswith((".py", ".txt", ".md", ".json", ".csv")):
                # This looks like a directory path, not a file path
                if os.path.isdir(path):
                    if self.debug_mode:
                        print(f"[DEBUG] Detected a directory path sent to read_file: {path}")
                        print(f"[DEBUG] Redirecting to set_working_directory tool")
                
                    # Redirect to the set_working_directory tool
                    handler = self.tool_handlers.get('set_working_directory')
                    if handler:
                        return await handler.handle_tool_use({
                            "name": "set_working_directory",
                            "input": {"path": path}
                        })

        # Find the appropriate handler
        handler = self.tool_handlers.get(tool_name)

        if not handler:
            error_msg = f"No handler found for tool: {tool_name}"
            if self.debug_mode:
                print(f"[DEBUG] ❌ {error_msg}")
                print(f"[DEBUG] Available handlers: {list(self.tool_handlers.keys())}")
            return {"error": error_msg}

        try:
            # Call the handler with the tool input
            if self.debug_mode:
                print(f"[DEBUG] 🛠️ Executing tool: {tool_name}")
                print(f"[DEBUG] 📝 Tool parameters: {json.dumps(tool_input, indent=2) if isinstance(tool_input, dict) else tool_input}")

            result = await handler.handle_tool_use({
                "name": tool_name,
                "input": tool_input
            })

            if self.debug_mode:
                print(f"[DEBUG] 📊 Tool result: {json.dumps(result, indent=2)}")

            return result

        except Exception as e:
            error_msg = f"Error handling tool call: {str(e)}"
            if self.debug_mode:
                print(f"[DEBUG] ❌ {error_msg}")
                import traceback
                traceback.print_exc()
            return {"error": error_msg}

    async def _handle_slash_command(self, command):
        """
        Handle a slash command.

        Args:
            command: Command string (without the leading slash)

        Returns:
            Command result
        """
        if self.debug_mode:
            print(f"[DEBUG] 🔍 Processing slash command: /{command}")

        if command == 'help':
            return await self._show_help_command()
        elif command == 'exit':
            print("Exiting...")
            sys.exit(0)
        elif command == 'clear':
            self.conversation_manager.clear()
            return "Conversation cleared. New session started."
        elif command == 'status':
            return await self._show_status_command()
        elif command == 'debug':
            self.debug_mode = not self.debug_mode
            return f"Debug mode {'enabled' if self.debug_mode else 'disabled'}"
        elif command == 'tools':
            return self._show_tools_command()
        elif command == 'history':
            return self._show_tool_history()
        else:
            return f"Unknown command: /{command}"

    def _show_tools_command(self):
        """
        Show available tools.

        Returns:
            Tool information
        """
        if not self.tools:
            return "No tools registered."

        tools_info = "Available tools:\n\n"
        for tool in self.tools:
            tools_info += f"- {tool.name}: {tool.description}\n"

            # Show parameters
            if tool.input_schema and 'properties' in tool.input_schema:
                tools_info += "  Parameters:\n"
                for param_name, param_info in tool.input_schema['properties'].items():
                    required = "required" if ('required' in tool.input_schema and param_name in tool.input_schema['required']) else "optional"
                    default = f" (default: {param_info.get('default')})" if ('default' in param_info) else ""
                    tools_info += f"    - {param_name}: {param_info.get('description', 'No description')} ({required}){default}\n"
        tools_info += "\n"
        return tools_info
    
    def _show_tool_history(self):
        """
        Show tool call history.
        
        Returns:
            Tool call history
        """
        if not self.tool_call_history:
            return "No tool calls recorded."
        
        history_info = "Tool call history:\n\n"
        for i, tool_call in enumerate(self.tool_call_history):
            history_info += f"{i+1}. {tool_call['name']}\n"
            history_info += f"   Parameters: {json.dumps(tool_call['input'], indent=2)}\n\n"
        
        return history_info
    
    async def _show_help_command(self):
        """
        Show help information.
        
        Returns:
            Help message
        """
        help_text = f"""
=== ETMSonnet Assistant ===

Slash Commands:
  /help - Show this help information
  /exit - Exit the program
  /clear - Clear the conversation history
  /status - Show token usage and session information
  /debug - Toggle debug mode
  /tools - Show available tools
  /history - Show tool call history

Direct Code Commands:
  code:workdir:/path/to/dir - Set working directory
  code:read:path/to/file.py - Read a file
  code:read:file1.py,file2.py - Read multiple files
  code:find:directory - Find Python files in directory
  code:find:recursive:directory - Find Python files recursively
  code:list - Show loaded files
  code:generate:/path/to/file.py:prompt - Generate code with a prompt
  code:change:/path/to/file.py:prompt - Modify existing code

File Commands:
  You can use natural language to:
  - Read files: "Please read main.py" or "Show me the contents of config.py"
  - Find files: "Find all Python files in the directory" or "List files in the src folder"
  - Set working directory: "Change working directory to /path/to/dir"
  - List loaded files: "What files are currently loaded?"

Code Analysis Commands:
  You can ask the assistant to:
  - Analyze Python files: "Analyze the code in file.py" or "Review the structure of utils.py"
  - Generate code: "Generate a utility function for parsing JSON" or "Create a class for..."
  - Modify code: "Change the function in main.py to handle errors better"
  - Show differences: "What changes would you suggest for this code?"

System Information:
  - Model: {self.config.model}
  - Context tokens: {self.config.max_context_tokens}
  - Working directory: {self.file_manager.get_working_directory()}
  - Debug mode: {'Enabled' if self.debug_mode else 'Disabled'}

Tip: Type 'END' on a new line to finish multi-line input.
"""
        return help_text
    
    async def _show_status_command(self):
        """
        Show status information.
        
        Returns:
            Status message
        """
        session_info = await self.get_session_info()
        
        status_text = f"""
=== Session Status ===

Token usage: ~{session_info['token_count']:,} tokens ({session_info['token_percentage']:.1f}% of maximum)
Messages in history: {session_info['message_count']}
Model: {session_info['model']}
Working directory: {self.file_manager.get_working_directory()}
Debug mode: {'Enabled' if self.debug_mode else 'Disabled'}
Registered tools: {len(self.tools)}
Tool calls: {len(self.tool_call_history)}

{session_info['loaded_files_info']}
"""
        return status_text
        
    # Helper methods for direct tool usage
    
    async def _handle_list_directory(self, path: str) -> Dict[str, Any]:
        """
        Helper method to list directory contents.
        
        Args:
            path: Path to list
            
        Returns:
            Directory contents
        """
        handler = self.tool_handlers.get('list_directory')
        
        if not handler:
            return {"error": "list_directory tool not available"}
        
        # Record the tool call
        tool_call = {
            "name": "list_directory",
            "input": {"path": path},
            "id": "manual_call_2"
        }
        self.tool_call_history.append(tool_call)
        
        # Call the handler
        return await handler.handle_tool_use({
            "name": "list_directory",
            "input": {"path": path}
        })
    
    def _extract_text_from_response(self, response) -> str:
        """
        Extract text content from a Claude API response.
        
        Args:
            response: Claude API response
            
        Returns:
            Extracted text content
        """
        text_content = ""
        
        if hasattr(response, 'content') and response.content:
            for content_block in response.content:
                if content_block.type == "text":
                    text_content += content_block.text
        
        return text_content
    
    async def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current session.
        
        Returns:
            Dictionary with session information
        """
        token_count = self.conversation_manager.get_token_usage()
        token_percentage = self.conversation_manager.get_token_percentage()
        message_count = len(self.conversation_manager.get_messages())
        loaded_files_info = self.conversation_manager.get_loaded_files_info()
        
        return {
            "token_count": token_count,
            "token_percentage": token_percentage,
            "message_count": message_count,
            "loaded_files_info": loaded_files_info,
            "model": self.config.model
        }