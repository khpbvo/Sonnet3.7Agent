"""
Enhanced chat agent that handles interactions with Claude API,
including command recognition, tool calling, and tool chaining.
"""

import os
import sys
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
import anthropic
import json
import re

from config import Config
from utils.terminal_utils import print_status, print_colored


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
        self.pending_tool_chains = []  # Track pending tool chains
        
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
    tool execution works correctly and supports tool chaining.
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
                                            # New inference cases for code modification tools
                                            elif 'filepath' in snapshot and 'original_code' in snapshot and 'new_code' in snapshot:
                                                tool_name = 'modify_code'
                                                if self.debug_mode:
                                                    print(f"[DEBUG] 🔧 Inferred tool: {tool_name} (filepath, original_code, and new_code parameters)")
                                            elif 'original' in snapshot and 'modified' in snapshot:
                                                tool_name = 'generate_diff'
                                                if self.debug_mode:
                                                    print(f"[DEBUG] 🔧 Inferred tool: {tool_name} (original and modified parameters)")
                                            elif 'suggestion_text' in snapshot and len(snapshot) == 1:
                                                tool_name = 'parse_diff_suggestions'
                                                if self.debug_mode:
                                                    print(f"[DEBUG] 🔧 Inferred tool: {tool_name} (suggestion_text parameter)")
                                            elif 'filepath' in snapshot and 'changes' in snapshot:
                                                tool_name = 'apply_changes'
                                                if self.debug_mode:
                                                    print(f"[DEBUG] 🔧 Inferred tool: {tool_name} (filepath and changes parameters)")
                                        
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
                                            
                                            # Print tool status (regardless of debug mode)
                                            self.print_tool_status(tool_name, tool_input)
                                            
                                            # NEW: Check for potential tool chaining
                                            next_tool = self._check_for_tool_chain(tool_name, tool_input, complete_response)
                                            
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
                                            
                                            # NEW: Execute the next tool in the chain if needed
                                            if next_tool:
                                                if self.debug_mode:
                                                    print(f"[DEBUG] 🔗 Chaining to next tool: {next_tool['name']}")
                                                
                                                next_result = await self._handle_tool_call(next_tool)
                                                
                                                # Add chained tool result to response
                                                if self.debug_mode:
                                                    chain_result_text = f"\n\nChained Tool: {next_tool['name']}\nResult: {json.dumps(next_result, indent=2)}\n"
                                                    stream_callback(chain_result_text)
                                                    complete_response += chain_result_text
                                            
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
                                                
                                                # Add chained tool result to context if applicable
                                                if next_tool:
                                                    self.conversation_manager.add_message(
                                                        "system",
                                                        f"Chained tool '{next_tool['name']}' was called automatically with input: {json.dumps(next_tool['input'])} " +
                                                        f"and returned result: {json.dumps(next_result)}"
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

    def _check_for_tool_chain(self, tool_name: str, tool_input: Dict[str, Any], current_response: str) -> Optional[Dict[str, Any]]:
        """
        Check if a tool call should trigger a chain of tools.
        
        Args:
            tool_name: Name of the current tool
            tool_input: Input parameters for the current tool
            current_response: Current response text from Claude
            
        Returns:
            Next tool call information or None if no chaining needed
        """
        # Check for common chaining patterns
        
        # Pattern 1: Directory navigation should be followed by listing directory
        if tool_name == 'set_working_directory':
            path = tool_input.get('path', '.')
            
            # Create a chained list_directory call
            return {
                "name": "list_directory",
                "input": {"path": path},
                "id": f"chain-{len(self.tool_call_history)}"
            }
            
        # Pattern 2: Reading a file for analysis should be followed by analyze_code (if Python file)
        elif tool_name == 'read_file':
            path = tool_input.get('path', '')
            
            # If it's a Python file and context suggests analysis
            if path.endswith('.py') and ('analyze' in current_response.lower() or 'review' in current_response.lower()):
                return {
                    "name": "analyze_code",
                    "input": {"filepath": path, "analysis_type": "basic"},
                    "id": f"chain-{len(self.tool_call_history)}"
                }
                
        # Pattern 3: When finding files, if only one match, read it automatically
        elif tool_name == 'find_files':
            # Note: We don't have the result yet, so this isn't practical to implement here
            # This would need to be handled after we have the result from find_files
            pass
                
        return None

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

        # Always print tool status information
        self.print_tool_status(tool_name, tool_input)

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
                elif tool_name == "generate_diff":
                    # Handle string input as file path for diff
                    if ":" in tool_input:
                        # If format is like "old_file:new_file", split it
                        parts = tool_input.strip().split(":")
                        if len(parts) == 2:
                            tool_input = {"original": parts[0], "modified": parts[1]}
                elif tool_name == "modify_code":
                    # Try to convert string input to a filepath parameter
                    tool_input = {"filepath": tool_input.strip(), "original_code": "", "new_code": ""}
                elif tool_name == "generate_code":
                    # Handle string input as filepath
                    tool_input = {"filepath": tool_input.strip(), "code": ""}
                elif tool_name == "analyze_code":
                    # Handle string input as filepath with default analysis type
                    tool_input = {"filepath": tool_input.strip(), "analysis_type": "basic"}
                elif tool_name == "parse_diff_suggestions":
                    # Handle string input as suggestion text
                    tool_input = {"suggestion_text": tool_input.strip()}
                elif tool_name == "apply_changes":
                    # Handle string input as filepath
                    tool_input = {"filepath": tool_input.strip(), "changes": []}

        # NEW: Special handling for chains that require reading before modification
        if tool_name == "modify_code" and isinstance(tool_input, dict) and "filepath" in tool_input:
            filepath = tool_input["filepath"]
            # Check if we need to read the file first
            if not self._has_recent_read(filepath):
                if self.debug_mode:
                    print(f"[DEBUG] 🔗 Auto-chaining: Need to read {filepath} before modifying")
                
                # Read the file first
                read_handler = self.tool_handlers.get('read_file')
                if read_handler:
                    read_result = await read_handler.handle_tool_use({
                        "name": "read_file",
                        "input": {"path": filepath}
                    })
                    
                    if "error" not in read_result:
                        if self.debug_mode:
                            print(f"[DEBUG] 📄 Auto-read successful, proceeding with modification")
                        
                        # Update the original_code if it's empty
                        if not tool_input.get("original_code") and "content" in read_result:
                            tool_input["original_code"] = read_result["content"]
                    else:
                        return {"error": f"Failed to read file before modification: {read_result['error']}"}

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

        # Special handling for code:change: commands that might be misrouted
        if tool_name == "generate_code" and isinstance(tool_input, dict) and "filepath" in tool_input and "code" in tool_input:
            # Check if this is a code change command rather than code generation
            filepath = tool_input["filepath"]
            code = tool_input["code"]
            
            if self.debug_mode:
                print(f"[DEBUG] Checking if '{filepath}' should be a modify_code call instead")
            
            # If file exists, it might be a code modification rather than generation
            if os.path.exists(self.file_manager._get_absolute_path(filepath)):
                try:
                    # Read the file to see if it exists
                    content = await self.file_manager.read_file(filepath)
                    
                    if content.strip() and not tool_input.get("force_create", False):
                        if self.debug_mode:
                            print(f"[DEBUG] File exists and has content. This might be a code modification.")
                            
                        # Ask user for confirmation before overwriting completely
                        if "confirm" in tool_input and tool_input["confirm"]:
                            # This should be handled by the generate_code handler itself
                            pass
                        else:
                            # Try to see if this is a partial modification instead of full replacement
                            # For now, we'll leave this as generate_code
                            pass
                            
                except Exception:
                    # If file can't be read, continue with original handler
                    pass

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
            
            # NEW: Check for follow-up tool chaining based on result
            next_tool = self._check_for_follow_up_chain(tool_name, tool_input, result)
            if next_tool:
                if self.debug_mode:
                    print(f"[DEBUG] 🔗 Identified follow-up chain tool: {next_tool['name']}")
                
                # Execute the follow-up tool
                follow_up_result = await self._handle_tool_call(next_tool)
                
                # Merge results for easier reference
                result["chained_tool"] = next_tool["name"]
                result["chained_result"] = follow_up_result

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
            
    def _check_for_follow_up_chain(self, tool_name: str, tool_input: Dict[str, Any], tool_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if a tool call result should trigger a follow-up tool.
        
        Args:
            tool_name: Name of the current tool
            tool_input: Input parameters for the current tool
            tool_result: Result of the current tool
            
        Returns:
            Next tool call information or None if no chaining needed
        """
        # Check for result-based chaining patterns
        
        # Pattern 1: After find_files, if one match, read it
        if tool_name == 'find_files' and 'matches' in tool_result and len(tool_result['matches']) == 1:
            filepath = tool_result['matches'][0].get('path')
            
            if filepath:
                return {
                    "name": "read_file",
                    "input": {"path": filepath},
                    "id": f"follow-{len(self.tool_call_history)}"
                }
                
        # Pattern 2: After successful parse_diff_suggestions, apply the changes
        elif tool_name == 'parse_diff_suggestions' and 'changes' in tool_result and tool_result.get('count', 0) > 0:
            # Need to know what file to apply to - not easily available from just the parse result
            # This is where a more sophisticated planner would be useful
            pass
            
        # Add more follow-up patterns as needed
                
        return None
    
    def _has_recent_read(self, filepath: str) -> bool:
        """
        Check if a file has been read recently in the tool call history.
        
        Args:
            filepath: Path to the file
            
        Returns:
            True if file has been read recently, False otherwise
        """
        # Look back through recent tool calls
        max_lookback = 10  # Only look at the last 10 tool calls
        recent_tools = self.tool_call_history[-max_lookback:] if len(self.tool_call_history) > max_lookback else self.tool_call_history
        
        for tool_call in reversed(recent_tools):  # Start with most recent
            if tool_call.get('name') == 'read_file':
                tool_input = tool_call.get('input', {})
                if isinstance(tool_input, dict) and tool_input.get('path') == filepath:
                    return True
                elif isinstance(tool_input, str) and tool_input.strip() == filepath:
                    return True
                    
        return False

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
        elif command == 'chains':  # NEW command to show chaining information
            return self._show_chaining_info()
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
    
    def _show_chaining_info(self):
        """
        Show information about tool chaining capabilities.
        
        Returns:
            Tool chaining information
        """
        chaining_info = "Tool Chaining Information:\n\n"
        
        chaining_info += "Automatic Tool Chains:\n"
        chaining_info += "1. Directory Navigation Chain\n"
        chaining_info += "   - When you set a working directory, automatically lists its contents\n\n"
        
        chaining_info += "2. File Modification Chain\n"
        chaining_info += "   - When modifying a file, automatically reads it first if not already read\n\n"
        
        chaining_info += "3. File Search and Read Chain\n"
        chaining_info += "   - When finding a single file matching a pattern, automatically reads it\n\n"
        
        chaining_info += "4. Code Analysis Chain\n"
        chaining_info += "   - When analyzing code in a file, automatically reads it first\n\n"
        
        return chaining_info
    
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
  /chains - Show information about tool chaining capabilities

Direct Code Commands:
  code:workdir:/path/to/dir - Set working directory
  code:read:path/to/file.py - Read a file
  code:read:file1.py,file2.py - Read multiple files
  code:find:directory - Find Python files in directory
  code:find:recursive:directory - Find Python files recursively
  code:list - Show loaded files
  code:generate:/path/to/file.py:prompt - Generate code with a prompt
  code:change:/path/to/file.py:prompt - Modify existing code with prompt

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

Tool Chaining:
  The assistant can now automatically chain multiple tools for complex operations:
  - "Modify code in file.py" will first read the file, then apply modifications
  - "Change directory to /path/to/dir" will set directory and then list contents
  - "Find and read config files" will search for files and read matching results

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
        
    def print_tool_status(self, tool_name: str, tool_input: Dict[str, Any]) -> None:
        """
        Print user-friendly information about the tool being used.
        This function is called whenever a tool is used, regardless of debug mode.
        
        Args:
            tool_name: Name of the tool being used
            tool_input: Tool input parameters
        """
        # Format tool name
        icon = "🔧"
        tool_display = f"Using tool: {tool_name}"
        
        # Add file information if available
        filepath = None
        if isinstance(tool_input, dict):
            if 'path' in tool_input:
                filepath = tool_input['path']
            elif 'filepath' in tool_input:
                filepath = tool_input['filepath']
                
        if filepath:
            tool_display += f" on '{filepath}'"
            
        # Add special handling for specific tools
        if tool_name == 'set_working_directory':
            if isinstance(tool_input, dict) and 'path' in tool_input:
                icon = "📁"
                tool_display = f"Changing directory to: {tool_input['path']}"
        elif tool_name == 'read_file':
            if filepath:
                icon = "📖"
                tool_display = f"Reading file: {filepath}"
        elif tool_name == 'write_file':
            if filepath:
                icon = "✏️"
                tool_display = f"Writing to file: {filepath}"
        elif tool_name == 'generate_code':
            if filepath:
                icon = "✨"
                tool_display = f"Generating code in: {filepath}"
        elif tool_name == 'modify_code':
            if filepath:
                icon = "🔄"
                tool_display = f"Modifying code in: {filepath}"
        elif tool_name == 'analyze_code':
            if filepath:
                icon = "🔍"
                tool_display = f"Analyzing code in: {filepath}"
        elif tool_name == 'list_directory':
            if isinstance(tool_input, dict) and 'path' in tool_input:
                icon = "📋"
                tool_display = f"Listing directory: {tool_input['path']}"
        elif tool_name == 'find_files':
            if isinstance(tool_input, dict) and 'path' in tool_input:
                pattern = tool_input.get('pattern', '*')
                icon = "🔎"
                tool_display = f"Finding files in {tool_input['path']} matching: {pattern}"
        
        # Print the formatted string
        print_status(icon, tool_display, 'cyan')