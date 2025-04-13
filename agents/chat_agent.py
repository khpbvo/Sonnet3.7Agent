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
        # First check if message is a slash command
        if message.startswith('/'):
            command = message[1:].strip()
            result = await self._handle_slash_command(command)
            if stream_callback and result:
                for char in result:
                    stream_callback(char)
            return result
        
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
                print(f"[DEBUG] - System message length: {len(system_message if system_message else '')}")
                print(f"[DEBUG] - Message history: {len(formatted_messages)} messages")
                print(f"[DEBUG] - Tools: {len(formatted_tools) if formatted_tools else 0} tools")
            
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
                    # Track if we need to handle any tool calls
                    tool_calls = []
                    current_tool_call = None
                    
                    for event in stream:
                        # Handle different event types
                        if hasattr(event, 'type'):
                            # Debug each event
                            if self.debug_mode:
                                print(f"[DEBUG] Event type: {event.type}")
                            
                            # Content block events - regular text responses
                            if event.type == "content_block_delta":
                                if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                                    chunk_text = event.delta.text
                                    if chunk_text:
                                        complete_response += chunk_text
                                        stream_callback(chunk_text)
                            
                            # Tool call events
                            elif event.type == "tool_call_start":
                                if self.debug_mode:
                                    print(f"\n[DEBUG] 🔧 Tool call started: {event.tool_call.name}")
                                
                                current_tool_call = {
                                    "name": event.tool_call.name,
                                    "input": "",
                                    "id": event.tool_call.id
                                }
                            elif event.type == "tool_call_delta":
                                if current_tool_call and hasattr(event.delta, 'input'):
                                    current_tool_call["input"] += event.delta.input
                            elif event.type == "tool_call_end":
                                if current_tool_call:
                                    # Parse the JSON input
                                    try:
                                        input_data = current_tool_call["input"]
                                        if isinstance(input_data, str):
                                            try:
                                                input_data = json.loads(input_data)
                                                current_tool_call["input"] = input_data
                                            except:
                                                # If it's not valid JSON, keep it as a string
                                                pass
                                        
                                        if self.debug_mode:
                                            print(f"[DEBUG] 🔧 Tool call completed: {current_tool_call['name']}")
                                            print(f"[DEBUG] 📝 Tool parameters: {json.dumps(current_tool_call['input'], indent=2)}")
                                    except Exception as e:
                                        if self.debug_mode:
                                            print(f"[DEBUG] ⚠️ Error parsing tool input: {str(e)}")
                                        
                                    tool_calls.append(current_tool_call)
                                    current_tool_call = None
                
                # Handle tool calls after streaming is complete
                if tool_calls:
                    if self.debug_mode:
                        print(f"\n[DEBUG] 🧰 Processing {len(tool_calls)} tool calls")
                    
                    # Process each tool call
                    tool_results = []
                    for tool_call in tool_calls:
                        self.tool_call_history.append(tool_call)
                        result = await self._handle_tool_call(tool_call)
                        tool_results.append({
                            "tool_call_id": tool_call.get("id"),
                            "output": json.dumps(result)
                        })
                        
                        if self.debug_mode:
                            print(f"[DEBUG] 📊 Tool result: {json.dumps(result, indent=2)}")
                    
                    # Call Claude again with the tool results
                    if self.debug_mode:
                        print("[DEBUG] 🔄 Sending tool results back to Claude")
                    
                    additional_response = await self._send_tool_results(tool_results, formatted_messages)
                    if additional_response:
                        stream_callback("\n\n")
                        for chunk in additional_response.splitlines():
                            stream_callback(chunk + "\n")
                        complete_response += "\n\n" + additional_response
            else:
                # Non-streaming response
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_response_tokens,
                    system=system_message,
                    messages=formatted_messages,
                    tools=formatted_tools,
                    thinking=thinking
                )
                
                # Extract text from the response
                text_content = self._extract_text_from_response(response)
                complete_response = text_content
                
                # Handle tool calls if any
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    if self.debug_mode:
                        print(f"\n[DEBUG] 🧰 Processing {len(response.tool_calls)} tool calls")
                    
                    # Process each tool call
                    tool_results = []
                    for tool_call in response.tool_calls:
                        # Parse the input if it's a string
                        input_data = tool_call.input
                        if isinstance(input_data, str):
                            try:
                                input_data = json.loads(input_data)
                            except:
                                pass
                        
                        tool_call_info = {
                            "name": tool_call.name,
                            "input": input_data,
                            "id": tool_call.id
                        }
                        
                        self.tool_call_history.append(tool_call_info)
                        
                        if self.debug_mode:
                            print(f"[DEBUG] 🔧 Tool call: {tool_call.name}")
                            print(f"[DEBUG] 📝 Tool parameters: {json.dumps(input_data, indent=2)}")
                        
                        result = await self._handle_tool_call(tool_call_info)
                        
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(result)
                        })
                        
                        if self.debug_mode:
                            print(f"[DEBUG] 📊 Tool result: {json.dumps(result, indent=2)}")
                    
                    # Call Claude again with the tool results
                    if tool_results:
                        if self.debug_mode:
                            print("[DEBUG] 🔄 Sending tool results back to Claude")
                        
                        additional_response = await self._send_tool_results(tool_results, formatted_messages)
                        if additional_response:
                            complete_response += "\n\n" + additional_response
            
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
    
    async def _send_tool_results(self, tool_results, previous_messages):
        """
        Send tool results back to Claude for further processing.
        
        Args:
            tool_results: List of tool results
            previous_messages: Previously formatted messages
            
        Returns:
            Claude's response after processing tool results
        """
        try:
            system_message, _ = await self.conversation_manager.extract_system_message()
            
            if self.debug_mode:
                print(f"[DEBUG] Sending {len(tool_results)} tool results to Claude")
            
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_response_tokens,
                system=system_message,
                messages=previous_messages,
                tool_results=tool_results
            )
            
            return self._extract_text_from_response(response)
            
        except Exception as e:
            print(f"Error sending tool results: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return f"Error processing tool results: {str(e)}"
    
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
            
            result = await handler.handle_tool_use({
                "name": tool_name,
                "input": tool_input
            })
            
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
                    required = "required" if 'required' in tool.input_schema and param_name in tool.input_schema['required'] else "optional"
                    default = f" (default: {param_info.get('default')})" if 'default' in param_info else ""
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