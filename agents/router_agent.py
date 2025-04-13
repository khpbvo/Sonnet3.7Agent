"""
Router agent for interpreting user input and routing to appropriate commands.
"""

import os
import sys
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
import anthropic
import json

from config import Config


class RouterAgent:
    """
    Router agent that interprets user input and determines whether it's a command or chat message.
    Uses Claude API for more sophisticated command understanding.
    """
    
    def __init__(self, api_key: str, config: Config):
        """
        Initialize the router agent.
        
        Args:
            api_key: Anthropic API key
            config: Application configuration
        """
        # Create the client with the API key directly
        # Debug to make sure we have an API key
        if not api_key:
            print("Warning: No API key provided to RouterAgent")
            
        self.client = anthropic.Anthropic(api_key=api_key)
        self.config = config
        self.available_commands = self._get_available_commands()
    
    async def route_input(self, user_input: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        # Direct routing for slash commands and explicit command syntax
        if user_input.startswith('/'):
            command = user_input[1:].strip()
            return 'command', {'command_type': 'slash', 'command': command, 'args': []}
        
        elif user_input.lower().startswith('code:'):
            parts = user_input.split(':', 2)
        
            if len(parts) < 2:
                return 'command', {'command_type': 'invalid', 'error': 'Invalid command format'}
            
            subcommand = parts[1].lower()
            args = parts[2].split(':') if len(parts) > 2 else []
            
            # Debug to confirm direct command processing
            print(f"Processing direct command - type: code, command: {subcommand}, args: {args}")
            
            return 'command', {'command_type': 'code', 'command': subcommand, 'args': args}
    
        # Only use Claude for actual natural language, not direct commands
        router_prompt = self._create_router_prompt(user_input)
        
        try:
            # Make API call to Claude to determine if input is a command
            response = self.client.messages.create(
                model=self.config.router_model,
                max_tokens=self.config.router_max_tokens,
                system=router_prompt,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": user_input}
                        ]
                    }
                ]
            )
            
            # Parse the response
            response_text = self._extract_text_from_response(response)
            command_data = self._parse_router_response(response_text, user_input)
            
            if command_data and 'command_type' in command_data:
                return 'command', command_data
            else:
                return 'chat', None
                
        except Exception as e:
            print(f"Router error: {str(e)}", file=sys.stderr)
            # Fall back to chat if there's an error
            return 'chat', None
    
    def _create_router_prompt(self, user_input: str) -> str:
        """
        Create a prompt for the router agent.
    
        Args:
            user_input: Raw user input
        
        Returns:
            Router prompt
        """
        commands_description = "\n".join([f"- {cmd['name']}: {cmd['description']}" for cmd in self.available_commands])
    
        return f"""You are a command router for an AI coding assistant. Your job is to determine if user input should be treated as a command or regular chat message.

Available Commands:
{commands_description}

IMPORTANT: There are ONLY THREE possible command types:
1. "slash" - Commands that start with / (e.g., /help, /exit)
2. "code" - Commands that explicitly start with code: (e.g., code:read:file.py)
3. "inferred" - Natural language that implies a command (e.g., "please read file.py" → code:read:file.py)

Instructions:
1. Analyze the user input and determine if it matches one of the available commands.
2. Return ONLY valid JSON in the following format:
   
   For slash commands:
   {{
     "is_command": true,
     "command_type": "slash",
     "command": "<command_without_slash>",
     "args": [],
     "confidence": <0.0-1.0>
   }}
   
   For explicit code commands:
   {{
     "is_command": true,
     "command_type": "code",
     "command": "<subcommand>",
     "args": ["<arg1>", "<arg2>", ...],
     "confidence": <0.0-1.0>
   }}
   
   For natural language that implies a command:
   {{
     "is_command": true,
     "command_type": "inferred",
     "command": "<specific_command>",
     "args": ["<arg1>", "<arg2>", ...],
     "confidence": <0.0-1.0>,
     "original_query": "<remaining_user_query>"
   }}
   
   For regular chat messages:
   {{
     "is_command": false
   }}

Examples:
1. "/help" → {{"is_command": true, "command_type": "slash", "command": "help", "args": [], "confidence": 1.0}}
2. "code:read:main.py" → {{"is_command": true, "command_type": "code", "command": "read", "args": ["main.py"], "confidence": 1.0}}
3. "Please read the file app.py" → {{"is_command": true, "command_type": "inferred", "command": "read", "args": ["app.py"], "confidence": 0.9, "original_query": ""}}
4. "How do I optimize this code?" → {{"is_command": false}}

IMPORTANT: Only use the three command types listed above: "slash", "code", or "inferred".
Do NOT create other command types like "explicit", "direct", etc.

IMPORTANT: ONLY return JSON - no explanations, markdown, or other text.
"""
    
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
    
    def _parse_router_response(self, response_text: str, original_input: str) -> Optional[Dict[str, Any]]:
        try:
            response_json = json.loads(response_text)
        
            if not response_json.get('is_command', False):
                return None
            
            command_type = response_json.get('command_type', '')
            command = response_json.get('command', '')
            args = response_json.get('args', [])
            confidence = response_json.get('confidence', 0.0)
        
            # For inferred commands, DON'T add the code: prefix here
            # We'll handle that in process_command
            if command_type == 'inferred':
                original_query = response_json.get('original_query', '')
                return {
                    'command_type': 'inferred',
                    'command': command,  # Don't modify the command here
                    'args': args,
                    'confidence': confidence,
                    'original_query': original_query
                }
        
            return {
                'command_type': command_type,
                'command': command,
                'args': args,
                'confidence': confidence
            }
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing router response: {str(e)}")
            return None
    
    def _get_available_commands(self) -> List[Dict[str, str]]:
        """
        Define available commands for the router.
        
        Returns:
            List of command definitions
        """
        return [
            {"name": "help", "description": "Show help information"},
            {"name": "exit", "description": "Exit the program"},
            {"name": "clear", "description": "Clear the conversation history"},
            {"name": "status", "description": "Show token usage and session information"},
            
            {"name": "code:workdir", "description": "Set working directory - code:workdir:/path/to/dir"},
            {"name": "code:read", "description": "Read a file - code:read:path/to/file.py"},
            {"name": "code:find", "description": "Find Python files - code:find:directory or code:find:recursive:directory"},
            {"name": "code:list", "description": "List loaded files - code:list"},
            {"name": "code:diff", "description": "Show diff for changes - code:diff:path/to/file.py"},
            {"name": "code:apply", "description": "Apply changes to file - code:apply:path/to/file.py"},
            
            {"name": "code:analyze", "description": "Analyze Python file - code:analyze:path/to/file.py"},
            {"name": "code:structure", "description": "Show file structure - code:structure:path/to/file.py"},
            {"name": "code:pylint", "description": "Run Pylint analysis - code:pylint:path/to/file.py"},
            {"name": "code:fullanalysis", "description": "Run full analysis - code:fullanalysis:path/to/file.py"},
            
            {"name": "code:generate", "description": "Generate code with prompt - code:generate:path/to/file.py:prompt"},
            {"name": "code:change", "description": "Change existing code - code:change:path/to/file.py:prompt"}
        ]
