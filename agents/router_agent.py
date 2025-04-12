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
        """
        Determine whether the input is a command or chat message.
        
        Args:
            user_input: Raw user input
            
        Returns:
            Tuple of (route_type, command_data) where route_type is either 'command' or 'chat',
            and command_data contains command details if it's a command
        """
        # Direct routing for slash commands and explicit command syntax (without using Claude)
        if user_input.startswith('/'):
            command = user_input[1:].strip()
            return 'command', {'command_type': 'slash', 'command': command, 'args': []}
            
        elif user_input.lower().startswith('code:'):
            parts = user_input.split(':', 2)
            
            if len(parts) < 2:
                return 'command', {'command_type': 'invalid', 'error': 'Invalid command format'}
                
            subcommand = parts[1].lower()
            args = parts[2].split(':') if len(parts) > 2 else []
                
            return 'command', {'command_type': 'code', 'command': subcommand, 'args': args}
        
        # Use Claude for more complex command understanding
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

Instructions:
1. Analyze the user input and determine if it matches one of the available commands.
2. Return ONLY valid JSON in the following format:
   
   For command matches:
   {{
     "is_command": true,
     "command_type": "<command_type>",
     "command": "<specific_command>",
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

3. For commands, set the confidence level (0.0-1.0) based on how certain you are that the user intends to use a command.
4. For natural language that implies a command (e.g., "please open the file test.py" -> code:read:test.py), set command_type to "inferred".
5. Only include the original_query field for inferred commands, showing the portion of text that isn't part of the command.

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
        """
        Parse the router response to extract command information.
        
        Args:
            response_text: Response from Claude
            original_input: Original user input
            
        Returns:
            Command data or None if not a command
        """
        try:
            # Extract JSON from response (in case there's text around it)
            json_match = response_text.strip()
            
            # Sometimes Claude adds markdown code blocks
            if json_match.startswith('```json') and json_match.endswith('```'):
                json_match = json_match[7:-3].strip()
            elif json_match.startswith('```') and json_match.endswith('```'):
                json_match = json_match[3:-3].strip()
            
            data = json.loads(json_match)
            
            if not data.get('is_command', False):
                return None
                
            # If inferred command has high confidence, return it
            command_data = {
                'command_type': data.get('command_type', 'unknown'),
                'command': data.get('command', ''),
                'args': data.get('args', []),
                'confidence': data.get('confidence', 0.0)
            }
            
            # Add original query for inferred commands
            if data.get('command_type') == 'inferred' and 'original_query' in data:
                command_data['original_query'] = data['original_query']
                
            return command_data
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing router response: {str(e)}", file=sys.stderr)
            print(f"Response: {response_text}", file=sys.stderr)
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
