"""
Chat agent that handles interactions with Claude API.
"""

import os
import sys
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
import anthropic
from anthropic.types import ContentBlock, ContentBlockDeltaEvent, TextDelta, MessageParam, MessageStreamEvent

from config import Config


class ChatAgent:
    """
    Chat agent that interacts with Claude API.
    Handles message sending, receiving, and streaming.
    """
    
    def __init__(self, api_key: str, config: Config, conversation_manager):
        """
        Initialize the chat agent.
        
        Args:
            api_key: Anthropic API key
            config: Application configuration
            conversation_manager: Manager for conversation history
        """
        self.client = anthropic.AsyncClient(api_key=api_key)
        self.config = config
        self.conversation_manager = conversation_manager
    
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
            system_message, messages = await self.conversation_manager.extract_system_message()
            
            # Prepare thinking parameters if enabled
            thinking_params = {"type": "enabled", "budget_tokens": 16000} if thinking_enabled else None
            
            complete_response = ""
            
            # Stream the response if needed
            if stream_callback:
                async with self.client.messages.stream(
                    model=self.config.model,
                    max_tokens=self.config.max_response_tokens,
                    system=system_message,
                    messages=messages,
                    thinking=thinking_params
                ) as stream:
                    async for event in stream:
                        if isinstance(event, ContentBlockDeltaEvent) and event.delta and isinstance(event.delta, TextDelta):
                            chunk_text = event.delta.text
                            if chunk_text:
                                complete_response += chunk_text
                                stream_callback(chunk_text)
            else:
                # Non-streaming response
                response = await self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_response_tokens,
                    system=system_message,
                    messages=messages,
                    thinking=thinking_params
                )
                
                # Extract text from the response
                complete_response = self._extract_text_from_response(response)
            
            # Add the complete response to conversation history
            if complete_response:
                self.conversation_manager.add_message("assistant", complete_response)
            
            return complete_response if not stream_callback else None
            
        except Exception as e:
            error_msg = f"Error in send_message: {str(e)}"
            print(error_msg, file=sys.stderr)
            return f"Error: {str(e)}"
    
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
                if hasattr(content_block, 'text') and content_block.text:
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
