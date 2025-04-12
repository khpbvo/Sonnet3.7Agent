"""
Manages conversation history and context for Claude interactions.
"""

import os
from typing import List, Dict, Optional, Any, Tuple
import tiktoken
import asyncio


class ConversationManager:
    """
    Manages the conversation history with optimizations for large context windows.
    Implements token tracking and truncation strategies.
    """
    def __init__(self, max_tokens: int = 200000):
        """
        Initialize the conversation manager.
        
        Args:
            max_tokens: Maximum number of tokens to maintain in context
        """
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, str]] = []
        self.token_count = 0
        self.summary: Optional[str] = None
        self.loaded_files: Dict[str, str] = {}  # Cache for loaded files
        
        # Initialize the tokenizer for Claude
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Claude uses cl100k_base
        except:
            # Fallback to simpler estimation if tiktoken not available
            self.tokenizer = None
            print("Warning: tiktoken not available, using approximate token counting")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history and track token usage.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        # Format content according to Messages API format
        message = {"role": role, "content": content}
        est_tokens = self._count_tokens(content)
        
        self.messages.append(message)
        self.token_count += est_tokens
        
        # If we're near the token limit, optimize the history
        if self.token_count > self.max_tokens * 0.9:
            self._optimize_history()
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in the conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.messages
    
    def add_loaded_file(self, filepath: str, content: str) -> None:
        """
        Add a loaded file to the internal cache.
        
        Args:
            filepath: Path to the file
            content: File content
        """
        self.loaded_files[filepath] = content
    
    def get_loaded_file(self, filepath: str) -> Optional[str]:
        """
        Get a loaded file from the cache.
        
        Args:
            filepath: Path to the file
            
        Returns:
            File content or None if not cached
        """
        return self.loaded_files.get(filepath)
    
    def get_loaded_files_info(self) -> str:
        """
        Get a summary of all loaded files.
        
        Returns:
            String summarizing loaded files
        """
        if not self.loaded_files:
            return "No files loaded."
        
        info = "Loaded files:\n"
        for filepath in self.loaded_files:
            file_lines = self.loaded_files[filepath].count('\n') + 1
            file_size = len(self.loaded_files[filepath])
            info += f"- {filepath} ({file_lines} lines, {file_size} bytes)\n"
        return info
    
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
            
        if self.tokenizer:
            # Use tiktoken for accurate counting
            return len(self.tokenizer.encode(text))
        else:
            # Approximate tokens (4 chars per token is a common approximation)
            return len(text) // 4 + 1
    
    def _optimize_history(self) -> None:
        """
        Optimize the conversation history to stay within token limits.
        Uses a strategy of summarizing older messages and preserving recent context.
        """
        if len(self.messages) <= 4:
            # Not enough messages to optimize
            return
        
        # Calculate how much we need to reduce
        target_reduction = int(self.token_count - (self.max_tokens * 0.7))
        if target_reduction <= 0:
            return
        
        # Always preserve the last 3 exchanges (6 messages)
        preserve_count = min(6, len(self.messages))
        candidates_for_summary = self.messages[:-preserve_count]
        
        if not candidates_for_summary:
            return
        
        # Create or update the summary
        self._update_summary(candidates_for_summary)
        
        # Calculate token counts
        removed_tokens = sum(self._count_tokens(msg["content"]) for msg in candidates_for_summary)
        summary_tokens = self._count_tokens(self.summary if self.summary else "")
        
        # Only summarize if it offers significant reduction
        if removed_tokens > summary_tokens:
            self.messages = [{"role": "system", "content": self.summary}] + self.messages[-preserve_count:]
            self.token_count = summary_tokens + sum(self._count_tokens(msg["content"]) for msg in self.messages[-preserve_count:])
    
    def _update_summary(self, messages_to_summarize: List[Dict[str, str]]) -> None:
        """
        Create a summary of older messages.
        
        Args:
            messages_to_summarize: List of messages to summarize
        """
        if not messages_to_summarize:
            return
        
        # Simple summary approach - keep most important points
        summary_parts = []
        
        if self.summary:
            summary_parts.append("PREVIOUS CONVERSATION HISTORY: " + self.summary)
        
        summary_parts.append("SUMMARY OF OLDER MESSAGES:")
        
        for i, msg in enumerate(messages_to_summarize):
            if i >= 10:  # Limit to max 10 messages in summary
                summary_parts.append(f"... plus {len(messages_to_summarize) - 10} older messages.")
                break
            
            # Truncate content if needed
            content = msg["content"]
            if len(content) > 500:
                content = content[:250] + "..." + content[-250:]
            
            summary_parts.append(f"{msg['role'].upper()}: {content[:100]}...")
        
        self.summary = "\n".join(summary_parts)
    
    def clear(self) -> None:
        """Clear the conversation history, but keep loaded files."""
        self.messages = []
        self.token_count = 0
        self.summary = None
        
    def get_token_usage(self) -> int:
        """
        Get the current token usage.
        
        Returns:
            Current token count
        """
        return self.token_count
    
    def get_token_percentage(self) -> float:
        """
        Get the percentage of tokens used relative to the maximum.
        
        Returns:
            Percentage of token limit used
        """
        return (self.token_count / self.max_tokens) * 100
        
    async def extract_system_message(self) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Extract the system message from the conversation history.
        
        Returns:
            Tuple of (system_message, regular_messages)
        """
        system_message = None
        regular_messages = []
        
        for msg in self.messages:
            if msg["role"] == "system":
                # Keep the latest system message
                system_message = msg["content"]
            else:
                # Ensure message format is correct
                regular_messages.append(msg)
                
        return system_message, regular_messages

    def format_messages_for_api(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format messages for the Anthropic API.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted messages for the API
        """
        formatted_messages = []
        for msg in messages:
            content = msg["content"]
            # Format content as an array with text type if it's a string
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            
            formatted_messages.append({
                "role": msg["role"],
                "content": content
            })
        
        return formatted_messages
