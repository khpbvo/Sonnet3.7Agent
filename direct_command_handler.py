"""
Direct command handler for ETMSonnet.
This module provides direct command processing that bypasses Claude's tool calling.
"""

import re
import os
import asyncio
from typing import Dict, List, Any, Optional

# Import terminal utilities
from utils.terminal_utils import print_status

class DirectCommandHandler:
    """
    Handles commands directly, bypassing Claude's tool calling when needed.
    """
    
    def __init__(self, file_manager, tool_handlers, conversation_manager):
        """
        Initialize the direct command handler.
        
        Args:
            file_manager: File manager instance
            tool_handlers: Dictionary of tool handlers
            conversation_manager: Conversation manager instance
        """
        self.file_manager = file_manager
        self.tool_handlers = tool_handlers
        self.conversation_manager = conversation_manager
        self.debug_mode = False
    
    def print_command_status(self, command_type: str, details: str = "") -> None:
        """
        Print user-friendly information about the direct command being processed.
        
        Args:
            command_type: Type of command being processed
            details: Additional details about the command
        """
        if command_type == "directory":
            print_status("📁", f"Processing directory command: {details}", "magenta")
        elif command_type == "list":
            print_status("📋", f"Processing list command: {details}", "magenta")
        elif command_type == "read":
            print_status("📖", f"Processing file read command: {details}", "magenta")
        elif command_type == "code":
            print_status("💻", f"Processing code command: {details}", "magenta")
        else:
            print_status("🔄", f"Processing direct command: {details}", "magenta")
    
    def set_debug_mode(self, debug_mode: bool):
        """Set debug mode"""
        self.debug_mode = debug_mode
    
    async def process_command(self, message: str) -> Optional[str]:
        """
        Process a command directly.
        
        Args:
            message: User message
            
        Returns:
            Response message if command was processed, None otherwise
        """
        # Check for slash commands - these are handled by the main loop now
        if message.startswith('/'):
            return None
            
        # Check for directory commands
        if await self._handle_directory_command(message):
            return "Directory command processed."
            
        # Check for file listing commands
        if await self._handle_list_command(message):
            return "File listing command processed."
            
        # Check for file read commands
        if await self._handle_read_command(message):
            return "File read command processed."
            
        # Check for code commands
        if await self._handle_code_command(message):
            return "Code command processed."
            
        # No direct command matched
        return None
        
    async def _handle_code_command(self, message: str) -> bool:
        """
        Handle code generation and modification commands.
        
        Args:
            message: User message
            
        Returns:
            True if command was processed, False otherwise
        """
        # Detect code command patterns
        code_patterns = [
            r'code:generate:(.+)',
            r'code:change:(.+)',
            r'generate\s+code\s+(?:for|in)\s+(.+)',
            r'modify\s+(?:the\s+)?code\s+(?:in|of)\s+(.+)'
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                # We found a code command, but we'll let the agent handle it
                # through regular tool calls rather than executing directly
                if self.debug_mode:
                    print(f"[DIRECT] Detected code command: {match.group(0)}")
                    print(f"[DIRECT] Letting the agent handle it via tools")
                return False  # Return False to let the agent handle it
                
        return False  # No code command matched

    async def _handle_directory_command(self, message: str) -> bool:
        """
        Handle directory change commands.
        
        Args:
            message: User message
            
        Returns:
            True if command was processed, False otherwise
        """
        # Detect directory change intent
        directory_patterns = [
            # Explicit commands
            r'code:workdir:(.+)',
            r'set\s+working\s+directory\s+to\s+(.+)',
            r'change\s+(?:the\s+)?(?:working\s+)?directory\s+to\s+(.+)',
            r'cd\s+(.+)',
            # More general patterns
            r'(?:use|switch\s+to)\s+(?:the\s+)?directory\s+(.+)',
            r'(?:make|set)\s+(.+)\s+(?:as|the)\s+(?:working|current)\s+directory',
            r'working\s+directory\s+(?:should\s+be|is)\s+(.+)',
            # Additional patterns to catch more variations
            r'set\s+(?:the\s+)?workingdir\s+to\s+(.+)',
            r'change\s+(?:the\s+)?workingdir\s+to\s+(.+)',
            r'(?:can\s+you|please|could\s+you)?\s+set\s+(?:the\s+)?(?:working\s+directory|workingdir)\s+to\s+(.+)',
            r'(?:can\s+you|please|could\s+you)?\s+change\s+(?:the\s+)?(?:working\s+directory|workingdir)\s+to\s+(.+)'
        ]
    
        path = None
        for pattern in directory_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                path = match.group(1).strip().strip('"\'')
                break
    
        if not path:
            return False
    
        # Print command status
        self.print_command_status("directory", f"Changing to {path}")
    
        if self.debug_mode:
            print(f"[DIRECT] Setting working directory to: {path}")
    
        # Get the handler
        handler = self.tool_handlers.get('set_working_directory')
        if not handler:
            print("[ERROR] No handler found for set_working_directory")
            return False
    
        # Execute the tool directly
        result = await handler.handle_tool_use({
            "name": "set_working_directory",
            "input": {"path": path}
        })
    
        # Print the result
        if "error" in result:
            print(f"[ERROR] {result['error']}")
            return False
    
        print(f"[SUCCESS] Working directory set to: {path}")
    
        # Show directory contents (condensed)
        await self._list_current_directory(condensed=True)
    
        return True

    async def _handle_list_command(self, message: str) -> bool:
        """
        Handle list directory commands and code:list commands.
        
        Args:
            message: User message
            
        Returns:
            True if command was processed, False otherwise
        """
        # Detect list intent
        list_patterns = [
            r'code:list',
            r'list\s+(?:the\s+)?(?:files|contents)\s+(?:in|of)?\s+(?:the\s+)?(?:directory|folder)?',
            r'show\s+(?:the\s+)?(?:files|contents)\s+(?:in|of)?\s+(?:the\s+)?(?:directory|folder)?',
            r'what\s+(?:files|contents)\s+(?:are|do\s+we\s+have)\s+(?:in|of)?\s+(?:the\s+)?(?:directory|folder)?',
            r'directory\s+(?:files|contents)',
            r'\bls\b',
            r'\bdir\b'
        ]
        
        # Special case for 'code:list'
        if message.strip() == 'code:list':
            # Print command status
            self.print_command_status("list", "Listing loaded files")
            
            if self.debug_mode:
                print("[DIRECT] Handling code:list command (list loaded files)")
            
            handler = self.tool_handlers.get('list_loaded_files')
            if not handler:
                print("[ERROR] No handler found for list_loaded_files")
                return False
                
            result = await handler.handle_tool_use({
                "name": "list_loaded_files",
                "input": {}
            })
            
            if "error" in result:
                print(f"[ERROR] {result['error']}")
                return False
                
            # Display loaded files
            if "files" in result and result["files"]:
                print("\nLoaded files:")
                for file_info in result["files"]:
                    print(f"- {file_info['path']} ({file_info.get('lines', '?')} lines)")
                
            if "summary" in result:
                print(f"\n{result['summary']}")
                
            return True
            
        # Regular directory listing
        is_list_command = False
        for pattern in list_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                is_list_command = True
                break
        
        if not is_list_command:
            return False
        
        # Extract path or use current directory
        path_match = re.search(r'(?:in|of)\s+(?:the\s+)?(?:directory|folder)?\s+([^\s,]+)', message, re.IGNORECASE)
        path = path_match.group(1).strip().strip('"\'') if path_match else self.file_manager.get_working_directory()
        
        # Print command status
        self.print_command_status("list", f"Listing directory {path}")
        
        if self.debug_mode:
            print(f"[DIRECT] Listing directory: {path}")
        
        # Get the handler
        handler = self.tool_handlers.get('list_directory')
        if not handler:
            print("[ERROR] No handler found for list_directory")
            return False
        
        # Execute the tool directly
        result = await handler.handle_tool_use({
            "name": "list_directory",
            "input": {"path": path}
        })
        
        # Print the result
        if "error" in result:
            print(f"[ERROR] {result['error']}")
            return False
        
        # Display in a more condensed format
        self._display_directory_contents(result, path)
        
        return True
        
    def _display_directory_contents(self, result, path):
        """Display directory contents in a condensed format."""
        print(f"Contents of {path}:")
        
        if "directories" in result and result["directories"]:
            print("\nFolders:")
            for d in result["directories"]:
                print(f"- {d['name']}/")
        
        if "files" in result and result["files"]:
            print("\nFiles:")
            # Display files with condensed information
            for f in result["files"]:
                size = f.get('size_bytes', 0)
                # Format size more neatly
                if size < 1024:
                    size_str = f"{size} bytes"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size/(1024*1024):.1f} MB"
                    
                print(f"- {f['name']} ({size_str})")
        
        print(f"\nTotal: {result.get('total_entries', 0)} items")
        
    async def _list_current_directory(self, condensed=False) -> None:
        """List the contents of the current working directory."""
        path = self.file_manager.get_working_directory()
        
        # Get the handler
        handler = self.tool_handlers.get('list_directory')
        if not handler:
            print("[ERROR] No handler found for list_directory")
            return
        
        # Execute the tool directly
        result = await handler.handle_tool_use({
            "name": "list_directory",
            "input": {"path": path}
        })
        
        # Print the result
        if "error" in result:
            print(f"[ERROR] {result['error']}")
            return
        
        # Display contents
        self._display_directory_contents(result, path)

    async def _handle_read_command(self, message: str) -> bool:
        """
        Handle file read commands.
        
        Args:
            message: User message
            
        Returns:
            True if command was processed, False otherwise
        """
        # Detect read intent
        read_patterns = [
            r'code:read:(.+)',
            r'read\s+(?:the\s+)?file\s+(?:called\s+)?([^\s,]+)',
            r'show\s+(?:the\s+)?(?:content|contents)\s+of\s+(?:file\s+)?([^\s,]+)',
            r'display\s+(?:the\s+)?(?:file|content)\s+(?:of\s+)?([^\s,]+)',
            r'open\s+(?:the\s+)?file\s+([^\s,]+)',
            r'cat\s+([^\s,]+)'
        ]
        
        filepath = None
        multiple_files = False
        
        # Check for explicit code:read: command
        if message.startswith('code:read:'):
            # Extract everything after code:read:
            command_param = message[len('code:read:'):].strip()
            
            # Check if it contains commas (multiple files)
            if ',' in command_param:
                filepaths = [f.strip() for f in command_param.split(',')]
                multiple_files = True
                # Print command status
                self.print_command_status("read", f"Reading multiple files: {command_param}")
            else:
                filepath = command_param
                # Print command status
                self.print_command_status("read", f"Reading file: {command_param}")
        else:
            # Try the regular patterns
            for pattern in read_patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    filepath = match.group(1).strip().strip('"\'')
                    # Print command status
                    self.print_command_status("read", f"Reading file: {filepath}")
                    break
        
        if not filepath and not multiple_files:
            return False
        
        # Handle multiple files case
        if multiple_files:
            for file_path in filepaths:
                await self._read_single_file(file_path)
            return True
        
        # Handle single file case
        return await self._read_single_file(filepath)
    
    async def _read_single_file(self, filepath: str) -> bool:
        """Read a single file and display its info."""
        if self.debug_mode:
            print(f"[DIRECT] Reading file: {filepath}")
        
        # Print command status
        self.print_command_status("read", f"Reading file: {filepath}")
        
        # Get the handler
        handler = self.tool_handlers.get('read_file')
        if not handler:
            print("[ERROR] No handler found for read_file")
            return False
        
        # Execute the tool directly
        result = await handler.handle_tool_use({
            "name": "read_file",
            "input": {"path": filepath}
        })
        
        # Print the result - only print basic info, not the full file content
        if "error" in result:
            print(f"[ERROR] {result['error']}")
            return False
        
        print(f"Successfully read file: {filepath}")
        # Show a preview in non-debug mode as well, but keep it shorter
        content_preview = result['content'][:200] + '...' if len(result['content']) > 200 else result['content']
        print(f"\nPreview:\n```\n{content_preview}\n```\n")
        
        return True