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
    
    def __init__(self, file_manager, tool_handlers, conversation_manager, tool_chain_manager=None):
        """
        Initialize the direct command handler.
        
        Args:
            file_manager: File manager instance
            tool_handlers: Dictionary of tool handlers
            conversation_manager: Conversation manager instance
            tool_chain_manager: Optional tool chain manager instance
        """
        self.file_manager = file_manager
        self.tool_handlers = tool_handlers
        self.conversation_manager = conversation_manager
        self.tool_chain_manager = tool_chain_manager
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
        elif command_type == "chain":
            print_status("🔗", f"Processing tool chain: {details}", "magenta")
        else:
            print_status("🔄", f"Processing direct command: {details}", "magenta")
    
    def set_debug_mode(self, debug_mode: bool):
        """Set debug mode"""
        self.debug_mode = debug_mode
    
    def set_tool_chain_manager(self, tool_chain_manager):
        """Set the tool chain manager"""
        self.tool_chain_manager = tool_chain_manager
    
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
        
        if self.debug_mode:
            print(f"[DIRECT] Processing command: {message}")
        
        # NEW: Try to process with tool chain manager if available
        if self.tool_chain_manager:
            chain_result = await self.tool_chain_manager.identify_and_execute_chain(message)
            if chain_result:
                self.print_command_status("chain", f"{chain_result.get('chain_type', 'Unknown chain')}")
                return f"Tool chain executed: {chain_result.get('chain_type')}"
         
        # Check for explicit commands first - highest priority
        if message.startswith('code:'):
            if message.startswith('code:read:'):
                if await self._handle_read_command(message):
                    return "File read command processed."
            elif message.startswith('code:workdir:'):
                if await self._handle_directory_command(message):
                    return "Directory command processed."
            elif message.startswith('code:list'):
                if await self._handle_list_command(message):
                    return "File listing command processed."
            # Let other code: commands continue to the standard parsing
        
        # Handle compound commands like "change directory to X and read files Y"  
        if re.search(r'(?:change|set)\s+(?:the\s+)?(?:working\s+)?directory.*?(?:and|then)\s+(?:read|show|display)', message, re.IGNORECASE):
            if self.debug_mode:
                print(f"[DIRECT] Detected compound directory+files command")
            
            # Use the updated directory command handler which now properly handles compound commands
            if await self._handle_directory_command(message):
                return "Compound directory and file command processed."
                
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
        # First, check if message contains a compound command with "and" or "then"
        compound_cmd = False
        base_message = message
        files_to_read = []
        
        # Extract files to read from compound commands
        compound_match = re.search(r'(?:and|then)\s+read\s+(?:the\s+)?(?:files?\s+)?(.+?)(?:$|;)', message, re.IGNORECASE)
        if compound_match:
            compound_cmd = True
            files_part = compound_match.group(1).strip()
            
            # Get the files to read (handling comma-separated lists)
            if ',' in files_part:
                files_to_read = [f.strip() for f in files_part.split(',')]
            else:
                # Try to intelligently split the file list
                words = files_part.split()
                if len(words) == 1:
                    # Just one word, assume it's a filename
                    files_to_read = [words[0]]
                elif "and" in files_part.lower():
                    # Try to handle "file1.py and file2.py"
                    file_matches = re.findall(r'(\S+\.\w+)', files_part)
                    if file_matches:
                        files_to_read = file_matches
                    else:
                        # Fallback to files that have extensions
                        files_to_read = [word for word in words if '.' in word]
                else:
                    # Multiple words, assume files have extensions
                    files_to_read = [word for word in words if '.' in word]
            
            # Remove any "and" words that might have been caught
            files_to_read = [f for f in files_to_read if f.lower() != 'and']
            
            if self.debug_mode:
                print(f"[DIRECT] Files to read after parsing: {files_to_read}")
            
            # Trim the message to only include the directory part
            base_message = message[:compound_match.start()].strip()
            
            if self.debug_mode:
                print(f"[DIRECT] Detected compound command")
                print(f"[DIRECT] Base message: {base_message}")
                print(f"[DIRECT] Files to read: {files_to_read}")
        
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
            match = re.search(pattern, base_message, re.IGNORECASE)
            if match:
                path = match.group(1).strip().strip('"\'')
                
                # Clean up the path - remove any trailing "and" or "then" phrases
                # that might have been included in the match
                and_then_match = re.search(r'^(.*?)(?:\s+(?:and|then)\s+.*)?$', path, re.IGNORECASE)
                if and_then_match:
                    path = and_then_match.group(1).strip()
                
                if self.debug_mode:
                    print(f"[DIRECT] Extracted directory path: {path}")
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
    
        # Always chain with list_directory to show directory contents
        list_handler = self.tool_handlers.get('list_directory')
        if list_handler:
            list_result = await list_handler.handle_tool_use({
                "name": "list_directory",
                "input": {"path": path}
            })
            
            # Display directory contents in a condensed format
            self._display_directory_contents(list_result, path)
        else:
            # Fallback to the old method if the list_directory handler isn't available
            await self._list_current_directory(condensed=True)
            
        # If there are files to read after directory change, read them
        if files_to_read:
            if self.debug_mode:
                print(f"[DIRECT] Chaining file reading after directory change: {files_to_read}")
            
            for file_path in files_to_read:
                await self._read_single_file(file_path)
    
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
            r'read\s+(?:the\s+)?(?:python\s+)?files?\s+(?:called\s+)?([^\s,]+(?:\s*,\s*[^\s,]+)*)',
            r'show\s+(?:the\s+)?(?:content|contents)\s+of\s+(?:file\s+)?([^\s,]+(?:\s*,\s*[^\s,]+)*)',
            r'display\s+(?:the\s+)?(?:file|content)\s+(?:of\s+)?([^\s,]+(?:\s*,\s*[^\s,]+)*)',
            r'open\s+(?:the\s+)?files?\s+([^\s,]+(?:\s*,\s*[^\s,]+)*)',
            r'cat\s+([^\s,]+(?:\s*,\s*[^\s,]+)*)'
        ]
        
        filepath = None
        multiple_files = False
        filepaths = []
        
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
                    matched_paths = match.group(1).strip().strip('"\'')
                    
                    # Check if it's a comma-separated list
                    if ',' in matched_paths:
                        filepaths = [f.strip() for f in matched_paths.split(',')]
                        multiple_files = True
                        # Print command status
                        self.print_command_status("read", f"Reading multiple files: {matched_paths}")
                    else:
                        filepath = matched_paths
                        # Print command status
                        self.print_command_status("read", f"Reading file: {filepath}")
                    break
        
        if not filepath and not multiple_files and not filepaths:
            return False
        
        # Handle multiple files case
        if multiple_files or filepaths:
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
        
        # NEW: Check if we should automatically analyze Python files
        if filepath.endswith('.py'):
            analyze_handler = self.tool_handlers.get('analyze_code')
            if analyze_handler:
                print_status("🔗", f"Auto-analyzing Python file: {filepath}", "magenta")
                
                analyze_result = await analyze_handler.handle_tool_use({
                    "name": "analyze_code",
                    "input": {
                        "filepath": filepath,
                        "analysis_type": "basic"
                    }
                })
                
                if "error" not in analyze_result:
                    print(f"\nBasic code analysis:")
                    print(f"- Lines: {analyze_result.get('line_count', '?')}")
                    print(f"- Size: {analyze_result.get('size_bytes', '?')} bytes")
                
        return True