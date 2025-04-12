"""
Manages file operations including reading, finding, and analyzing files.
"""

import os
import re
import difflib
import asyncio
from typing import List, Dict, Optional, Tuple, Any, Set


class FileManager:
    """
    Handles file operations such as reading, writing, and finding files.
    Coordinates with the conversation manager to track loaded files.
    """
    
    def __init__(self, conversation_manager):
        """
        Initialize the file manager.
        
        Args:
            conversation_manager: The conversation manager to use
        """
        self.conversation_manager = conversation_manager
        self.working_dir = os.getcwd()
    
    async def read_file(self, filepath: str) -> str:
        """
        Read a file and add it to the conversation manager's cache.
        
        Args:
            filepath: Path to the file to read
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: For other reading errors
        """
        filepath = self._get_absolute_path(filepath)
        
        # Check if the file is already loaded
        cached_content = self.conversation_manager.get_loaded_file(filepath)
        if cached_content is not None:
            return cached_content
        
        try:
            # Try primary encoding first
            file_obj = await asyncio.to_thread(lambda: open(filepath, 'r', encoding='utf-8'))
            try:
                content = await asyncio.to_thread(file_obj.read)
                # Cache the file
                self.conversation_manager.add_loaded_file(filepath, content)
                return content
            finally:
                await asyncio.to_thread(file_obj.close)
                
        except UnicodeDecodeError:
            # Try fallback encoding
            try:
                file_obj = await asyncio.to_thread(lambda: open(filepath, 'r', encoding='latin-1'))
                try:
                    content = await asyncio.to_thread(file_obj.read)
                    
                    # Cache the file
                    self.conversation_manager.add_loaded_file(filepath, content)
                    return content
                finally:
                    await asyncio.to_thread(file_obj.close)
                    
            except Exception as e:
                raise IOError(f"Cannot read file: {str(e)}")
    
    async def write_file(self, filepath: str, content: str) -> bool:
        """
        Write content to a file.
        
        Args:
            filepath: Path to the file to write
            content: Content to write
            
        Returns:
            True if write was successful, False otherwise
        """
        filepath = self._get_absolute_path(filepath)
        
        try:
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(filepath)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
            # Write the file
            async with asyncio.to_thread(open, filepath, 'w', encoding='utf-8') as file:
                await asyncio.to_thread(file.write, content)
                
            # Update the cache
            self.conversation_manager.add_loaded_file(filepath, content)
            return True
            
        except Exception as e:
            print(f"Error writing file: {str(e)}")
            return False
    
    async def find_python_files(self, directory: str = '.', recursive: bool = False) -> List[str]:
        """
        Find Python files in a directory.
        
        Args:
            directory: Directory to search in
            recursive: Whether to search recursively
            
        Returns:
            List of Python file paths
        """
        directory = self._get_absolute_path(directory)
        
        python_files = []
        
        # Make sure directory exists
        if not os.path.exists(directory):
            return []
            
        try:
            if recursive:
                # Use os.walk for recursive search
                for root, _, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.py'):
                            python_files.append(os.path.join(root, file))
            else:
                # Only search in the top-level directory
                for file in os.listdir(directory):
                    filepath = os.path.join(directory, file)
                    if file.endswith('.py') and os.path.isfile(filepath):
                        python_files.append(filepath)
                        
            return python_files
            
        except Exception as e:
            print(f"Error finding Python files: {str(e)}")
            return []
    
    def set_working_directory(self, directory: str) -> str:
        """
        Set the working directory for file operations.
        
        Args:
            directory: Directory to use as working directory
            
        Returns:
            Message indicating success or failure
        """
        if not os.path.exists(directory):
            return f"Error: Directory '{directory}' does not exist"
            
        if not os.path.isdir(directory):
            return f"Error: '{directory}' is not a directory"
            
        self.working_dir = os.path.abspath(directory)
        return f"Working directory set to: {self.working_dir}"
    
    def _get_absolute_path(self, filepath: str) -> str:
        """
        Convert a relative path to an absolute path.
        
        Args:
            filepath: Relative or absolute path
            
        Returns:
            Absolute path
        """
        if os.path.isabs(filepath):
            return filepath
        else:
            return os.path.join(self.working_dir, filepath)
    
    def get_working_directory(self) -> str:
        """
        Get the current working directory.
        
        Returns:
            Current working directory
        """
        return self.working_dir
    
    async def generate_diff(
        self, 
        original: str, 
        modified: str, 
        filepath: str = ""
    ) -> str:
        """
        Generate a unified diff between two versions of text.
        
        Args:
            original: Original text
            modified: Modified text
            filepath: Filepath for the diff header
            
        Returns:
            Unified diff as a string
        """
        if not filepath:
            filepath = "code.py"
            
        # If there's no difference, return empty string
        if original == modified:
            return ""
            
        # Split the code into lines
        original_lines = original.splitlines(True)
        modified_lines = modified.splitlines(True)
        
        # Generate the diff
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"original/{filepath}",
            tofile=f"modified/{filepath}",
            n=3  # Context lines
        )
        
        return ''.join(diff)
