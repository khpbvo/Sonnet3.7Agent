"""
Tools for file-related operations using Claude's tool functionality.
"""

import os
import sys
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import difflib
import re

#from anthropic import Tool, ToolUseBlock
from anthropic import Tool, ToolUseBlock

def register_file_tools() -> List[Tool]:
    """
    Register Claude-compatible file-related tools.
    
    Returns:
        List of tools to register with Claude
    """
    tools = [
        read_file_tool(),
        write_file_tool(),
        list_directory_tool(),
        find_files_tool(),
        diff_tool()
    ]
    
    return tools


def read_file_tool() -> Tool:
    """
    Create a tool for reading files.
    
    Returns:
        Tool specification
    """
    return Tool(
        name="read_file",
        description="Read the contents of a file",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8"
                }
            },
            "required": ["path"]
        }
    )


def write_file_tool() -> Tool:
    """
    Create a tool for writing files.
    
    Returns:
        Tool specification
    """
    return Tool(
        name="write_file",
        description="Write content to a file",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8"
                }
            },
            "required": ["path", "content"]
        }
    )


def list_directory_tool() -> Tool:
    """
    Create a tool for listing directory contents.
    
    Returns:
        Tool specification
    """
    return Tool(
        name="list_directory",
        description="List the contents of a directory",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Whether to include hidden files (starting with .)",
                    "default": False
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Regex pattern to filter files (e.g., '\\.py$' for Python files)",
                    "default": ""
                }
            },
            "required": ["path"]
        }
    )


def find_files_tool() -> Tool:
    """
    Create a tool for finding files recursively.
    
    Returns:
        Tool specification
    """
    return Tool(
        name="find_files",
        description="Find files recursively in a directory",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Root directory to search in"
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to match files (e.g., '\\.py$' for Python files)"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively in subdirectories",
                    "default": True
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth to search (0 for unlimited)",
                    "default": 0
                }
            },
            "required": ["path", "pattern"]
        }
    )


def diff_tool() -> Tool:
    """
    Create a tool for generating diffs between texts.
    
    Returns:
        Tool specification
    """
    return Tool(
        name="generate_diff",
        description="Generate a unified diff between two texts",
        input_schema={
            "type": "object",
            "properties": {
                "original": {
                    "type": "string",
                    "description": "Original text"
                },
                "modified": {
                    "type": "string",
                    "description": "Modified text"
                },
                "filename": {
                    "type": "string",
                    "description": "Filename to use in diff headers",
                    "default": "file.txt"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines in the diff",
                    "default": 3
                }
            },
            "required": ["original", "modified"]
        }
    )


class FileTools:
    """
    Implementation of file tools for use with Claude.
    """
    
    def __init__(self, file_manager):
        """
        Initialize file tools.
        
        Args:
            file_manager: File manager to use for operations
        """
        self.file_manager = file_manager
    
    async def handle_tool_use(self, tool_use: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a tool use request from Claude.
        
        Args:
            tool_use: Tool use request
            
        Returns:
            Tool use response
        """
        tool_name = tool_use.get('name')
        tool_params = tool_use.get('input', {})
        
        try:
            if tool_name == 'read_file':
                return await self._handle_read_file(tool_params)
            elif tool_name == 'write_file':
                return await self._handle_write_file(tool_params)
            elif tool_name == 'list_directory':
                return await self._handle_list_directory(tool_params)
            elif tool_name == 'find_files':
                return await self._handle_find_files(tool_params)
            elif tool_name == 'generate_diff':
                return await self._handle_generate_diff(tool_params)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {
                "error": str(e)
            }
    
    async def _handle_read_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle read_file tool.
        
        Args:
            params: Tool parameters
            
        Returns:
            Tool response
        """
        path = params.get('path')
        encoding = params.get('encoding', 'utf-8')
        
        if not path:
            return {"error": "Missing required parameter: path"}
        
        try:
            content = await self.file_manager.read_file(path)
            return {
                "content": content,
                "encoding": encoding,
                "path": path,
                "size_bytes": len(content.encode(encoding))
            }
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}
        except Exception as e:
            return {"error": f"Error reading file: {str(e)}"}
    
    async def _handle_write_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle write_file tool.
        
        Args:
            params: Tool parameters
            
        Returns:
            Tool response
        """
        path = params.get('path')
        content = params.get('content')
        
        if not path:
            return {"error": "Missing required parameter: path"}
        if content is None:
            return {"error": "Missing required parameter: content"}
        
        try:
            success = await self.file_manager.write_file(path, content)
            
            if success:
                return {
                    "success": True,
                    "path": path,
                    "size_bytes": len(content.encode('utf-8'))
                }
            else:
                return {"error": f"Failed to write file: {path}"}
                
        except Exception as e:
            return {"error": f"Error writing file: {str(e)}"}
    
    async def _handle_list_directory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle list_directory tool.
        
        Args:
            params: Tool parameters
            
        Returns:
            Tool response
        """
        path = params.get('path', '.')
        include_hidden = params.get('include_hidden', False)
        file_pattern = params.get('file_pattern', '')
        
        # Make path absolute
        path = self.file_manager._get_absolute_path(path)
        
        try:
            if not os.path.exists(path):
                return {"error": f"Directory not found: {path}"}
                
            if not os.path.isdir(path):
                return {"error": f"Not a directory: {path}"}
            
            # Get directory contents
            entries = os.listdir(path)
            
            # Compile regex if pattern provided
            pattern = None
            if file_pattern:
                try:
                    pattern = re.compile(file_pattern)
                except re.error:
                    return {"error": f"Invalid regex pattern: {file_pattern}"}
            
            # Filter and classify entries
            files = []
            directories = []
            
            for entry in entries:
                # Skip hidden files if not requested
                if not include_hidden and entry.startswith('.'):
                    continue
                
                # Check pattern if provided
                if pattern and not pattern.search(entry):
                    continue
                
                full_path = os.path.join(path, entry)
                
                if os.path.isdir(full_path):
                    directories.append({
                        "name": entry,
                        "type": "directory",
                        "path": full_path
                    })
                else:
                    size = os.path.getsize(full_path)
                    files.append({
                        "name": entry,
                        "type": "file",
                        "path": full_path,
                        "size_bytes": size
                    })
            
            return {
                "path": path,
                "directories": directories,
                "files": files,
                "total_entries": len(directories) + len(files)
            }
                
        except Exception as e:
            return {"error": f"Error listing directory: {str(e)}"}
    
    async def _handle_find_files(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle find_files tool.
        
        Args:
            params: Tool parameters
            
        Returns:
            Tool response
        """
        path = params.get('path', '.')
        pattern_str = params.get('pattern', '')
        recursive = params.get('recursive', True)
        max_depth = params.get('max_depth', 0)
        
        # Make path absolute
        path = self.file_manager._get_absolute_path(path)
        
        try:
            if not os.path.exists(path):
                return {"error": f"Directory not found: {path}"}
                
            if not os.path.isdir(path):
                return {"error": f"Not a directory: {path}"}
            
            # Compile regex
            try:
                pattern = re.compile(pattern_str)
            except re.error:
                return {"error": f"Invalid regex pattern: {pattern_str}"}
            
            # Find matching files
            matches = []
            
            for root, dirs, files in os.walk(path):
                # Check depth limit
                if max_depth > 0:
                    relative_path = os.path.relpath(root, path)
                    depth = 0 if relative_path == '.' else relative_path.count(os.sep) + 1
                    if depth >= max_depth:
                        dirs.clear()  # Don't descend further
                
                # Process files in this directory
                for filename in files:
                    if pattern.search(filename):
                        file_path = os.path.join(root, filename)
                        try:
                            size = os.path.getsize(file_path)
                            matches.append({
                                "name": filename,
                                "path": file_path,
                                "size_bytes": size
                            })
                        except:
                            # Skip if we can't get file info
                            pass
                
                # If not recursive, break after first iteration
                if not recursive:
                    break
            
            return {
                "matches": matches,
                "total_matches": len(matches),
                "search_path": path,
                "pattern": pattern_str
            }
                
        except Exception as e:
            return {"error": f"Error finding files: {str(e)}"}
    
    async def _handle_generate_diff(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle generate_diff tool.
        
        Args:
            params: Tool parameters
            
        Returns:
            Tool response
        """
        original = params.get('original', '')
        modified = params.get('modified', '')
        filename = params.get('filename', 'file.txt')
        context_lines = params.get('context_lines', 3)
        
        # Generate diff
        original_lines = original.splitlines(True)
        modified_lines = modified.splitlines(True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"original/{filename}",
            tofile=f"modified/{filename}",
            n=context_lines
        )
        
        diff_text = ''.join(diff)
        
        # Analyze changes
        added_lines = 0
        removed_lines = 0
        
        for line in diff_text.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                added_lines += 1
            elif line.startswith('-') and not line.startswith('---'):
                removed_lines += 1
        
        return {
            "diff": diff_text,
            "changes": {
                "added_lines": added_lines,
                "removed_lines": removed_lines,
                "total_changes": added_lines + removed_lines
            },
            "has_changes": added_lines > 0 or removed_lines > 0
        }
