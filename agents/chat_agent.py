"""
Tool Chain Manager for automatically chaining tools together.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple, Callable

class ToolChainManager:
    """
    Manages the automatic chaining of tools to accomplish complex tasks.
    Identifies when a user request requires multiple tools and executes them in sequence.
    """
    
    def __init__(self, tool_handlers, file_manager, debug_mode=False):
        """
        Initialize the tool chain manager.
        
        Args:
            tool_handlers: Dictionary mapping tool names to handler objects
            file_manager: Manager for file operations
            debug_mode: Whether to print debug information
        """
        self.tool_handlers = tool_handlers
        self.file_manager = file_manager
        self.debug_mode = debug_mode
        
        # Define common tool chains
        self.tool_chains = {
            "file_modification": self._file_modification_chain,
            "code_analysis": self._code_analysis_chain,
            "directory_navigation": self._directory_navigation_chain,
            "file_search_and_read": self._file_search_and_read_chain,
        }
        
    def set_debug_mode(self, debug_mode: bool):
        """Set debug mode"""
        self.debug_mode = debug_mode
    
    async def identify_and_execute_chain(self, user_message: str) -> Optional[Dict[str, Any]]:
        """
        Identify if a user message should trigger a tool chain and execute it.
        
        Args:
            user_message: The user's input message
            
        Returns:
            Results from the tool chain execution or None if no chain was identified
        """
        # Try to identify which chain to use based on the user message
        chain_type = self._identify_chain_type(user_message)
        
        if not chain_type:
            return None
            
        if self.debug_mode:
            print(f"[CHAIN] Identified tool chain: {chain_type}")
            
        # Extract parameters from the user message
        params = self._extract_parameters(user_message, chain_type)
        
        if self.debug_mode:
            print(f"[CHAIN] Extracted parameters: {params}")
            
        # Execute the identified chain
        return await self.tool_chains[chain_type](params)
    
    def _identify_chain_type(self, message: str) -> Optional[str]:
        """
        Identify which tool chain type (if any) should be used for a message.
        
        Args:
            message: User message
            
        Returns:
            Chain type identifier or None if no chain matches
        """
        # File modification patterns
        file_mod_patterns = [
            r'modify\s+(?:the\s+)?(?:code|file)\s+(?:in|of)\s+([^\s,]+)',
            r'change\s+(?:the\s+)?(?:code|file)\s+(?:in|of)\s+([^\s,]+)',
            r'update\s+(?:the\s+)?(?:code|file)\s+(?:in|of)\s+([^\s,]+)',
            r'edit\s+(?:the\s+)?(?:code|file)\s+(?:in|of)\s+([^\s,]+)',
            r'code:change:(.+)'
        ]
        
        # Code analysis patterns
        code_analysis_patterns = [
            r'analyze\s+(?:the\s+)?code\s+(?:in|of)\s+([^\s,]+)',
            r'review\s+(?:the\s+)?code\s+(?:in|of)\s+([^\s,]+)',
            r'examine\s+(?:the\s+)?code\s+(?:in|of)\s+([^\s,]+)',
            r'code:analyze:(.+)'
        ]
        
        # Directory navigation patterns
        dir_nav_patterns = [
            r'(?:go|navigate|change)\s+to\s+directory\s+([^\s,]+)',
            r'set\s+working\s+directory\s+to\s+([^\s,]+)',
            r'change\s+(?:the\s+)?directory\s+to\s+([^\s,]+)',
            r'cd\s+([^\s,]+)',
            r'code:workdir:(.+)'
        ]
        
        # File search and read patterns
        file_search_patterns = [
            r'find\s+(?:and\s+)?(?:read|open|show)\s+([^\s,]+)',
            r'search\s+for\s+(?:and\s+)?(?:read|open|show)\s+([^\s,]+)',
            r'locate\s+(?:and\s+)?(?:read|open|show)\s+([^\s,]+)'
        ]
        
        # Check each pattern group
        for patterns, chain_type in [
            (file_mod_patterns, "file_modification"),
            (code_analysis_patterns, "code_analysis"),
            (dir_nav_patterns, "directory_navigation"),
            (file_search_patterns, "file_search_and_read")
        ]:
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return chain_type
                    
        return None
    
    def _extract_parameters(self, message: str, chain_type: str) -> Dict[str, Any]:
        """
        Extract parameters from a user message based on the chain type.
        
        Args:
            message: User message
            chain_type: Identified chain type
            
        Returns:
            Dictionary of parameters for the chain
        """
        params = {}
        
        if chain_type == "file_modification":
            # Extract filepath and modification details
            filepath_match = re.search(r'(?:modify|change|update|edit)(?:\s+the)?\s+(?:code|file)\s+(?:in|of)\s+([^\s,]+)', message, re.IGNORECASE)
            
            # Also check for code:change: format
            if not filepath_match:
                code_change_match = re.search(r'code:change:([^:]+)(?::(.+))?', message)
                if code_change_match:
                    params['filepath'] = code_change_match.group(1).strip()
                    if code_change_match.group(2):
                        params['prompt'] = code_change_match.group(2).strip()
            else:
                params['filepath'] = filepath_match.group(1).strip()
                
            # Extract modification description if available
            prompt_match = re.search(r'(?:to|with|by)\s+(.+?)(?:\.|\?|$)', message)
            if prompt_match and 'prompt' not in params:
                params['prompt'] = prompt_match.group(1).strip()
                
        elif chain_type == "code_analysis":
            # Extract filepath
            filepath_match = re.search(r'(?:analyze|review|examine)(?:\s+the)?\s+code\s+(?:in|of)\s+([^\s,]+)', message, re.IGNORECASE)
            
            # Also check for code:analyze: format
            if not filepath_match:
                code_analyze_match = re.search(r'code:analyze:([^:]+)', message)
                if code_analyze_match:
                    params['filepath'] = code_analyze_match.group(1).strip()
            else:
                params['filepath'] = filepath_match.group(1).strip()
                
            # Extract analysis type if specified
            if 'structure' in message.lower():
                params['analysis_type'] = 'structure'
            elif 'pylint' in message.lower():
                params['analysis_type'] = 'pylint'
            elif 'full' in message.lower():
                params['analysis_type'] = 'full'
            else:
                params['analysis_type'] = 'basic'
                
        elif chain_type == "directory_navigation":
            # Extract directory path
            dir_match = re.search(r'(?:go|navigate|change|cd)\s+to\s+directory\s+([^\s,]+)', message, re.IGNORECASE)
            if not dir_match:
                dir_match = re.search(r'set\s+working\s+directory\s+to\s+([^\s,]+)', message, re.IGNORECASE)
            if not dir_match:
                dir_match = re.search(r'change\s+(?:the\s+)?directory\s+to\s+([^\s,]+)', message, re.IGNORECASE)
            if not dir_match:
                dir_match = re.search(r'cd\s+([^\s,]+)', message)
            if not dir_match:
                dir_match = re.search(r'code:workdir:(.+)', message)
                
            if dir_match:
                params['path'] = dir_match.group(1).strip()
                
        elif chain_type == "file_search_and_read":
            # Extract file pattern or name
            search_match = re.search(r'(?:find|search|locate)(?:\s+(?:and\s+)?(?:read|open|show))?\s+([^\s,]+)', message, re.IGNORECASE)
            if search_match:
                params['pattern'] = search_match.group(1).strip()
                
            # Extract directory path if specified
            dir_match = re.search(r'in\s+(?:directory|folder)?\s+([^\s,]+)', message, re.IGNORECASE)
            if dir_match:
                params['path'] = dir_match.group(1).strip()
            else:
                params['path'] = '.'  # Default to current directory
                
            # Check if recursive search is needed
            params['recursive'] = 'recursive' in message.lower()
            
        return params
    
    async def _file_modification_chain(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a file modification chain: read_file → modify_code
        
        Args:
            params: Parameters for the chain
            
        Returns:
            Results from the chain execution
        """
        filepath = params.get('filepath')
        prompt = params.get('prompt', 'Improve this code')
        
        if not filepath:
            return {"error": "No filepath specified for file modification"}
            
        if self.debug_mode:
            print(f"[CHAIN] Executing file modification chain for {filepath}")
            
        # Step 1: Read the file
        read_handler = self.tool_handlers.get('read_file')
        if not read_handler:
            return {"error": "read_file tool not available"}
            
        read_result = await read_handler.handle_tool_use({
            "name": "read_file",
            "input": {"path": filepath}
        })
        
        if "error" in read_result:
            return {"error": f"Failed to read file: {read_result['error']}"}
            
        # Get file content
        original_content = read_result.get('content', '')
        
        # Step 2: Modify the code
        modify_handler = self.tool_handlers.get('modify_code')
        if not modify_handler:
            return {"error": "modify_code tool not available"}
            
        # For a real implementation, you might want to use Claude to suggest modifications
        # based on the original content and the prompt. For now, we'll just use a placeholder.
        new_content = original_content  # In a real implementation, this would be modified
        
        modify_result = await modify_handler.handle_tool_use({
            "name": "modify_code",
            "input": {
                "filepath": filepath,
                "original_code": original_content,
                "new_code": new_content,
                "prompt": prompt
            }
        })
        
        return {
            "chain_type": "file_modification",
            "filepath": filepath,
            "read_result": read_result,
            "modify_result": modify_result,
            "success": "error" not in modify_result
        }
    
    async def _code_analysis_chain(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a code analysis chain: read_file → analyze_code
        
        Args:
            params: Parameters for the chain
            
        Returns:
            Results from the chain execution
        """
        filepath = params.get('filepath')
        analysis_type = params.get('analysis_type', 'basic')
        
        if not filepath:
            return {"error": "No filepath specified for code analysis"}
            
        if self.debug_mode:
            print(f"[CHAIN] Executing code analysis chain for {filepath}")
            
        # Step 1: Read the file
        read_handler = self.tool_handlers.get('read_file')
        if not read_handler:
            return {"error": "read_file tool not available"}
            
        read_result = await read_handler.handle_tool_use({
            "name": "read_file",
            "input": {"path": filepath}
        })
        
        if "error" in read_result:
            return {"error": f"Failed to read file: {read_result['error']}"}
            
        # Step 2: Analyze the code
        analyze_handler = self.tool_handlers.get('analyze_code')
        if not analyze_handler:
            return {"error": "analyze_code tool not available"}
            
        analyze_result = await analyze_handler.handle_tool_use({
            "name": "analyze_code",
            "input": {
                "filepath": filepath,
                "analysis_type": analysis_type
            }
        })
        
        return {
            "chain_type": "code_analysis",
            "filepath": filepath,
            "analysis_type": analysis_type,
            "read_result": read_result,
            "analyze_result": analyze_result,
            "success": "error" not in analyze_result
        }
    
    async def _directory_navigation_chain(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a directory navigation chain: set_working_directory → list_directory
        
        Args:
            params: Parameters for the chain
            
        Returns:
            Results from the chain execution
        """
        path = params.get('path')
        
        if not path:
            return {"error": "No path specified for directory navigation"}
            
        if self.debug_mode:
            print(f"[CHAIN] Executing directory navigation chain for {path}")
            
        # Step 1: Set working directory
        dir_handler = self.tool_handlers.get('set_working_directory')
        if not dir_handler:
            return {"error": "set_working_directory tool not available"}
            
        dir_result = await dir_handler.handle_tool_use({
            "name": "set_working_directory",
            "input": {"path": path}
        })
        
        if "error" in dir_result:
            return {"error": f"Failed to set directory: {dir_result['error']}"}
            
        # Step 2: List directory contents
        list_handler = self.tool_handlers.get('list_directory')
        if not list_handler:
            return {"error": "list_directory tool not available"}
            
        list_result = await list_handler.handle_tool_use({
            "name": "list_directory",
            "input": {"path": path}
        })
        
        return {
            "chain_type": "directory_navigation",
            "path": path,
            "directory_result": dir_result,
            "listing_result": list_result,
            "success": "error" not in list_result
        }
    
    async def _file_search_and_read_chain(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a file search and read chain: find_files → read_file
        
        Args:
            params: Parameters for the chain
            
        Returns:
            Results from the chain execution
        """
        pattern = params.get('pattern')
        path = params.get('path', '.')
        recursive = params.get('recursive', False)
        
        if not pattern:
            return {"error": "No pattern specified for file search"}
            
        if self.debug_mode:
            print(f"[CHAIN] Executing file search and read chain for pattern '{pattern}' in {path}")
            
        # Step 1: Find files
        find_handler = self.tool_handlers.get('find_files')
        if not find_handler:
            return {"error": "find_files tool not available"}
            
        find_result = await find_handler.handle_tool_use({
            "name": "find_files",
            "input": {
                "path": path,
                "pattern": pattern,
                "recursive": recursive
            }
        })
        
        if "error" in find_result:
            return {"error": f"Failed to find files: {find_result['error']}"}
            
        # Get matches
        matches = find_result.get('matches', [])
        
        if not matches:
            return {
                "chain_type": "file_search_and_read",
                "pattern": pattern,
                "path": path,
                "find_result": find_result,
                "success": True,
                "message": f"No files matching '{pattern}' found in {path}"
            }
            
        # Step 2: Read the first matching file
        # In a full implementation, you might want to handle multiple files
        first_match = matches[0]
        filepath = first_match.get('path')
        
        read_handler = self.tool_handlers.get('read_file')
        if not read_handler:
            return {"error": "read_file tool not available"}
            
        read_result = await read_handler.handle_tool_use({
            "name": "read_file",
            "input": {"path": filepath}
        })
        
        return {
            "chain_type": "file_search_and_read",
            "pattern": pattern,
            "path": path,
            "find_result": find_result,
            "read_result": read_result,
            "matches": matches,
            "read_filepath": filepath,
            "success": "error" not in read_result
        }
