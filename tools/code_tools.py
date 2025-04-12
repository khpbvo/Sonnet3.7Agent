"""
Tools for code-related operations using Claude's tool functionality.
"""

import os
import sys
import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple

from anthropic.types import Tool


def register_code_tools() -> List[Tool]:
    """
    Register Claude-compatible code-related tools.
    
    Returns:
        List of tools to register with Claude
    """
    tools = [
        generate_code_tool(),
        modify_code_tool(),
        parse_diff_suggestions_tool(),
        apply_changes_tool(),
        analyze_code_tool()
    ]
    
    return tools


def generate_code_tool() -> Tool:
    """
    Create a tool for generating code.
    
    Returns:
        Tool specification
    """
    return Tool(
        name="generate_code",
        description="Generate new code based on a prompt and save it to a file",
        input_schema={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to save the generated code"
                },
                "code": {
                    "type": "string",
                    "description": "The code to write to the file"
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Whether to ask for confirmation before saving",
                    "default": True
                }
            },
            "required": ["filepath", "code"]
        }
    )


def modify_code_tool() -> Tool:
    """
    Create a tool for modifying existing code.
    
    Returns:
        Tool specification
    """
    return Tool(
        name="modify_code",
        description="Modify existing code in a file",
        input_schema={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to modify"
                },
                "original_code": {
                    "type": "string",
                    "description": "The original code segment to replace"
                },
                "new_code": {
                    "type": "string",
                    "description": "The new code to replace it with"
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Whether to ask for confirmation before saving",
                    "default": True
                }
            },
            "required": ["filepath", "original_code", "new_code"]
        }
    )


def parse_diff_suggestions_tool() -> Tool:
    """
    Create a tool for parsing diff suggestions.
    
    Returns:
        Tool specification
    """
    return Tool(
        name="parse_diff_suggestions",
        description="Parse code change suggestions from text into structured format",
        input_schema={
            "type": "object",
            "properties": {
                "suggestion_text": {
                    "type": "string",
                    "description": "Text containing code change suggestions"
                }
            },
            "required": ["suggestion_text"]
        }
    )


def apply_changes_tool() -> Tool:
    """
    Create a tool for applying changes to a file.
    
    Returns:
        Tool specification
    """
    return Tool(
        name="apply_changes",
        description="Apply structured code changes to a file",
        input_schema={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to modify"
                },
                "changes": {
                    "type": "array",
                    "description": "Array of changes to apply",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line": {
                                "type": "integer",
                                "description": "Line number to modify (0 for whole-file changes)"
                            },
                            "old_code": {
                                "type": "string",
                                "description": "Original code to replace"
                            },
                            "new_code": {
                                "type": "string",
                                "description": "New code to insert"
                            }
                        },
                        "required": ["line", "old_code", "new_code"]
                    }
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Whether to ask for confirmation before saving",
                    "default": True
                }
            },
            "required": ["filepath", "changes"]
        }
    )


def analyze_code_tool() -> Tool:
    """
    Create a tool for analyzing code.
    
    Returns:
        Tool specification
    """
    return Tool(
        name="analyze_code",
        description="Analyze a code file and provide structure information",
        input_schema={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to analyze"
                },
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis to perform",
                    "enum": ["basic", "structure", "pylint", "full"],
                    "default": "basic"
                }
            },
            "required": ["filepath"]
        }
    )


class CodeTools:
    """
    Implementation of code-related tools for use with Claude.
    """
    
    def __init__(self, file_manager, code_analyzer=None):
        """
        Initialize code tools.
        
        Args:
            file_manager: File manager to use for operations
            code_analyzer: Optional code analyzer for advanced analysis
        """
        self.file_manager = file_manager
        self.code_analyzer = code_analyzer
    
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
            if tool_name == 'generate_code':
                return await self._handle_generate_code(tool_params)
            elif tool_name == 'modify_code':
                return await self._handle_modify_code(tool_params)
            elif tool_name == 'parse_diff_suggestions':
                return await self._handle_parse_diff_suggestions(tool_params)
            elif tool_name == 'apply_changes':
                return await self._handle_apply_changes(tool_params)
            elif tool_name == 'analyze_code':
                return await self._handle_analyze_code(tool_params)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {
                "error": str(e)
            }
    
    async def _handle_generate_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle generate_code tool.
        
        Args:
            params: Tool parameters
            
        Returns:
            Tool response
        """
        filepath = params.get('filepath')
        code = params.get('code')
        confirm = params.get('confirm', True)
        
        if not filepath:
            return {"error": "Missing required parameter: filepath"}
        if not code:
            return {"error": "Missing required parameter: code"}
        
        try:
            # Check if file exists
            file_exists = os.path.exists(self.file_manager._get_absolute_path(filepath))
            
            # If file exists, generate a diff
            if file_exists:
                existing_content = await self.file_manager.read_file(filepath)
                diff = await self.file_manager.generate_diff(existing_content, code, filepath)
                
                # In a real application, you'd show the diff and ask for confirmation here
                if confirm:
                    # This is a placeholder for the confirmation logic
                    # In the actual implementation, you'd integrate with your UI
                    pass
            
            # Write the code to the file
            success = await self.file_manager.write_file(filepath, code)
            
            if success:
                return {
                    "success": True,
                    "filepath": filepath,
                    "action": "created" if not file_exists else "updated",
                    "size_bytes": len(code.encode('utf-8'))
                }
            else:
                return {"error": f"Failed to write file: {filepath}"}
                
        except Exception as e:
            return {"error": f"Error generating code: {str(e)}"}
    
    async def _handle_modify_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle modify_code tool.
        
        Args:
            params: Tool parameters
            
        Returns:
            Tool response
        """
        filepath = params.get('filepath')
        original_code = params.get('original_code')
        new_code = params.get('new_code')
        confirm = params.get('confirm', True)
        
        if not filepath:
            return {"error": "Missing required parameter: filepath"}
        if original_code is None:
            return {"error": "Missing required parameter: original_code"}
        if new_code is None:
            return {"error": "Missing required parameter: new_code"}
        
        try:
            # Check if file exists
            absolute_path = self.file_manager._get_absolute_path(filepath)
            if not os.path.exists(absolute_path):
                return {"error": f"File not found: {filepath}"}
            
            # Read the current content
            content = await self.file_manager.read_file(filepath)
            
            # Replace the code
            if original_code not in content:
                return {"error": f"Original code segment not found in {filepath}"}
            
            modified_content = content.replace(original_code, new_code)
            
            # Generate a diff
            diff = await self.file_manager.generate_diff(content, modified_content, filepath)
            
            # In a real application, you'd show the diff and ask for confirmation here
            if confirm:
                # This is a placeholder for the confirmation logic
                # In the actual implementation, you'd integrate with your UI
                pass
            
            # Write the modified content
            success = await self.file_manager.write_file(filepath, modified_content)
            
            if success:
                return {
                    "success": True,
                    "filepath": filepath,
                    "diff": diff
                }
            else:
                return {"error": f"Failed to write file: {filepath}"}
                
        except Exception as e:
            return {"error": f"Error modifying code: {str(e)}"}
    
    async def _handle_parse_diff_suggestions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle parse_diff_suggestions tool.
        
        Args:
            params: Tool parameters
            
        Returns:
            Tool response
        """
        suggestion_text = params.get('suggestion_text')
        
        if not suggestion_text:
            return {"error": "Missing required parameter: suggestion_text"}
        
        try:
            changes = []
            
            # Pattern 1: "Line X: replace ... with ..."
            line_pattern = re.compile(r"line\s+(\d+):\s*(?:replace|change)\s+'([^']+)'\s+(?:with|to)\s+'([^']+)'", re.IGNORECASE)
            for match in line_pattern.finditer(suggestion_text):
                line_num = int(match.group(1))
                old_code = match.group(2)
                new_code = match.group(3)
                changes.append({"line": line_num, "old_code": old_code, "new_code": new_code})
            
            # Pattern 2: Code blocks with - and + prefixes
            code_block_pattern = re.compile(r"```(?:python|diff)?\n(.*?)```", re.DOTALL)
            for block_match in code_block_pattern.finditer(suggestion_text):
                block = block_match.group(1)
                lines = block.split('\n')
                
                # Check if this is a git-style diff
                if any(line.startswith('@@ ') for line in lines) or any(line.startswith('--- ') for line in lines):
                    # Process git-style diff - simplified approach
                    current_line = 0
                    for i, line in enumerate(lines):
                        if line.startswith('@@ '):
                            # Parse hunk header
                            header_match = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)', line)
                            if header_match:
                                current_line = int(header_match.group(1))
                        elif line.startswith('-') and i+1 < len(lines) and lines[i+1].startswith('+'):
                            # Replacement
                            old = line[1:].strip()
                            new = lines[i+1][1:].strip()
                            changes.append({"line": current_line, "old_code": old, "new_code": new})
                        elif not line.startswith('+') and not line.startswith('-') and not line.startswith('---') and not line.startswith('+++'):
                            # Context line
                            current_line += 1
                    continue
                
                # Process regular +/- block
                current_line = 0
                for i, line in enumerate(lines):
                    if line.startswith('-') and i+1 < len(lines) and lines[i+1].startswith('+'):
                        old = line[1:].strip()
                        new = lines[i+1][1:].strip()
                        # Use an approximate line number since we don't have exact references
                        changes.append({"line": current_line, "old_code": old, "new_code": new})
                    
                    if not line.startswith('+'):
                        current_line += 1
            
            # Pattern 3: "Replace this: ... With this: ..." sections
            replace_pattern = re.compile(r"(?:replace|change) this:\s*```(?:python)?\n(.*?)```\s*(?:with|to)(?::|) this:\s*```(?:python)?\n(.*?)```", re.IGNORECASE | re.DOTALL)
            for match in replace_pattern.finditer(suggestion_text):
                old_code = match.group(1).strip()
                new_code = match.group(2).strip()
                # Use line number 0 as placeholder for whole-file changes
                changes.append({"line": 0, "old_code": old_code, "new_code": new_code})
            
            return {
                "changes": changes,
                "count": len(changes)
            }
                
        except Exception as e:
            return {"error": f"Error parsing suggestions: {str(e)}"}
    
    async def _handle_apply_changes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle apply_changes tool.
        
        Args:
            params: Tool parameters
            
        Returns:
            Tool response
        """
        filepath = params.get('filepath')
        changes = params.get('changes', [])
        confirm = params.get('confirm', True)
        
        if not filepath:
            return {"error": "Missing required parameter: filepath"}
        if not changes:
            return {"error": "No changes provided"}
        
        try:
            # Check if file exists
            absolute_path = self.file_manager._get_absolute_path(filepath)
            if not os.path.exists(absolute_path):
                return {"error": f"File not found: {filepath}"}
            
            # Read the current content
            content = await self.file_manager.read_file(filepath)
            
            # Apply changes
            modified_content = content
            lines = content.splitlines()
            
            # Sort changes from highest to lowest line number to avoid index shifting
            sorted_changes = sorted(changes, key=lambda x: x.get('line', 0), reverse=True)
            
            for change in sorted_changes:
                line_num = change.get('line', 0)
                old_code = change.get('old_code', '')
                new_code = change.get('new_code', '')
                
                if line_num == 0:
                    # Whole-file replacement
                    if old_code in modified_content:
                        modified_content = modified_content.replace(old_code, new_code)
                elif 1 <= line_num <= len(lines):
                    # Line-specific replacement
                    if old_code in lines[line_num-1]:
                        lines[line_num-1] = lines[line_num-1].replace(old_code, new_code)
            
            # Rebuild content from lines if we made line-specific changes
            if any(change.get('line', 0) > 0 for change in changes):
                modified_content = '\n'.join(lines)
            
            # Generate a diff
            diff = await self.file_manager.generate_diff(content, modified_content, filepath)
            
            # In a real application, you'd show the diff and ask for confirmation here
            if confirm:
                # This is a placeholder for the confirmation logic
                # In the actual implementation, you'd integrate with your UI
                pass
            
            # Write the modified content
            success = await self.file_manager.write_file(filepath, modified_content)
            
            if success:
                return {
                    "success": True,
                    "filepath": filepath,
                    "changes_applied": len(changes),
                    "diff": diff
                }
            else:
                return {"error": f"Failed to write file: {filepath}"}
                
        except Exception as e:
            return {"error": f"Error applying changes: {str(e)}"}
    
    async def _handle_analyze_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle analyze_code tool.
        
        Args:
            params: Tool parameters
            
        Returns:
            Tool response
        """
        filepath = params.get('filepath')
        analysis_type = params.get('analysis_type', 'basic')
        
        if not filepath:
            return {"error": "Missing required parameter: filepath"}
        
        try:
            # Check if file exists
            absolute_path = self.file_manager._get_absolute_path(filepath)
            if not os.path.exists(absolute_path):
                return {"error": f"File not found: {filepath}"}
            
            # Read the file
            content = await self.file_manager.read_file(filepath)
            
            # Basic analysis - always performed
            result = {
                "filepath": filepath,
                "size_bytes": len(content.encode('utf-8')),
                "line_count": content.count('\n') + 1,
                "is_python": filepath.endswith('.py')
            }
            
            # Process based on analysis type
            if self.code_analyzer and filepath.endswith('.py'):
                if analysis_type == 'structure':
                    # Get structure information
                    structure = await asyncio.to_thread(
                        self.code_analyzer.get_structure_overview, 
                        absolute_path
                    )
                    result["structure"] = structure
                    
                elif analysis_type == 'pylint':
                    # Get Pylint report
                    if hasattr(self.code_analyzer, 'get_pylint_report'):
                        pylint_report = await asyncio.to_thread(
                            self.code_analyzer.get_pylint_report, 
                            absolute_path
                        )
                        result["pylint_report"] = pylint_report
                    else:
                        result["error"] = "Pylint analysis not available"
                        
                elif analysis_type == 'full':
                    # Get comprehensive report
                    if hasattr(self.code_analyzer, 'get_combined_report'):
                        full_report = await asyncio.to_thread(
                            self.code_analyzer.get_combined_report, 
                            absolute_path
                        )
                        result["full_report"] = full_report
                    else:
                        result["error"] = "Full analysis not available"
            
            return result
                
        except Exception as e:
            return {"error": f"Error analyzing code: {str(e)}"}
