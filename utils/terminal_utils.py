"""
Terminal utilities for the ETMSonnet Assistant.
"""

import sys
import time
import asyncio
import os
from typing import Optional, Callable

# Try to import colorama for Windows color support
try:
    import colorama
    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False


def get_multiline_input(prompt: str = "You: ") -> str:
    """
    Get multi-line input from the user, ending when 'END' is entered on a new line.
    
    Args:
        prompt: The prompt to display
        
    Returns:
        The complete multi-line input
    """
    print(prompt, end="", flush=True)
    lines = []
    
    while True:
        try:
            line = input()
            if line.strip().lower() == 'end':
                break
            lines.append(line)
        except EOFError:
            # Handle Ctrl+D gracefully
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nInput aborted.")
            return ""
    
    # Join all lines with newlines
    return "\n".join(lines)


# ANSI color codes
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
BOLD = "\033[1m"


def print_colored(text: str, color: str = None, bold: bool = False) -> None:
    """
    Print colored text if supported by the terminal.
    
    Args:
        text: Text to print
        color: Color name (red, green, yellow, blue, magenta, cyan)
        bold: Whether to print in bold
    """
    # Color code mapping
    colors = {
        "red": RED,
        "green": GREEN,
        "yellow": YELLOW,
        "blue": BLUE, 
        "magenta": MAGENTA,
        "cyan": CYAN
    }
    
    bold_code = BOLD if bold else ""
    
    # Check if colors are supported
    use_colors = False
    
    # Check if we're in a terminal that supports colors
    if os.name == 'nt':  # Windows
        use_colors = HAS_COLORAMA
    else:  # Unix/Linux/Mac
        use_colors = sys.stdout.isatty()
    
    if use_colors and color in colors:
        print(f"{bold_code}{colors[color]}{text}{RESET}")
    else:
        print(text)


async def stream_output(text: str, delay: float = 0.01) -> None:
    """
    Stream text to the console with a typing effect.
    
    Args:
        text: Text to stream
        delay: Delay between characters for typing effect
    """
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        await asyncio.sleep(delay)


def create_stream_callback(delay: float = 0.01) -> Callable[[str], None]:
    """
    Create a callback function for streaming output with typing effect.
    
    Args:
        delay: Delay between characters for typing effect
        
    Returns:
        Callback function for streaming
    """
    def callback(chunk: str) -> None:
        for char in chunk:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
    
    return callback


def print_status(icon: str, message: str, color: str = 'cyan') -> None:
    """
    Print a status message with icon and color.
    
    Args:
        icon: Icon to display at the beginning of the message
        message: Status message to display
        color: Color to use (default: cyan)
    """
    print_colored(f"{icon} {message}", color, bold=True)
