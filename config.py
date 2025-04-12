"""
Configuration settings for the ETMSonnet Assistant.
"""

class Config:
    """Configuration settings for the application."""
    
    def __init__(self):
        """Initialize configuration with default values."""
        # Claude API settings
        self.model = "claude-3-7-sonnet-20250219"  # Most recent model as of the provided date
        self.max_context_tokens = 200000
        self.max_response_tokens = 4096
        
        # Application settings
        self.stream_responses = True
        self.typing_simulation_delay = 0.01  # Delay for simulated typing effect
        self.use_colors = True
        
        # File settings
        self.default_encoding = "utf-8"
        self.fallback_encoding = "latin-1"
        
        # Router settings
        self.router_model = "claude-3-7-sonnet-20250219"  # Could use a smaller model if needed
        self.router_max_tokens = 1000
