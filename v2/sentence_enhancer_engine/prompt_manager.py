# prompt_manager.py
import json
from typing import Dict,List

class PromptManager:
    def __init__(self, config_path: str = "prompts.json"):
        with open(config_path) as f:
            self.prompts = json.load(f)
    
    def get_prompt(self, style: str) -> Dict[str, str]:
        return self.prompts.get(style)
    
    def get_available_styles(self) -> List[str]:
        """Get list of available enhancement styles"""
        return list(self.prompts.keys())
