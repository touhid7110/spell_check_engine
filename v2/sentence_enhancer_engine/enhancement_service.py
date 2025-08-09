#INSTALLATION
#pip install google-genai
from google import genai 
from google.genai import types
from prompt_manager import PromptManager   
from string import Template
import os
import json

#GEMMA_API_KEY=os.environ.get("GEMMA_API_KEY")
GEMMA_API_KEY=##add your key here##

# enhancement_service.py
class EnhancementService:
    def __init__(self):
        self.prompt_manager = PromptManager()
        self.genai_client = genai.Client(api_key=GEMMA_API_KEY)
        self.model="gemma-3-27b-it"
    
    #async 
    def enhance(self, text: str, style: str):
        prompt_config = self.prompt_manager.get_prompt(style)
        #enhanced_text = await self.genai_client.generate(
        enhanced_text = self.genai_client.models.generate_content(
            model=self.model,
            contents=prompt_config["system"]+"\n"+Template(prompt_config["template"]).substitute(text=text),

    )
        #print(enhanced_text)
        enhanced_text_reccomendations = json.loads((enhanced_text.text).replace("```json","").replace("```",""))
        return {"enhanced_reccomendations": enhanced_text_reccomendations, "style": style}



# if __name__=="__main__":
#     obj=EnhancementService()
#     print(obj.enhance("I have merely adopted the dark","formal"))
