import os
from abc import ABC, abstractmethod
import google.generativeai as genai
from autogen import AssistantAgent
from dotenv import load_dotenv

load_dotenv()

class SARBaseAgent(AssistantAgent):
    def __init__(self, name, role, system_message, knowledge_base=None):
        super().__init__(
            name=name,
            system_message=system_message
        )
        self.role = role
        self.kb = knowledge_base
        self.mission_status = "standby"

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    def query_gemini(self, prompt, model="gemini-pro", max_tokens=200):
        """Query Google Gemini API and return response."""
        try:
            response = genai.GenerativeModel(model).generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"

    @abstractmethod
    def process_request(self, message):
        """Process incoming requests - must be implemented by specific agents"""
        pass

    def update_status(self, status):
        """Update agent's mission status"""
        self.mission_status = status
        return {"status": "updated", "new_status": status}

    def get_status(self):
        """Return current status"""
        return self.mission_status
