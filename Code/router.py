import os
from openai import OpenAI
from dotenv import load_dotenv

class Router():
    def __init__(self, model):

        load_dotenv()

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        # Add all needed models to this list
        if model == 'ChatGPT-5.4':
            self.model = "openai/gpt-5.4"
        elif model == 'ChatGPT-5.4 mini':
            self.model = "openai/gpt-5.4-mini"
        elif model == 'ChatGPT-5.4 nano':
            self.model = "openai/gpt-5.4-nano"
        elif model == 'ChatGPT-5.4 pro':
            self.model = "openai/gpt-5.4-pro"

        return
    
    def send_message(self, message):
        if not type(message) == str:
            return "Wrong message data type"
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ]
        )