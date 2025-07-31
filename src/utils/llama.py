import os
from openai import OpenAI
import openai
import pdb
with open('./deepseek_api_azure.key', 'r') as f:
    api_key = f.read().strip()
    
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

endpoint = "https://ai-ctod426436914330.services.ai.azure.com/models"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key),
)

def llama3_70b(prompt):
    response = client.complete(
        messages=[
            UserMessage(content=prompt)
        ],
        max_tokens=2048,
        model="Llama-3.3-70B-Instruct"
    )

    return response.choices[0].message.content

def llama3_8b(prompt):
    response = client.complete(
        messages=[
            UserMessage(content=prompt)
        ],
        max_tokens=2048,
        model="Meta-Llama-3.1-8B-Instruct"
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    temp = llama3_8b("Translate this sentence from English to French. I love programming.")
    print(temp)