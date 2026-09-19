from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=settings.google_api_key,
        temperature=0,
    )