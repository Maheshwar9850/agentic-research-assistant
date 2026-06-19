import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FailoverLLMWrapper:
    def __init__(self, primary_llm, fallback_llm):
        self.primary = primary_llm
        self.fallback = fallback_llm
        self.current_provider = "Gemini"
        self.fallback_count = 0

    def invoke(self, prompt, **kwargs):
        try:
            self.current_provider = "Gemini"
            logger.info("Using Gemini")
            return self.primary.invoke(prompt, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if any(term in error_msg for term in ["429", "resource_exhausted", "quota", "rate limit", "too many requests"]):
                self.fallback_count += 1
                self.current_provider = "Groq"
                logger.warning("Gemini quota exceeded")
                logger.info("Switching to Groq fallback")
                logger.info(f"Fallback Count: {self.fallback_count}")
                try:
                    return self.fallback.invoke(prompt, **kwargs)
                except Exception as groq_e:
                    logger.error(f"Groq fallback also failed: {groq_e}")
                    return AIMessage(content="Error: Both primary and fallback AI models are currently experiencing high traffic or quota limits. Please wait a moment and try again.")
            raise e

def get_llm():
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("google_api_key")
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    primary = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.2,
    )
    
    fallback = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.2,
    )
    
    return FailoverLLMWrapper(primary, fallback)
