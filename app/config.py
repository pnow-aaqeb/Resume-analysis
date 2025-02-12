import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def get_openai_client() -> OpenAI:
    """
    Creates and returns an OpenAI client instance with proper authentication.
    Raises ValueError if API key is not found.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
            "or create a .env file with OPENAI_API_KEY=your-key-here"
        )
    
    return OpenAI(api_key=api_key)

# Create a single client instance to be imported by other modules
client = get_openai_client()