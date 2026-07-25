# utils/llm.py
import os
from langchain_groq import ChatGroq

# We'll support multiple providers, but default to Groq
def get_llm():
    """Returns the configured LLM instance."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=api_key,
            temperature=0.1
        )
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI support requires langchain-openai. "
                "Run: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1
        )
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "Anthropic support requires langchain-anthropic. "
                "Run: pip install langchain-anthropic"
            )
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported: 'groq', 'openai', 'anthropic'"
        )

def llm_available():
    """Checks if the configured LLM API key is present in the environment."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    elif provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    elif provider == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return False