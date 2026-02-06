"""
LLM Client Module
Unified interface for calling LLM models.
Supports:
- Google Gemini 2.5 Pro via google-genai SDK
- Anthropic Claude Sonnet 4 via anthropic SDK
- USAI for GSA-internal Claude access
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv


class GeminiClient:
    """
    Google Gemini API client using the google-genai SDK.
    Default model: gemini-2.5-pro
    """
    
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable."
            )
        
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            print(f"✅ Gemini Client initialized (google-genai SDK)")
        except ImportError:
            raise ImportError(
                "google-genai package not installed. Install with: pip install google-genai"
            )
    
    def call(
        self, 
        prompt: str, 
        model: str = "gemini-2.5-pro",
        system_message: str = "You are a helpful policy analyst.",
        max_retries: int = 3
    ) -> Optional[str]:
        """Make an LLM API call with retry logic."""
        full_prompt = f"{system_message}\n\n{prompt}"
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=full_prompt
                )
                return response.text
                
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "quota" in error_str:
                    wait_time = 2 ** attempt
                    print(f"🔄 Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif "timeout" in error_str:
                    print(f"⏳ Timeout on attempt {attempt + 1}/{max_retries}")
                    time.sleep(2 ** attempt)
                else:
                    print(f"❌ Gemini API Error: {e}")
                    return None
        
        print(f"❌ All {max_retries} attempts failed")
        return None
    
    def test_connection(self) -> bool:
        """Test API connectivity."""
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents="Say hello in one word."
            )
            print(f"✅ Gemini connected")
            return True
        except Exception as e:
            print(f"❌ Gemini connection failed: {e}")
            return False


class AnthropicClient:
    """
    Anthropic Claude API client using the anthropic SDK.
    Default model: claude-sonnet-4-20250514
    """
    
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY') or os.getenv('CLAUDE_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print(f"✅ Anthropic Client initialized (Claude Sonnet 4)")
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )
    
    def call(
        self, 
        prompt: str, 
        model: str = "claude-sonnet-4-20250514",
        system_message: str = "You are a helpful policy analyst.",
        max_retries: int = 3
    ) -> Optional[str]:
        """Make an LLM API call with retry logic."""
        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_message,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text
                
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "overloaded" in error_str:
                    wait_time = 2 ** attempt
                    print(f"🔄 Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Anthropic API Error: {e}")
                    return None
        
        print(f"❌ All {max_retries} attempts failed")
        return None
    
    def test_connection(self) -> bool:
        """Test API connectivity."""
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=50,
                messages=[{"role": "user", "content": "Say hello in one word."}]
            )
            print(f"✅ Claude connected")
            return True
        except Exception as e:
            print(f"❌ Claude connection failed: {e}")
            return False


def create_client(model: str = "gemini-2.5-pro"):
    """
    Factory function to create the appropriate LLM client based on model.
    
    Args:
        model: Model identifier
        
    Returns:
        GeminiClient or AnthropicClient
    """
    if model.startswith("gemini"):
        return GeminiClient()
    elif model.startswith("claude"):
        return AnthropicClient()
    else:
        # Default to Gemini
        return GeminiClient()
