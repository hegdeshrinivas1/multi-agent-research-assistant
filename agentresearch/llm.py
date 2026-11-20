"""Unified LLM client for AgentResearch with Gemini, retry backoff, and tracing."""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logger = logging.getLogger(__name__)

# Default high-performance model
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Check if valid Langfuse credentials exist
langfuse_pk = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
langfuse_sk = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
HAS_VALID_LANGFUSE = bool(
    langfuse_pk
    and langfuse_sk
    and not langfuse_pk.startswith("your_")
    and not langfuse_pk.startswith("pk-placeholder")
)

if HAS_VALID_LANGFUSE:
    try:
        from langfuse import observe
    except ImportError:
        def observe(*args, **kwargs):
            return lambda fn: fn
else:
    # Disable Otel background attempts when Langfuse is not actively configured
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
    def observe(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


def get_genai_client() -> genai.Client:
    """Initialize and return Google GenAI client from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")
    return genai.Client(api_key=api_key)


@observe(as_type="generation")
def call_gemini(
    prompt: str,
    system_instruction: Optional[str] = None,
    json_mode: bool = False,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> Tuple[str, Dict[str, int]]:
    """Invoke Gemini model with exponential backoff retries and token metrics.

    Args:
        prompt: User prompt content.
        system_instruction: Optional system instruction prompt.
        json_mode: Whether to enforce application/json response format.
        model: Model name to use (defaults to gemini-3.5-flash-lite).
        temperature: Generation temperature.
        max_retries: Number of retry attempts on transient network/rate errors.

    Returns:
        Tuple containing (response_text, token_usage_dict).
    """
    client = get_genai_client()

    config_args: Dict[str, Any] = {
        "temperature": temperature,
    }
    if system_instruction:
        config_args["system_instruction"] = system_instruction
    if json_mode:
        config_args["response_mime_type"] = "application/json"

    config = types.GenerateContentConfig(**config_args)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if response.usage_metadata:
                tokens["prompt_tokens"] = response.usage_metadata.prompt_token_count or 0
                tokens["completion_tokens"] = response.usage_metadata.candidates_token_count or 0
                tokens["total_tokens"] = (
                    tokens["prompt_tokens"] + tokens["completion_tokens"]
                )

            text = response.text or ""
            return text, tokens

        except Exception as exc:
            err_str = str(exc)
            logger.warning("Gemini invocation attempt %d/%d failed: %s", attempt + 1, max_retries, err_str)
            if attempt == max_retries - 1:
                logger.error("All Gemini retry attempts failed: %s", err_str)
                raise exc
            # Backoff before retrying
            backoff_time = (2 ** attempt) * 0.8
            time.sleep(backoff_time)

    return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def parse_json_safely(raw_text: str, default_data: Any) -> Any:
    """Safely parse JSON response with markdown fence stripping and error recovery.

    Args:
        raw_text: The text string to parse.
        default_data: Default fallback structure if parsing fails.

    Returns:
        Parsed dictionary/list or default_data.
    """
    clean_text = raw_text.strip()
    if clean_text.startswith("```"):
        lines = clean_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean_text = "\n".join(lines).strip()

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as err:
        logger.warning("Failed to parse JSON: %s. Raw text: %s", err, raw_text[:200])
        return default_data
