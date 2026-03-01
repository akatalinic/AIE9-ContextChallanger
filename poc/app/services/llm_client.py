import os
import logging

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI client initialization failed because OPENAI_API_KEY is missing.")
        raise RuntimeError("OPENAI_API_KEY is missing.")
    logger.debug("OpenAI client initialized")
    return OpenAI(api_key=api_key)


def get_model_name(name: str) -> str:
    value = os.getenv(name)
    if not value:
        logger.error("Missing model env var | name=%s", name)
        raise RuntimeError(f"Missing model env var: {name}")
    logger.debug("Model resolved | name=%s model=%s", name, value)
    return value
