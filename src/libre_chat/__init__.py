"""API and UI to deploy LLM models with OpenCog cognitive capabilities."""
from .utils import Prompt, log
from .conf import default_conf, parse_conf
from .llm import Llm
from .cognitive_llm import CognitiveLlm
from .router import ChatRouter
from .endpoint import ChatEndpoint

__version__ = "0.1.0"  # Updated for OpenCog integration
