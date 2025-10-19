import os
from typing import Dict, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_yaml import parse_yaml_raw_as

from libre_chat.utils import BOLD, END, YELLOW, log

__all__ = ["ChatConf", "parse_conf"]


class BaseConf(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="librechat_", extra="allow", protected_namespaces=("settings_",)
    )


class SettingsInfo(BaseConf):
    examples: List[str] = [
        "What is the capital of the Netherlands?",
        "Which drugs are approved by the FDA to mitigate Alzheimer symptoms?",
    ]
    title: str = "Libre Chat"
    version: str = "0.1.0"
    description: str = """Open source and free chatbot powered by [LangChain](https://python.langchain.com) and [llama.cpp](https://github.com/ggerganov/llama.cpp)"""
    public_url: str = "https://your-endpoint-url"
    repository_url: str = "https://github.com/vemonet/libre-chat"
    favicon: str = "https://raw.github.com/vemonet/libre-chat/main/docs/docs/assets/logo.png"
    license_info: Dict[str, str] = {
        "name": "MIT license",
        "url": "https://raw.github.com/vemonet/libre-chat/main/LICENSE",
    }
    contact: Dict[str, str] = {
        "name": "Vincent Emonet",
        "email": "vincent.emonet@gmail.com",
    }
    workers: int = 4


class SettingsVector(BaseConf):
    embeddings_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    # or embeddings_path: str = "./embeddings/all-MiniLM-L6-v2"
    embeddings_download: Optional[str] = None
    vector_path: Optional[str] = None
    vector_download: Optional[str] = None
    documents_path: str = "documents/"
    documents_download: Optional[str] = None

    chunk_size: int = 500
    chunk_overlap: int = 50
    chain_type: str = "stuff"  # Or: map_reduce, reduce, map_rerank https://docs.langchain.com/docs/components/chains/index_related_chains
    search_type: str = "similarity"  # Or: similarity_score_threshold, mmr https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore
    return_sources_count: int = 4
    score_threshold: Optional[float] = None  # Between 0 and 1


class SettingsLlm(BaseConf):
    model_type: str = "llama"  # TODO: Remove?
    model_path: str = "./models/mixtral-8x7b-instruct-v0.1.Q2_K.gguf"
    model_download: Optional[str] = None
    max_new_tokens: int = 1024
    temperature: float = 0.01
    gpu_layers: int = 100  # Number of layers to run on the GPU (if detected)
    prompt_variables: List[str] = ["input", "history"]
    prompt_template: str = ""


class SettingsAuth(BaseConf):
    admin_pass: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    redirect_uri: str = "http://localhost:8000/auth/callback"
    scope: str = "https://www.googleapis.com/auth/userinfo.email"
    token_url: str = "https://oauth2.googleapis.com/token"
    authorization_url: str = "https://accounts.google.com/o/oauth2/auth"
    admin_users: List[str] = []
    regular_users: List[str] = []


class SettingsOpenCog(BaseConf):
    enabled: bool = True
    atomspace_max_size: int = 100000
    attention_agent_enabled: bool = True
    attention_cycle_interval: float = 1.0
    attention_focus_boundary: float = 10.0
    attention_bank: float = 1000.0
    importance_decay_rate: float = 0.005
    moses_enabled: bool = True
    moses_population_size: int = 50
    moses_mutation_rate: float = 0.1
    moses_crossover_rate: float = 0.7
    moses_elitism_rate: float = 0.2
    moses_evolution_interval: float = 5.0
    cognitive_reasoning_enabled: bool = True
    pattern_matching_enabled: bool = True
    forward_chaining_enabled: bool = True
    backward_chaining_enabled: bool = True
    max_inference_depth: int = 3


class ChatConf(BaseConf):
    conf_path: str = "chat.yml"
    conf_url: Optional[str] = None
    llm: SettingsLlm = SettingsLlm()
    vector: SettingsVector = SettingsVector()
    info: SettingsInfo = SettingsInfo()
    auth: SettingsAuth = SettingsAuth()
    opencog: SettingsOpenCog = SettingsOpenCog()


default_conf = ChatConf()


def parse_conf(path: str = default_conf.conf_path) -> ChatConf:
    if os.path.exists(path):
        with open(path) as file:
            conf = parse_yaml_raw_as(ChatConf, file.read())
            log.info(f"📋 Loaded config from {BOLD}{YELLOW}{path}{END}")
            return conf
    else:
        return default_conf
