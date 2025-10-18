<div align="center">

# <span><img height="30" src="https://raw.github.com/vemonet/libre-chat/main/docs/docs/assets/logo.png"></span> Libre-Cog: OpenCog Cognitive Chatbot

[![Test package](https://github.com/vemonet/libre-chat/actions/workflows/test.yml/badge.svg)](https://github.com/vemonet/libre-chat/actions/workflows/test.yml) [![Coverage](https://coverage-badge.samuelcolvin.workers.dev/vemonet/libre-chat.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/vemonet/libre-chat)

[![PyPI - Version](https://img.shields.io/pypi/v/libre-chat.svg?logo=pypi&label=PyPI&logoColor=silver)](https://pypi.org/project/libre-chat/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/libre-chat.svg?logo=python&label=Python&logoColor=silver)](https://pypi.org/project/libre-chat/)
[![License](https://img.shields.io/pypi/l/libre-chat)](https://github.com/vemonet/libre-chat/blob/main/LICENSE.txt) [![Pull requests welcome](https://img.shields.io/badge/pull%20requests-welcome-brightgreen)](https://github.com/vemonet/libre-chat/fork)

Easily configure and deploy a **fully self-hosted cognitive chatbot web service** powered by [OpenCog](https://opencog.org) and open source Large Language Models (LLMs), such as [Mixtral](https://mistral.ai/news/mixtral-of-experts) or [Llama 2](https://ai.meta.com/llama/), featuring advanced cognitive capabilities including structured knowledge representation, attention mechanisms, and self-improving reasoning.

</div>

- 🧠 **Cognitive AI Architecture**: Powered by OpenCog's AtomSpace for structured knowledge representation and reasoning
- 🎯 **Attention Mechanisms**: ECAN (Economic Cognitive Attention Network) for intelligent focus management
- 🧬 **Self-Improving**: Moses evolutionary engine for continuous learning and program synthesis
- 🔗 **Advanced Reasoning**: Forward/backward chaining, pattern matching, and inference capabilities
- 🌐 Free and Open Source chatbot web service with UI and API.
- 🏡 Fully self-hosted, not tied to any service, and offline capable. Forget about API keys! Models and embeddings can be pre-downloaded, and the training and inference processes can run off-line if necessary.
- 🔌 Web API described using OpenAPI specs: GET/POST operations, websocket for streaming response
- 🪶 Chat web UI working well on desktop and mobile, with streaming response, and markdown rendering. Alternative gradio-based UI also available.
- 🚀 Easy to setup, no need to program, just configure the service with a [YAML](https://yaml.org/) file, and start it with 1 command
- 📦 Available as a `pip` package 🐍, or `docker` image 🐳
- 🐌 No need for GPU, this will work even on your laptop CPU! That said, just running on CPUs can be quite slow (up to 1min to answer a documents-base question on recent laptops).
- 🦜 Powered by [OpenCog](https://opencog.org), [`LangChain`](https://python.langchain.com) and [`llama.cpp`](https://github.com/ggerganov/llama.cpp) to perform inference locally.
- 🤖 Various types of cognitive agents can be deployed:
  - **🧠 Cognitive Reasoning**: Uses AtomSpace knowledge representation with attention-guided inference
  - **💬 Generic conversation**: do not need any additional training, just configure settings such as the template prompt
  - **📚 Documents-based question answering** (experimental): automatically build similarity vectors from documents uploaded through the API UI, the chatbot will use them to answer your question, and return which documents were used to generate the answer (PDF, CSV, HTML, JSON, markdown, and more supported).
  - **🎯 Hybrid Mode**: Combines cognitive reasoning with traditional LLM responses for optimal performance
- 🔍 Readable logs to understand what is going on, including cognitive state monitoring.

## 📖 Documentation

For more details on how to use Libre-Cog check the documentation at **[cogpy/libre-cog](https://github.com/cogpy/libre-cog)** and the **[OpenCog Integration Guide](docs/opencog-integration.md)**


![UI screenshot](https://raw.github.com/vemonet/libre-chat/main/docs/docs/assets/screenshot.png)

![UI screenshot](https://raw.github.com/vemonet/libre-chat/main/docs/docs/assets/screenshot-light.png)

## 🧠 Cognitive Features Quick Start

To deploy the cognitive chatbot with OpenCog capabilities:

```bash
# Clone the repository
git clone https://github.com/cogpy/libre-cog.git
cd libre-cog

# Install dependencies
pip install -e .

# Start with cognitive configuration
libre-chat start config/chat-opencog-cognitive.yml
```

### Cognitive API Endpoints

Once running, access the cognitive features:

```bash
# Check cognitive system status
curl http://localhost:8000/cognitive/state

# View current attentional focus
curl http://localhost:8000/cognitive/attention

# Monitor evolution progress
curl http://localhost:8000/cognitive/evolution

# Provide learning feedback
curl -X POST http://localhost:8000/cognitive/learn \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?", "response": "AI is...", "feedback": "good"}'
```

### Example Cognitive Interaction

```python
import requests

# Ask a cognitive question
response = requests.post("http://localhost:8000/prompt", json={
    "prompt": "What relationships exist between learning and intelligence?"
})

result = response.json()
print("Response:", result["result"])

# Check what concepts the AI is currently focusing on
attention = requests.get("http://localhost:8000/cognitive/attention")
print("Current focus:", attention.json()["focus_atoms"])
```

## 🏗️ Work in progress

> [!WARNING]
> This project is a work in progress, use it with caution.

Those checkpoints are features we plan to work on in the future, feel free to let us know in the issues if you have any comment or request.

- [x] Stream response to the websocket to show words as they are generated
- [ ] Add button to let the user stop the chatbot generation
- [ ] Add authentication mechanisms? (OAuth/OpenID Connect) https://github.com/vemonet/libre-chat/issues/5
- [ ] Add conversational history? https://milvus.io/blog/conversational-memory-in-langchain.md
- [ ] Add an admin dashboard web UI to enable users to upload/inspect/delete documents for QA, see/edit the config of the chatbot.
- [ ] Kubernetes deployment (Helm chart?)

## 🐳 Deploy with docker

If you just want to quickly deploy it using the pre-trained model `Mixtral-8x7B-Instruct`, you can use docker:

```bash
docker run -it -p 8000:8000 ghcr.io/vemonet/libre-chat:main
```

You can configure the deployment using environment variables. For this using a `docker compose` and a `.env` file is easier, first create the `docker-compose.yml` file:

```yaml
version: "3"
services:
  libre-chat:
    image: ghcr.io/vemonet/libre-chat:main
    volumes:
      # ⚠️ Share folders from the current directory to the /data dir in the container
      - ./chat.yml:/data/chat.yml
      - ./models:/data/models
      - ./documents:/data/documents
      - ./embeddings:/data/embeddings
      - ./vectorstore:/data/vectorstore
    ports:
      - 8000:8000
```

And create a `chat.yml` file with your configuration in the same folder as the `docker-compose.yml`:

```yaml
llm:
  model_path: ./models/mixtral-8x7b-instruct-v0.1.Q2_K.gguf
  model_download: https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q2_K.gguf
  temperature: 0.01    # Config how creative, but also potentially wrong, the model can be. 0 is safe, 1 is adventurous
  max_new_tokens: 1024 # Max number of words the LLM can generate
  # Always use input for the human input variable with a generic agent
  prompt_variables: [input, history]
  prompt_template: |
    Your are an assistant, please help me

    {history}
    User: {input}
    AI Assistant:

vector:
  vector_path: null # Path to the vectorstore to do QA retrieval, e.g. ./vectorstore/db_faiss
  # Set to null to deploy a generic conversational agent
  vector_download: null
  embeddings_path: ./embeddings/all-MiniLM-L6-v2 # Path to embeddings used to generate the vectors, or use directly from HuggingFace: sentence-transformers/all-MiniLM-L6-v2
  embeddings_download: https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/all-MiniLM-L6-v2.zip
  documents_path: ./documents # Path to documents to vectorize
  chunk_size: 500             # Maximum size of chunks, in terms of number of characters
  chunk_overlap: 50           # Overlap in characters between chunks
  chain_type: stuff           # Or: map_reduce, reduce, map_rerank. More details: https://docs.langchain.com/docs/components/chains/index_related_chains
  search_type: similarity     # Or: similarity_score_threshold, mmr. More details: https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore
  return_sources_count: 2     # Number of sources to return when generating an answer
  score_threshold: null       # If using the similarity_score_threshold search type. Between 0 and 1

info:
  title: "Libre Chat"
  version: "0.1.0"
  description: |
    Open source and free chatbot powered by [LangChain](https://python.langchain.com) and [llama.cpp](https://github.com/ggerganov/llama.cpp)
  examples:
  - What is the capital of the Netherlands?
  - Which drugs are approved by the FDA to mitigate Alzheimer symptoms?
  - How can I create a logger with timestamp using python logging?
  favicon: https://raw.github.com/vemonet/libre-chat/main/docs/docs/assets/logo.png
  repository_url: https://github.com/vemonet/libre-chat
  public_url: https://chat.semanticscience.org
  contact:
    name: Vincent Emonet
    email: vincent.emonet@gmail.com
  license_info:
    name: MIT license
    url: https://raw.github.com/vemonet/libre-chat/main/LICENSE.txt
```

Finally start your chat service with:

```bash
docker compose up
```

## 📦️ Usage with pip

This package requires Python >=3.8, simply install it with `pipx` or `pip`:

```bash
pip install libre-chat
```

### ⌨️ Use as a command-line interface

You can easily start a new chat web service including UI and API using your terminal:

```bash
libre-chat start
```

Provide a specific config file:

```bash
libre-chat start config/chat-vectorstore-qa.yml
```

For re-build of the vectorstore:

```bash
libre-chat build --vector vectorstore/db_faiss --documents documents
```

Get a full rundown of the available options with:

```bash
libre-chat --help
```

### 🐍 Use with python

Or you can use this package in python scripts:

```python
import logging

import uvicorn
from libre_chat import ChatConf, ChatEndpoint, Llm

logging.basicConfig(level=logging.getLevelName("INFO"))
conf = ChatConf(
  model_path="./models/mixtral-8x7b-instruct-v0.1.Q2_K.gguf",
  vector_path=None
)
llm = Llm(conf=conf)
print(llm.query("What is the capital of the Netherlands?"))

# Create and deploy a FastAPI app based on your LLM
app = ChatEndpoint(llm=llm, conf=conf)
uvicorn.run(app)
```

## 🤝 Credits

Inspired by:

- https://github.com/kennethleungty/Llama-2-Open-Source-LLM-CPU-Inference
- https://github.com/lm-sys/FastChat
- https://github.com/liltom-eth/llama2-webui

<a href="https://www.flaticon.com/free-icons/llama" title="llama icons">Llama icons created by Freepik - Flaticon</a>
