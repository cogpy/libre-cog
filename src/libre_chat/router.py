import os
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import werkzeug
from fastapi import APIRouter, Body, File, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.memory import ConversationBufferMemory

from libre_chat.conf import ChatConf, default_conf
from libre_chat.utils import ChatResponse, Prompt, log

__all__ = [
    "ChatRouter",
]

api_responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = {
    200: {
        "description": "Chat response",
        "content": {
            "application/json": {
                "result": "",
                "source_docs": [],
            },
        },
    },
    400: {"description": "Bad Request"},
    422: {"description": "Unprocessable Entity"},
}


@dataclass
class PromptResponse:
    result: str
    source_documents: Optional[List[Any]] = None


class ChatRouter(APIRouter):
    """
    Class to deploy a LLM router with FastAPI.
    """

    def __init__(
        self,
        *args: Any,
        llm: Any,
        path: str = "/prompt",
        conf: Optional[ChatConf] = None,
        examples: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Constructor of the LLM API router with the actual calls
        """
        self.path = path
        self.llm = llm
        self.conf = conf if conf else default_conf
        self.title = self.conf.info.title
        self.description = self.conf.info.description
        self.version = self.conf.info.version
        self.examples = examples if examples else self.conf.info.examples
        example_post = {"prompt": self.examples[0]}

        # Instantiate APIRouter
        super().__init__(
            *args,
            responses=api_responses,
            **kwargs,
        )
        # Create a list to store all connected WebSocket clients
        self.connected_clients: List[WebSocket] = []

        @self.get(
            self.path,
            name="Prompt the LLM",
            description=self.description,
            response_model=PromptResponse,
        )
        def get_prompt(request: Request, prompt: str = self.examples[0]) -> JSONResponse:
            """Send a prompt to the chatbot through HTTP GET operation.

            :param request: The HTTP GET request with a .body()
            :param prompt: Prompt to send to the LLM
            """
            return JSONResponse(self.llm.query(prompt))

        @self.post(
            self.path,
            name="Prompt the LLM",
            description=self.description,
            response_description="Prompt response",
            response_model=PromptResponse,
        )
        def post_prompt(
            request: Request,
            prompt: Prompt = Body(..., example=example_post),
        ) -> JSONResponse:
            """Send a prompt to the chatbot through HTTP POST operation.

            :param request: The HTTP POST request with a .body()
            :param prompt: Prompt to send to the LLM.
            """
            return JSONResponse(self.llm.query(prompt.prompt))

        @self.post(
            "/documents",
            description="""Upload documents to be added to the vectorstore, you can provide a zip file that will be automatically unzipped.""",
            response_description="Operation result",
            response_model={},
            tags=["vectorstore"],
        )
        def upload_documents(
            files: List[UploadFile] = File(...),
            admin_pass: Optional[str] = None,
            # current_user: User = Depends(get_current_user),
        ) -> JSONResponse:
            os.makedirs(self.conf.vector.documents_path, exist_ok=True)
            if self.conf.auth.admin_pass and admin_pass != self.conf.auth.admin_pass:
                raise HTTPException(
                    status_code=403,
                    detail="The admin pass key provided was wrong",
                )
            for uploaded in files:
                if uploaded.filename:  # no cov
                    file_path = werkzeug.utils.safe_join(self.conf.vector.documents_path, uploaded.filename)
                    if file_path is None:
                        raise HTTPException(
                            status_code=403,
                            detail=f"Invalid file name: {uploaded.filename}",
                        )

                    with open(file_path, "wb") as file:
                        file.write(uploaded.file.read())
                    # Check if the uploaded file is a zip file
                    if uploaded.filename.endswith(".zip"):
                        log.info(f"🤐 Unzipping {file_path}")
                        with zipfile.ZipFile(file_path, "r") as zip_ref:
                            zip_ref.extractall(self.conf.vector.documents_path)
                        os.remove(file_path)
            # TODO: add just the uploaded files instead of rebuilding the triplestore
            self.llm.build_vectorstore()
            self.llm.setup_dbqa()
            return JSONResponse(
                {
                    "message": f"Documents uploaded in {self.conf.vector.documents_path}, vectorstore rebuilt."
                }
            )

        # Cognitive endpoints for OpenCog integration
        @self.get(
            "/cognitive/state",
            name="Get Cognitive State", 
            description="Get the current state of the cognitive system including AtomSpace, attention, and evolution status",
            response_model=Dict[str, Any],
            tags=["cognitive"],
        )
        def get_cognitive_state() -> JSONResponse:
            """Get current cognitive state of the system."""
            try:
                if hasattr(self.llm, 'get_cognitive_state'):
                    state = self.llm.get_cognitive_state()
                    return JSONResponse(state)
                else:
                    return JSONResponse({
                        "status": "not_available", 
                        "message": "Cognitive features not enabled"
                    })
            except Exception as e:
                log.error(f"Error getting cognitive state: {e}")
                return JSONResponse({"error": "Internal server error"}, status_code=500)

        @self.get(
            "/cognitive/attention",
            name="Get Attention Focus",
            description="Get atoms currently in attentional focus",
            response_model=Dict[str, Any], 
            tags=["cognitive"],
        )
        def get_attention_focus() -> JSONResponse:
            """Get current attentional focus."""
            try:
                if (hasattr(self.llm, 'attention_agent') and 
                    self.llm.attention_agent is not None):
                    focus_atoms = self.llm.attention_agent.get_attentional_focus()
                    return JSONResponse({
                        "focus_atoms": [str(atom) for atom in focus_atoms],
                        "focus_size": len(focus_atoms),
                        "attention_bank": self.llm.attention_agent.attention_bank
                    })
                else:
                    return JSONResponse({
                        "status": "not_available",
                        "message": "Attention agent not enabled"
                    })
            except Exception as e:
                log.error(f"Error getting attention focus: {e}")
                return JSONResponse({"error": "Internal server error"}, status_code=500)

        @self.get(
            "/cognitive/atomspace",
            name="Get AtomSpace Statistics",
            description="Get statistics about the current AtomSpace",
            response_model=Dict[str, Any],
            tags=["cognitive"],
        )
        def get_atomspace_stats() -> JSONResponse:
            """Get AtomSpace statistics."""
            try:
                if (hasattr(self.llm, 'atomspace') and 
                    self.llm.atomspace is not None):
                    top_atoms = self.llm.atomspace.get_atoms_by_importance(limit=10)
                    return JSONResponse({
                        "total_atoms": self.llm.atomspace.size(),
                        "top_concepts": [
                            {
                                "atom": str(atom),
                                "importance": atom.importance,
                                "truth_value": atom.get_truth_value()
                            }
                            for atom in top_atoms
                        ],
                        "atom_types": list(self.llm.atomspace.atom_types)
                    })
                else:
                    return JSONResponse({
                        "status": "not_available",
                        "message": "AtomSpace not enabled"
                    })
            except Exception as e:
                log.error(f"Error getting atomspace stats: {e}")
                return JSONResponse({"error": "Internal server error"}, status_code=500)

        @self.get(
            "/cognitive/evolution",
            name="Get Evolution Status",
            description="Get status of the Moses evolution engine",
            response_model=Dict[str, Any],
            tags=["cognitive"],
        )
        def get_evolution_status() -> JSONResponse:
            """Get Moses evolution status."""
            try:
                if (hasattr(self.llm, 'moses_engine') and 
                    self.llm.moses_engine is not None):
                    summary = self.llm.moses_engine.get_evolution_summary()
                    return JSONResponse(summary)
                else:
                    return JSONResponse({
                        "status": "not_available",
                        "message": "Moses evolution engine not enabled"
                    })
            except Exception as e:
                log.error(f"Error getting evolution status: {e}")
                return JSONResponse({"error": "Internal server error"}, status_code=500)

        @self.post(
            "/cognitive/learn",
            name="Learn from Interaction",
            description="Provide feedback to help the system learn from interactions",
            response_model=Dict[str, str],
            tags=["cognitive"],
        )
        def learn_from_interaction(
            interaction_data: Dict[str, Any] = Body(..., example={
                "query": "What is machine learning?",
                "response": "Machine learning is a subset of AI...", 
                "feedback": "good explanation"
            })
        ) -> JSONResponse:
            """Learn from user interaction and feedback."""
            try:
                if hasattr(self.llm, 'learn_from_interaction'):
                    query = interaction_data.get("query", "")
                    response = interaction_data.get("response", "")
                    feedback = interaction_data.get("feedback")
                    
                    self.llm.learn_from_interaction(query, response, feedback)
                    
                    return JSONResponse({
                        "message": "Learning from interaction completed",
                        "status": "success"
                    })
                else:
                    return JSONResponse({
                        "status": "not_available",
                        "message": "Learning capability not enabled"
                    })
            except Exception as e:
                log.error(f"Error learning from interaction: {e}")
                return JSONResponse({"error": "Internal server error"}, status_code=500)

        @self.get(
            "/cognitive/export",
            name="Export Cognitive Knowledge",
            description="Export cognitive knowledge for analysis or backup",
            response_model=Dict[str, Any],
            tags=["cognitive"],
        )
        def export_cognitive_knowledge() -> JSONResponse:
            """Export cognitive knowledge."""
            try:
                if hasattr(self.llm, 'export_cognitive_knowledge'):
                    export_data = self.llm.export_cognitive_knowledge()
                    if export_data:
                        return JSONResponse(export_data)
                    else:
                        return JSONResponse({
                            "status": "no_data",
                            "message": "No cognitive data available for export"
                        })
                else:
                    return JSONResponse({
                        "status": "not_available",
                        "message": "Cognitive export not enabled"
                    })
            except Exception as e:
                log.error(f"Error exporting cognitive knowledge: {e}")
                return JSONResponse({"error": "Internal server error"}, status_code=500)

        @self.get(
            "/documents",
            description="""List documents uploaded to the server.""",
            response_description="List of files",
            response_model={},
            tags=["vectorstore"],
        )
        def list_documents(
            admin_pass: Optional[str] = None,
            # Depends(get_current_user)
        ) -> JSONResponse:
            """List all documents in the documents folder."""
            if self.conf.auth.admin_pass and admin_pass != self.conf.auth.admin_pass:
                raise HTTPException(
                    status_code=403,
                    detail="The admin pass key provided was wrong",
                )
            file_list = os.listdir(self.conf.vector.documents_path)
            return JSONResponse({"count": len(file_list), "files": file_list})

        @self.get(
            "/config",
            name="Get Chat configuration",
            description="""Get the Chat web service configuration.""",
            response_description="Chat configuration",
            response_model=ChatConf,
            tags=["configuration"],
        )
        def get_config(
            admin_pass: Optional[str] = None,
        ) -> JSONResponse:
            """Get the Chat web service configuration."""
            if self.conf.auth.admin_pass and admin_pass != self.conf.auth.admin_pass:
                raise HTTPException(
                    status_code=403,
                    detail="The admin pass key provided was wrong",
                )
            return JSONResponse(self.conf.dict())

        @self.post(
            "/config",
            name="Edit Chat configuration",
            description="""Edit the Chat web service configuration.""",
            response_description="Chat configuration",
            response_model=ChatConf,
            tags=["configuration"],
        )
        def post_config(
            request: Request,
            config: ChatConf = Body(..., example=self.conf),
            admin_pass: Optional[str] = None,
        ) -> JSONResponse:
            """Edit the Chat web service configuration."""
            if self.conf.auth.admin_pass and admin_pass != self.conf.auth.admin_pass:
                raise HTTPException(
                    status_code=403,
                    detail="The admin pass key provided was wrong",
                )
            self.conf = config
            # TODO: save new config to disk, and make sure all workers reload the new config
            return JSONResponse(self.conf.dict())

        @self.websocket("/chat")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            self.connected_clients.append(websocket)
            log.info(
                f"🔌 New websocket connection: {len(self.connected_clients)} clients are connected"
            )
            memory = ConversationBufferMemory(ai_prefix="AI Assistant")
            try:
                # Loop to receive messages from the WebSocket client
                while True:
                    data = await websocket.receive_json()

                    start_resp = ChatResponse(sender="bot", message="", type="start")
                    await websocket.send_json(start_resp.dict())

                    resp = await self.llm.aquery(
                        data["prompt"],
                        memory=memory,
                        callbacks=[StreamWebsocketCallback(websocket)],
                    )
                    # chat_history.append((question, resp["result"]))
                    # log.warning("RESULTS!")
                    # log.warning(resp["result"])

                    end_resp = ChatResponse(
                        sender="bot",
                        message=resp["result"],
                        type="end",
                        sources=resp["source_documents"] if "source_documents" in resp else None,
                    )
                    await websocket.send_json(end_resp.model_dump())
            except Exception as e:
                log.error(f"WebSocket error: {e}")
            finally:
                self.connected_clients.remove(websocket)


# https://github.com/langchain-ai/chat-langchain/blob/master/main.py
# class StreamingWebsocketCallbackHandler(AsyncCallbackHandler):
class StreamWebsocketCallback(AsyncCallbackHandler):
    """Callback handler for streaming to websocket.
    Only works with LLMs that support streaming."""

    def __init__(
        self,
        websocket: WebSocket,
    ) -> None:
        """Initialize callback handler."""
        super().__init__()
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        resp = ChatResponse(message=token, sender="bot", type="stream")
        await self.websocket.send_json(resp.model_dump())
