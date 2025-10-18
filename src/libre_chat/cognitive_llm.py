"""Cognitive LLM implementation with OpenCog integration."""

import os
import threading
import time
from typing import Any, Dict, List, Optional

import torch

from libre_chat.conf import ChatConf, default_conf
from libre_chat.llm import Llm
from libre_chat.moses_evolution import MosesEvolutionEngine
from libre_chat.opencog_integration import (
    AtomSpace,
    CognitiveReasoningChain,
    ECANAttentionAgent,
)
from libre_chat.utils import BOLD, END, log
from libre_chat.vectorstore import DEFAULT_DOCUMENT_LOADERS, build_vectorstore

__all__ = ["CognitiveLlm"]


class CognitiveLlm(Llm):
    """
    Cognitive LLM that integrates OpenCog components for advanced reasoning.
    
    Extends the base Llm class with:
    - AtomSpace for knowledge representation
    - ECAN attention mechanisms
    - Moses evolution for self-improvement
    - Cognitive reasoning chains
    """

    def __init__(
        self,
        conf: Optional[ChatConf] = None,
        model_path: Optional[str] = None,
        model_download: Optional[str] = None,
        vector_path: Optional[str] = None,
        document_loaders: Optional[List[Dict[str, Any]]] = None,
        prompt_variables: Optional[List[str]] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        """Initialize the Cognitive LLM with OpenCog components."""
        
        # Initialize base LLM first
        super().__init__(
            conf=conf,
            model_path=model_path, 
            model_download=model_download,
            vector_path=vector_path,
            document_loaders=document_loaders,
            prompt_variables=prompt_variables,
            prompt_template=prompt_template
        )
        
        # Initialize OpenCog components if enabled
        self.opencog_enabled = self.conf.opencog.enabled
        self.atomspace = None
        self.attention_agent = None
        self.moses_engine = None
        self.cognitive_chain = None
        
        if self.opencog_enabled:
            self._initialize_opencog_components()
        else:
            log.info("🚫 OpenCog integration disabled in configuration")

    def _initialize_opencog_components(self):
        """Initialize all OpenCog cognitive components."""
        try:
            log.info("🧠 Initializing OpenCog cognitive components...")
            
            # Initialize AtomSpace
            self.atomspace = AtomSpace()
            self._populate_initial_knowledge()
            
            # Initialize ECAN Attention Agent
            if self.conf.opencog.attention_agent_enabled:
                self.attention_agent = ECANAttentionAgent(self.atomspace)
                self.attention_agent.attention_bank = self.conf.opencog.attention_bank
                self.attention_agent.attention_focus_boundary = self.conf.opencog.attention_focus_boundary
                self.attention_agent.start()
            
            # Initialize Moses Evolution Engine
            if self.conf.opencog.moses_enabled:
                self.moses_engine = MosesEvolutionEngine(
                    self.atomspace, 
                    population_size=self.conf.opencog.moses_population_size
                )
                self.moses_engine.mutation_rate = self.conf.opencog.moses_mutation_rate
                self.moses_engine.crossover_rate = self.conf.opencog.moses_crossover_rate
                self.moses_engine.elitism_rate = self.conf.opencog.moses_elitism_rate
                
                # Add basic learning objectives
                self._setup_learning_objectives()
                self.moses_engine.start_evolution()
            
            # Initialize Cognitive Reasoning Chain
            if self.conf.opencog.cognitive_reasoning_enabled and self.attention_agent:
                self.cognitive_chain = CognitiveReasoningChain(
                    self.atomspace, 
                    self.attention_agent
                )
            
            log.info("✅ OpenCog cognitive components initialized successfully")
            
        except Exception as e:
            log.error(f"❌ Failed to initialize OpenCog components: {e}")
            self.opencog_enabled = False

    def _populate_initial_knowledge(self):
        """Populate AtomSpace with initial knowledge and concepts."""
        try:
            # Create basic concept nodes
            basic_concepts = [
                "language", "communication", "reasoning", "knowledge", "learning",
                "question", "answer", "context", "memory", "attention", "inference",
                "pattern", "similarity", "concept", "relationship", "entity"
            ]
            
            concept_atoms = {}
            for concept in basic_concepts:
                atom = self.atomspace.create_concept(concept)
                atom.set_truth_value(0.9, 0.8)  # High confidence in basic concepts
                concept_atoms[concept] = atom
            
            # Create basic relationships
            relationships = [
                ("question", "requires", "answer"),
                ("learning", "improves", "knowledge"), 
                ("reasoning", "uses", "knowledge"),
                ("attention", "focuses", "memory"),
                ("pattern", "enables", "inference"),
                ("context", "influences", "answer")
            ]
            
            for subj, pred, obj in relationships:
                if subj in concept_atoms and obj in concept_atoms:
                    predicate = self.atomspace.add_atom("PredicateNode", pred)
                    evaluation = self.atomspace.create_evaluation(
                        predicate,
                        [concept_atoms[subj], concept_atoms[obj]]
                    )
                    evaluation.set_truth_value(0.8, 0.7)
            
            # Create inheritance hierarchies
            inheritances = [
                ("reasoning", "cognitive_process"),
                ("learning", "cognitive_process"),
                ("attention", "cognitive_process"),
                ("question", "communication_act"),
                ("answer", "communication_act")
            ]
            
            for child, parent in inheritances:
                if child in concept_atoms:
                    parent_atom = self.atomspace.create_concept(parent)
                    self.atomspace.create_inheritance(
                        concept_atoms[child], 
                        parent_atom, 
                        strength=0.9
                    )
            
            log.info(f"📚 Populated AtomSpace with {self.atomspace.size()} initial atoms")
            
        except Exception as e:
            log.error(f"Error populating initial knowledge: {e}")

    def _setup_learning_objectives(self):
        """Setup learning objectives for Moses evolution."""
        if not self.moses_engine:
            return
        
        try:
            # Define learning objectives
            objectives = [
                {
                    "description": "Improve question understanding and categorization",
                    "test_cases": [
                        {
                            "input": {"query": "What is the capital of France?"},
                            "expected": {"type": "factual_query", "category": "geography"}
                        },
                        {
                            "input": {"query": "How do I learn Python?"},
                            "expected": {"type": "procedural_query", "category": "education"}
                        }
                    ]
                },
                {
                    "description": "Enhance reasoning about relationships",
                    "test_cases": [
                        {
                            "input": {"concept1": "learning", "concept2": "knowledge"},
                            "expected": {"type": "causal_relation", "strength": 0.8}
                        }
                    ]
                },
                {
                    "description": "Optimize attention allocation for relevant concepts",
                    "test_cases": [
                        {
                            "input": {"query": "machine learning algorithms"},
                            "expected": {"focus_concepts": ["algorithm", "learning", "machine"]}
                        }
                    ]
                }
            ]
            
            for objective in objectives:
                self.moses_engine.add_learning_objective(
                    objective["description"],
                    objective.get("test_cases", [])
                )
            
            log.info(f"📈 Setup {len(objectives)} learning objectives for Moses")
            
        except Exception as e:
            log.error(f"Error setting up learning objectives: {e}")

    def query(
        self,
        prompt: str,
        memory: Any = None,
        config: Optional[Dict[str, Any]] = None,
        instructions: Optional[str] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced query method using cognitive reasoning.
        
        Falls back to base LLM if OpenCog is disabled or fails.
        """
        log.info(f"🧠 Processing cognitive query: {prompt}")
        
        # Use cognitive reasoning if available
        if self.opencog_enabled and self.cognitive_chain:
            try:
                # Extract context for cognitive reasoning
                context = []
                if memory and hasattr(memory, 'chat_memory'):
                    # Extract recent conversation context
                    recent_messages = memory.chat_memory.messages[-6:]  # Last 3 exchanges
                    for msg in recent_messages:
                        if hasattr(msg, 'content'):
                            context.append(msg.content[:200])  # Limit context length
                
                # Perform cognitive reasoning
                cognitive_result = self.cognitive_chain.process_query(prompt, context)
                
                # Enhance with traditional LLM if needed
                if self._should_use_hybrid_approach(cognitive_result):
                    llm_result = super().query(prompt, memory, config, instructions, callbacks)
                    return self._merge_results(cognitive_result, llm_result)
                else:
                    return cognitive_result
                    
            except Exception as e:
                log.error(f"❌ Cognitive reasoning failed: {e}, falling back to base LLM")
                return super().query(prompt, memory, config, instructions, callbacks)
        else:
            # Fall back to base LLM implementation
            return super().query(prompt, memory, config, instructions, callbacks)

    async def aquery(
        self,
        prompt: str,
        memory: Any = None,
        config: Optional[Dict[str, Any]] = None,
        instructions: Optional[str] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced async query method using cognitive reasoning.
        """
        # For now, delegate to synchronous cognitive query
        # In a full implementation, this would be truly async
        return self.query(prompt, memory, config, instructions, callbacks)

    def _should_use_hybrid_approach(self, cognitive_result: Dict[str, Any]) -> bool:
        """Determine if hybrid cognitive+LLM approach should be used."""
        cognitive_state = cognitive_result.get("cognitive_state", {})
        
        # Use hybrid if:
        # - Few conclusions found in cognitive reasoning
        # - Small attentional focus
        # - Low confidence indicators
        
        conclusions_count = cognitive_state.get("conclusions_found", 0)
        focus_size = cognitive_state.get("focus_size", 0)
        
        return conclusions_count < 2 or focus_size < 3

    def _merge_results(
        self, 
        cognitive_result: Dict[str, Any], 
        llm_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge cognitive reasoning with traditional LLM results."""
        
        # Combine the response texts
        cognitive_response = cognitive_result.get("result", "")
        llm_response = llm_result.get("result", "")
        
        merged_response = f"{cognitive_response}\n\n{llm_response}"
        
        # Merge metadata
        merged_result = {
            "result": merged_response,
            "cognitive_state": cognitive_result.get("cognitive_state", {}),
            "attention_summary": cognitive_result.get("attention_summary", {}),
            "source_documents": llm_result.get("source_documents", []),
            "approach": "hybrid_cognitive_llm"
        }
        
        return merged_result

    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current state of cognitive components."""
        if not self.opencog_enabled:
            return {"status": "disabled"}
        
        state = {"status": "enabled"}
        
        if self.atomspace:
            state["atomspace"] = {
                "size": self.atomspace.size(),
                "top_concepts": [
                    str(atom) for atom in self.atomspace.get_atoms_by_importance(limit=5)
                ]
            }
        
        if self.attention_agent:
            focus_atoms = self.attention_agent.get_attentional_focus()
            state["attention"] = {
                "focus_size": len(focus_atoms),
                "focus_concepts": [str(atom) for atom in focus_atoms[:5]]
            }
        
        if self.moses_engine:
            state["evolution"] = self.moses_engine.get_evolution_summary()
        
        return state

    def learn_from_interaction(self, query: str, response: str, feedback: Optional[str] = None):
        """Learn from user interactions to improve cognitive performance."""
        if not self.opencog_enabled or not self.atomspace:
            return
        
        try:
            # Create atoms for the interaction
            query_concept = self.atomspace.create_concept(f"query_{hash(query)}")
            response_concept = self.atomspace.create_concept(f"response_{hash(response)}")
            
            # Create relationship between query and response
            generates_predicate = self.atomspace.add_atom("PredicateNode", "generates")
            self.atomspace.create_evaluation(
                generates_predicate,
                [query_concept, response_concept]
            )
            
            # Update importance based on feedback
            if feedback:
                if "good" in feedback.lower() or "helpful" in feedback.lower():
                    query_concept.update_importance(2.0)
                    response_concept.update_importance(2.0)
                elif "bad" in feedback.lower() or "wrong" in feedback.lower():
                    query_concept.update_importance(-1.0)
                    response_concept.update_importance(-1.0)
            
            # Add learning objective to Moses if available
            if self.moses_engine and feedback:
                self.moses_engine.add_learning_objective(
                    f"Improve responses similar to: {query[:50]}...",
                    [{"input": {"query": query}, "expected": {"quality": feedback}}]
                )
            
            log.debug(f"📚 Learned from interaction: query={query[:30]}...")
            
        except Exception as e:
            log.error(f"Error learning from interaction: {e}")

    def export_cognitive_knowledge(self) -> Optional[Dict[str, Any]]:
        """Export cognitive knowledge for analysis or backup."""
        if not self.opencog_enabled or not self.atomspace:
            return None
        
        try:
            export_data = {
                "atomspace": self.atomspace.export_to_dict(),
                "cognitive_state": self.get_cognitive_state(),
                "timestamp": time.time()
            }
            
            if self.moses_engine:
                export_data["best_program"] = self.moses_engine.export_best_program()
            
            return export_data
            
        except Exception as e:
            log.error(f"Error exporting cognitive knowledge: {e}")
            return None

    def shutdown(self):
        """Properly shutdown cognitive components."""
        try:
            if self.attention_agent:
                self.attention_agent.stop()
            
            if self.moses_engine:
                self.moses_engine.stop_evolution()
            
            log.info("🧠 Cognitive components shut down")
            
        except Exception as e:
            log.error(f"Error shutting down cognitive components: {e}")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during cleanup