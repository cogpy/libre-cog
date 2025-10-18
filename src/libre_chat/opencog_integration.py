"""OpenCog integration module for cognitive chatbot functionality."""

import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from libre_chat.utils import log


class Atom:
    """Basic Atom implementation for OpenCog-style knowledge representation."""
    
    def __init__(self, atom_type: str, name: str = None, outgoing: List['Atom'] = None):
        self.id = str(uuid.uuid4())
        self.atom_type = atom_type
        self.name = name or ""
        self.outgoing = outgoing or []
        self.incoming: List['Atom'] = []
        self.tv_strength = 1.0  # Truth value strength
        self.tv_confidence = 0.5  # Truth value confidence
        self.importance = 0.0  # ECAN importance
        self.sti = 0.0  # Short-term importance
        self.lti = 0.0  # Long-term importance
        self.created_at = time.time()
        self.access_count = 0
        
    def get_truth_value(self) -> Tuple[float, float]:
        """Get atom's truth value as (strength, confidence)."""
        return (self.tv_strength, self.tv_confidence)
    
    def set_truth_value(self, strength: float, confidence: float):
        """Set atom's truth value."""
        self.tv_strength = max(0.0, min(1.0, strength))
        self.tv_confidence = max(0.0, min(1.0, confidence))
    
    def update_importance(self, delta: float):
        """Update atom's importance value."""
        self.importance += delta
        self.sti = max(0.0, self.importance)
    
    def access(self):
        """Mark atom as accessed for attention tracking."""
        self.access_count += 1
        self.update_importance(0.1)  # Small boost for being accessed
    
    def __str__(self):
        if self.name:
            return f"{self.atom_type}:{self.name}"
        elif self.outgoing:
            outgoing_str = " ".join(str(atom) for atom in self.outgoing)
            return f"{self.atom_type}({outgoing_str})"
        else:
            return f"{self.atom_type}()"
    
    def __hash__(self):
        return hash((self.atom_type, self.name, tuple(self.outgoing)))


class AtomSpace:
    """
    AtomSpace implementation for managing cognitive knowledge representation.
    
    This serves as the core memory system for the OpenCog cognitive chatbot,
    storing concepts, relationships, and learned patterns.
    """
    
    def __init__(self):
        self.atoms: Dict[str, Atom] = {}
        self.atom_types: Set[str] = {
            "ConceptNode", "PredicateNode", "VariableNode", "NumberNode",
            "ListLink", "SetLink", "EvaluationLink", "ImplicationLink", 
            "InheritanceLink", "SimilarityLink", "ContextLink"
        }
        self.lock = threading.RLock()
        log.info("🧠 AtomSpace initialized for cognitive memory management")
    
    def add_atom(self, atom_type: str, name: str = None, outgoing: List[Atom] = None) -> Atom:
        """Add an atom to the AtomSpace."""
        with self.lock:
            atom = Atom(atom_type, name, outgoing)
            
            # Update incoming sets for outgoing atoms
            if outgoing:
                for out_atom in outgoing:
                    out_atom.incoming.append(atom)
            
            self.atoms[atom.id] = atom
            log.debug(f"Added atom: {atom}")
            return atom
    
    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Retrieve an atom by ID."""
        atom = self.atoms.get(atom_id)
        if atom:
            atom.access()  # Track access for attention
        return atom
    
    def find_atoms(self, atom_type: str = None, name: str = None) -> List[Atom]:
        """Find atoms matching criteria."""
        with self.lock:
            result = []
            for atom in self.atoms.values():
                if atom_type and atom.atom_type != atom_type:
                    continue
                if name and atom.name != name:
                    continue
                result.append(atom)
                atom.access()  # Track access
            return result
    
    def create_concept(self, name: str) -> Atom:
        """Create a concept node."""
        existing = self.find_atoms("ConceptNode", name)
        if existing:
            return existing[0]
        return self.add_atom("ConceptNode", name)
    
    def create_evaluation(self, predicate: Atom, arguments: List[Atom]) -> Atom:
        """Create an evaluation link."""
        list_link = self.add_atom("ListLink", outgoing=arguments)
        return self.add_atom("EvaluationLink", outgoing=[predicate, list_link])
    
    def create_inheritance(self, child: Atom, parent: Atom, strength: float = 0.8) -> Atom:
        """Create an inheritance link."""
        inheritance = self.add_atom("InheritanceLink", outgoing=[child, parent])
        inheritance.set_truth_value(strength, 0.9)
        return inheritance
    
    def get_atoms_by_importance(self, limit: int = None) -> List[Atom]:
        """Get atoms sorted by importance (for ECAN attention)."""
        with self.lock:
            sorted_atoms = sorted(
                self.atoms.values(), 
                key=lambda a: a.importance, 
                reverse=True
            )
            return sorted_atoms[:limit] if limit else sorted_atoms
    
    def decay_importance(self, decay_rate: float = 0.01):
        """Decay importance of all atoms (ECAN mechanism)."""
        with self.lock:
            for atom in self.atoms.values():
                atom.importance *= (1.0 - decay_rate)
                atom.sti *= (1.0 - decay_rate)
    
    def size(self) -> int:
        """Get number of atoms in the AtomSpace."""
        return len(self.atoms)
    
    def clear(self):
        """Clear all atoms from the AtomSpace."""
        with self.lock:
            self.atoms.clear()
            log.info("AtomSpace cleared")
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export AtomSpace to dictionary format."""
        with self.lock:
            return {
                "atoms": [
                    {
                        "id": atom.id,
                        "type": atom.atom_type,
                        "name": atom.name,
                        "outgoing": [out.id for out in atom.outgoing],
                        "tv_strength": atom.tv_strength,
                        "tv_confidence": atom.tv_confidence,
                        "importance": atom.importance
                    }
                    for atom in self.atoms.values()
                ],
                "size": len(self.atoms)
            }


class ECANAttentionAgent:
    """
    Economic Cognitive Attention Network (ECAN) attention agent.
    
    Manages attention allocation and spreading in the cognitive system.
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.attention_bank = 1000.0  # Total attention currency
        self.min_sti = 0.0
        self.max_sti = 100.0
        self.attention_focus_boundary = 10.0
        self.running = False
        self.thread = None
        log.info("🎯 ECAN Attention Agent initialized")
    
    def start(self):
        """Start the attention agent in a separate thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_attention_cycle)
            self.thread.daemon = True
            self.thread.start()
            log.info("🎯 ECAN Attention Agent started")
    
    def stop(self):
        """Stop the attention agent."""
        self.running = False
        if self.thread:
            self.thread.join()
        log.info("🎯 ECAN Attention Agent stopped")
    
    def _run_attention_cycle(self):
        """Main attention cycle loop."""
        while self.running:
            try:
                self._update_attention()
                self._spread_attention()
                self.atomspace.decay_importance(0.005)  # Small decay each cycle
                time.sleep(1.0)  # Run cycle every second
            except Exception as e:
                log.error(f"Error in attention cycle: {e}")
    
    def _update_attention(self):
        """Update STI values based on recent activity."""
        high_importance_atoms = self.atomspace.get_atoms_by_importance(limit=20)
        
        for atom in high_importance_atoms:
            # Boost recently accessed atoms
            if atom.access_count > 0:
                boost = min(5.0, atom.access_count * 0.5)
                atom.sti = min(self.max_sti, atom.sti + boost)
                atom.access_count = 0  # Reset count
    
    def _spread_attention(self):
        """Spread attention from high-STI atoms to connected atoms."""
        attentional_focus = [
            atom for atom in self.atomspace.atoms.values() 
            if atom.sti > self.attention_focus_boundary
        ]
        
        for focal_atom in attentional_focus:
            # Spread to outgoing atoms
            for out_atom in focal_atom.outgoing:
                spread_amount = focal_atom.sti * 0.1
                out_atom.sti = min(self.max_sti, out_atom.sti + spread_amount)
            
            # Spread to incoming atoms
            for inc_atom in focal_atom.incoming:
                spread_amount = focal_atom.sti * 0.05
                inc_atom.sti = min(self.max_sti, inc_atom.sti + spread_amount)
    
    def get_attentional_focus(self) -> List[Atom]:
        """Get atoms currently in attentional focus."""
        return [
            atom for atom in self.atomspace.atoms.values()
            if atom.sti > self.attention_focus_boundary
        ]


class PatternMatcher:
    """
    Basic pattern matching for cognitive reasoning.
    
    Implements forward and backward chaining inference.
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        log.info("🔍 Pattern Matcher initialized")
    
    def find_pattern(self, pattern: Dict[str, Any]) -> List[Dict[str, Atom]]:
        """Find atoms matching a given pattern."""
        # Simplified pattern matching - could be extended with unification
        matches = []
        
        if pattern.get("type") == "ConceptNode" and pattern.get("name"):
            atoms = self.atomspace.find_atoms("ConceptNode", pattern["name"])
            for atom in atoms:
                matches.append({"matched": atom})
        
        return matches
    
    def forward_chain(self, premises: List[Atom]) -> List[Atom]:
        """Perform forward chaining inference."""
        conclusions = []
        
        # Simple forward chaining: find implications where premises match
        for atom in self.atomspace.atoms.values():
            if atom.atom_type == "ImplicationLink" and len(atom.outgoing) == 2:
                antecedent, consequent = atom.outgoing
                # Check if antecedent matches any premise
                if antecedent in premises:
                    conclusions.append(consequent)
                    consequent.update_importance(1.0)  # Boost inferred atoms
        
        return conclusions
    
    def backward_chain(self, goal: Atom, max_depth: int = 3) -> List[List[Atom]]:
        """Perform backward chaining to find proof paths."""
        proof_paths = []
        
        def search_backwards(current_goal: Atom, path: List[Atom], depth: int):
            if depth >= max_depth:
                return
            
            # Look for implications that conclude the current goal
            for atom in self.atomspace.atoms.values():
                if (atom.atom_type == "ImplicationLink" and 
                    len(atom.outgoing) == 2 and 
                    atom.outgoing[1] == current_goal):
                    
                    new_goal = atom.outgoing[0]
                    new_path = path + [atom, new_goal]
                    
                    # Check if new goal is already known (base case)
                    if self._is_known_fact(new_goal):
                        proof_paths.append(new_path)
                    else:
                        search_backwards(new_goal, new_path, depth + 1)
        
        search_backwards(goal, [goal], 0)
        return proof_paths
    
    def _is_known_fact(self, atom: Atom) -> bool:
        """Check if an atom is a known fact (high truth value)."""
        return atom.tv_strength > 0.7 and atom.tv_confidence > 0.6


class CognitiveReasoningChain:
    """
    Cognitive reasoning chain that integrates OpenCog components.
    
    Replaces/extends LangChain functionality with cognitive reasoning.
    """
    
    def __init__(self, atomspace: AtomSpace, attention_agent: ECANAttentionAgent):
        self.atomspace = atomspace
        self.attention_agent = attention_agent
        self.pattern_matcher = PatternMatcher(atomspace)
        log.info("🧠 Cognitive Reasoning Chain initialized")
    
    def process_query(self, query: str, context: List[str] = None) -> Dict[str, Any]:
        """Process a query using cognitive reasoning."""
        log.info(f"🧠 Processing cognitive query: {query}")
        
        # Create concept for the query
        query_concept = self.atomspace.create_concept(f"query_{hash(query)}")
        query_concept.update_importance(10.0)  # High importance for current query
        
        # Add context concepts if provided
        context_concepts = []
        if context:
            for ctx in context:
                ctx_concept = self.atomspace.create_concept(f"context_{hash(ctx)}")
                ctx_concept.update_importance(5.0)
                context_concepts.append(ctx_concept)
        
        # Get attentional focus to guide reasoning
        focus_atoms = self.attention_agent.get_attentional_focus()
        
        # Perform forward chaining from focused concepts
        premises = focus_atoms + context_concepts + [query_concept]
        conclusions = self.pattern_matcher.forward_chain(premises)
        
        # Attempt backward chaining for goal-directed reasoning
        proof_paths = self.pattern_matcher.backward_chain(query_concept)
        
        # Generate response based on reasoning results
        response_parts = []
        if conclusions:
            response_parts.append("Based on my knowledge, I can infer:")
            for conclusion in conclusions[:3]:  # Limit to top 3
                response_parts.append(f"- {conclusion}")
        
        if proof_paths:
            response_parts.append("I found reasoning paths for your query:")
            for i, path in enumerate(proof_paths[:2]):  # Limit to top 2
                response_parts.append(f"Path {i+1}: {' -> '.join(str(atom) for atom in path)}")
        
        if not response_parts:
            response_parts = ["I need to learn more about this topic to provide a better answer."]
        
        # Update atomspace with new relationships
        self._learn_from_interaction(query, context, conclusions)
        
        return {
            "result": "\n".join(response_parts),
            "cognitive_state": {
                "atomspace_size": self.atomspace.size(),
                "focus_size": len(focus_atoms),
                "conclusions_found": len(conclusions),
                "proof_paths": len(proof_paths)
            },
            "attention_summary": {
                "top_concepts": [str(atom) for atom in focus_atoms[:5]]
            }
        }
    
    def _learn_from_interaction(self, query: str, context: List[str], conclusions: List[Atom]):
        """Learn new relationships from the interaction."""
        try:
            # Create relationships between query and context
            query_concept = self.atomspace.find_atoms("ConceptNode", f"query_{hash(query)}")
            if query_concept and context:
                for ctx in context:
                    ctx_concepts = self.atomspace.find_atoms("ConceptNode", f"context_{hash(ctx)}")
                    if ctx_concepts:
                        # Create association between query and context
                        similarity_link = self.atomspace.add_atom(
                            "SimilarityLink",
                            outgoing=[query_concept[0], ctx_concepts[0]]
                        )
                        similarity_link.set_truth_value(0.6, 0.7)
            
            # Strengthen conclusions
            for conclusion in conclusions:
                conclusion.set_truth_value(
                    min(1.0, conclusion.tv_strength + 0.1),
                    min(1.0, conclusion.tv_confidence + 0.05)
                )
        except Exception as e:
            log.error(f"Error in learning from interaction: {e}")