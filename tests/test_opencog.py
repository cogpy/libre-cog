"""Tests for OpenCog integration components."""

import time
import unittest

from libre_chat.opencog_integration import (
    Atom,
    AtomSpace,
    CognitiveReasoningChain,
    ECANAttentionAgent,
    PatternMatcher,
)


class TestAtom(unittest.TestCase):
    """Test Atom class functionality."""
    
    def test_atom_creation(self):
        """Test basic atom creation."""
        atom = Atom("ConceptNode", "test_concept")
        self.assertEqual(atom.atom_type, "ConceptNode")
        self.assertEqual(atom.name, "test_concept")
        self.assertEqual(atom.tv_strength, 1.0)
        self.assertEqual(atom.tv_confidence, 0.5)
    
    def test_atom_truth_value(self):
        """Test truth value operations."""
        atom = Atom("ConceptNode", "test")
        atom.set_truth_value(0.8, 0.9)
        strength, confidence = atom.get_truth_value()
        self.assertEqual(strength, 0.8)
        self.assertEqual(confidence, 0.9)
    
    def test_atom_importance_update(self):
        """Test importance updating."""
        atom = Atom("ConceptNode", "test")
        initial_importance = atom.importance
        atom.update_importance(5.0)
        self.assertEqual(atom.importance, initial_importance + 5.0)
        self.assertEqual(atom.sti, 5.0)
    
    def test_atom_access_tracking(self):
        """Test access count tracking."""
        atom = Atom("ConceptNode", "test")
        initial_count = atom.access_count
        initial_importance = atom.importance
        
        atom.access()
        
        self.assertEqual(atom.access_count, initial_count + 1)
        self.assertGreater(atom.importance, initial_importance)


class TestAtomSpace(unittest.TestCase):
    """Test AtomSpace functionality."""
    
    def setUp(self):
        """Set up test atomspace."""
        self.atomspace = AtomSpace()
    
    def test_atomspace_creation(self):
        """Test atomspace initialization."""
        self.assertEqual(self.atomspace.size(), 0)
        self.assertIn("ConceptNode", self.atomspace.atom_types)
        self.assertIn("InheritanceLink", self.atomspace.atom_types)
    
    def test_add_atom(self):
        """Test adding atoms."""
        atom = self.atomspace.add_atom("ConceptNode", "test_concept")
        self.assertEqual(self.atomspace.size(), 1)
        self.assertIn(atom.id, self.atomspace.atoms)
    
    def test_create_concept(self):
        """Test concept creation."""
        concept = self.atomspace.create_concept("dog")
        self.assertEqual(concept.atom_type, "ConceptNode")
        self.assertEqual(concept.name, "dog")
        
        # Should return existing concept if created again
        same_concept = self.atomspace.create_concept("dog")
        self.assertEqual(concept.id, same_concept.id)
    
    def test_create_inheritance(self):
        """Test inheritance link creation."""
        child = self.atomspace.create_concept("dog")
        parent = self.atomspace.create_concept("animal")
        
        inheritance = self.atomspace.create_inheritance(child, parent, 0.9)
        
        self.assertEqual(inheritance.atom_type, "InheritanceLink")
        self.assertEqual(len(inheritance.outgoing), 2)
        self.assertEqual(inheritance.outgoing[0], child)
        self.assertEqual(inheritance.outgoing[1], parent)
        self.assertEqual(inheritance.tv_strength, 0.9)
    
    def test_create_evaluation(self):
        """Test evaluation link creation."""
        predicate = self.atomspace.add_atom("PredicateNode", "likes")
        arg1 = self.atomspace.create_concept("john")
        arg2 = self.atomspace.create_concept("pizza")
        
        evaluation = self.atomspace.create_evaluation(predicate, [arg1, arg2])
        
        self.assertEqual(evaluation.atom_type, "EvaluationLink")
        self.assertEqual(len(evaluation.outgoing), 2)
        self.assertEqual(evaluation.outgoing[0], predicate)
    
    def test_find_atoms(self):
        """Test finding atoms by criteria."""
        concept1 = self.atomspace.create_concept("dog")
        concept2 = self.atomspace.create_concept("cat")
        predicate = self.atomspace.add_atom("PredicateNode", "likes")
        
        # Find all concepts
        concepts = self.atomspace.find_atoms("ConceptNode")
        self.assertEqual(len(concepts), 2)
        
        # Find specific concept
        dogs = self.atomspace.find_atoms("ConceptNode", "dog")
        self.assertEqual(len(dogs), 1)
        self.assertEqual(dogs[0], concept1)
    
    def test_importance_ranking(self):
        """Test getting atoms by importance."""
        concept1 = self.atomspace.create_concept("important")
        concept2 = self.atomspace.create_concept("less_important")
        
        concept1.update_importance(10.0)
        concept2.update_importance(5.0)
        
        ranked = self.atomspace.get_atoms_by_importance(limit=2)
        self.assertEqual(ranked[0], concept1)
        self.assertEqual(ranked[1], concept2)
    
    def test_importance_decay(self):
        """Test importance decay functionality."""
        concept = self.atomspace.create_concept("test")
        concept.update_importance(10.0)
        initial_importance = concept.importance
        
        self.atomspace.decay_importance(0.1)
        
        self.assertLess(concept.importance, initial_importance)
    
    def test_export_to_dict(self):
        """Test exporting atomspace to dictionary."""
        concept = self.atomspace.create_concept("test")
        concept.update_importance(5.0)
        
        export_data = self.atomspace.export_to_dict()
        
        self.assertIn("atoms", export_data)
        self.assertIn("size", export_data)
        self.assertEqual(export_data["size"], 1)
        self.assertEqual(len(export_data["atoms"]), 1)
        
        atom_data = export_data["atoms"][0]
        self.assertEqual(atom_data["type"], "ConceptNode")
        self.assertEqual(atom_data["name"], "test")
        self.assertEqual(atom_data["importance"], 5.0)


class TestECANAttentionAgent(unittest.TestCase):
    """Test ECAN attention agent."""
    
    def setUp(self):
        """Set up test environment."""
        self.atomspace = AtomSpace()
        self.attention_agent = ECANAttentionAgent(self.atomspace)
    
    def test_attention_agent_creation(self):
        """Test attention agent initialization."""
        self.assertEqual(self.attention_agent.atomspace, self.atomspace)
        self.assertEqual(self.attention_agent.attention_bank, 1000.0)
        self.assertFalse(self.attention_agent.running)
    
    def test_attentional_focus(self):
        """Test attentional focus filtering."""
        concept1 = self.atomspace.create_concept("focused")
        concept2 = self.atomspace.create_concept("unfocused")
        
        # Set high STI for concept1
        concept1.sti = 15.0  # Above boundary
        concept2.sti = 5.0   # Below boundary
        
        focus = self.attention_agent.get_attentional_focus()
        
        self.assertIn(concept1, focus)
        self.assertNotIn(concept2, focus)


class TestPatternMatcher(unittest.TestCase):
    """Test pattern matching functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.atomspace = AtomSpace()
        self.pattern_matcher = PatternMatcher(self.atomspace)
    
    def test_find_pattern(self):
        """Test basic pattern finding."""
        concept = self.atomspace.create_concept("dog")
        
        pattern = {"type": "ConceptNode", "name": "dog"}
        matches = self.pattern_matcher.find_pattern(pattern)
        
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["matched"], concept)
    
    def test_forward_chaining(self):
        """Test forward chaining inference."""
        # Create premise and implication
        premise = self.atomspace.create_concept("premise")
        conclusion = self.atomspace.create_concept("conclusion")
        
        # Create implication: premise -> conclusion
        implication = self.atomspace.add_atom(
            "ImplicationLink", 
            outgoing=[premise, conclusion]
        )
        
        conclusions = self.pattern_matcher.forward_chain([premise])
        
        self.assertIn(conclusion, conclusions)
    
    def test_backward_chaining(self):
        """Test backward chaining inference."""
        goal = self.atomspace.create_concept("goal")
        premise = self.atomspace.create_concept("known_fact")
        premise.set_truth_value(0.9, 0.8)  # Mark as known fact
        
        # Create implication: premise -> goal  
        implication = self.atomspace.add_atom(
            "ImplicationLink",
            outgoing=[premise, goal]
        )
        
        proof_paths = self.pattern_matcher.backward_chain(goal, max_depth=2)
        
        # Should find at least one proof path
        self.assertGreater(len(proof_paths), 0)


class TestCognitiveReasoningChain(unittest.TestCase):
    """Test cognitive reasoning chain."""
    
    def setUp(self):
        """Set up test environment."""
        self.atomspace = AtomSpace()
        self.attention_agent = ECANAttentionAgent(self.atomspace)
        self.reasoning_chain = CognitiveReasoningChain(
            self.atomspace, 
            self.attention_agent
        )
    
    def test_process_query(self):
        """Test query processing."""
        # Add some knowledge
        dog = self.atomspace.create_concept("dog")
        animal = self.atomspace.create_concept("animal")
        self.atomspace.create_inheritance(dog, animal, 0.9)
        
        # Boost importance to get into focus
        dog.update_importance(15.0)
        animal.update_importance(12.0)
        
        result = self.reasoning_chain.process_query(
            "What do you know about dogs?",
            context=["Dogs are animals"]
        )
        
        self.assertIn("result", result)
        self.assertIn("cognitive_state", result)
        self.assertIn("attention_summary", result)
        
        # Check that atomspace grew
        self.assertGreater(self.atomspace.size(), 2)


if __name__ == '__main__':
    unittest.main()