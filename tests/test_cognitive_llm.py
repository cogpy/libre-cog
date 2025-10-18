"""Tests for Cognitive LLM with OpenCog integration."""

import unittest
from unittest.mock import MagicMock, patch

from libre_chat.conf import ChatConf, SettingsOpenCog
from libre_chat.cognitive_llm import CognitiveLlm


class TestCognitiveLlm(unittest.TestCase):
    """Test Cognitive LLM functionality."""
    
    @patch('libre_chat.cognitive_llm.torch')
    @patch('libre_chat.cognitive_llm.build_vectorstore')
    @patch('libre_chat.cognitive_llm.parallel_download')
    def setUp(self, mock_parallel_download, mock_build_vectorstore, mock_torch):
        """Set up test environment."""
        # Mock torch to avoid GPU requirements
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        
        # Create test configuration with OpenCog enabled
        self.conf = ChatConf()
        self.conf.opencog = SettingsOpenCog()
        self.conf.opencog.enabled = True
        self.conf.opencog.attention_agent_enabled = True
        self.conf.opencog.moses_enabled = True
        self.conf.opencog.cognitive_reasoning_enabled = True
        
        # Mock file operations to avoid requiring actual model files
        mock_parallel_download.return_value = None
        mock_build_vectorstore.return_value = None
        
        # Create mock LLM
        with patch('libre_chat.cognitive_llm.LlamaCpp') as mock_llama:
            mock_llama.return_value = MagicMock()
            self.cognitive_llm = CognitiveLlm(conf=self.conf)
    
    def test_cognitive_llm_initialization(self):
        """Test that cognitive LLM initializes properly."""
        self.assertTrue(self.cognitive_llm.opencog_enabled)
        self.assertIsNotNone(self.cognitive_llm.atomspace)
        self.assertIsNotNone(self.cognitive_llm.attention_agent)
        self.assertIsNotNone(self.cognitive_llm.moses_engine)
        self.assertIsNotNone(self.cognitive_llm.cognitive_chain)
    
    def test_cognitive_llm_disabled(self):
        """Test behavior when OpenCog is disabled."""
        # Create configuration with OpenCog disabled
        conf = ChatConf()
        conf.opencog.enabled = False
        
        with patch('libre_chat.cognitive_llm.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.device.return_value = "cpu"
            
            with patch('libre_chat.cognitive_llm.parallel_download'):
                with patch('libre_chat.cognitive_llm.LlamaCpp'):
                    cognitive_llm = CognitiveLlm(conf=conf)
        
        self.assertFalse(cognitive_llm.opencog_enabled)
        self.assertIsNone(cognitive_llm.atomspace)
        self.assertIsNone(cognitive_llm.attention_agent)
        self.assertIsNone(cognitive_llm.moses_engine)
        self.assertIsNone(cognitive_llm.cognitive_chain)
    
    def test_get_cognitive_state(self):
        """Test getting cognitive state."""
        state = self.cognitive_llm.get_cognitive_state()
        
        self.assertEqual(state["status"], "enabled")
        self.assertIn("atomspace", state)
        self.assertIn("attention", state)
        self.assertIn("evolution", state)
    
    def test_get_cognitive_state_disabled(self):
        """Test getting cognitive state when disabled."""
        self.cognitive_llm.opencog_enabled = False
        
        state = self.cognitive_llm.get_cognitive_state()
        
        self.assertEqual(state["status"], "disabled")
    
    def test_learn_from_interaction(self):
        """Test learning from user interactions."""
        query = "What is machine learning?"
        response = "Machine learning is a type of AI..."
        feedback = "good explanation"
        
        # Should not raise an exception
        self.cognitive_llm.learn_from_interaction(query, response, feedback)
        
        # Should have added concepts to atomspace
        query_concepts = self.cognitive_llm.atomspace.find_atoms(
            "ConceptNode", 
            f"query_{hash(query)}"
        )
        self.assertEqual(len(query_concepts), 1)
    
    def test_learn_from_interaction_disabled(self):
        """Test learning when OpenCog is disabled."""
        self.cognitive_llm.opencog_enabled = False
        
        # Should not raise an exception even when disabled
        self.cognitive_llm.learn_from_interaction(
            "test query", 
            "test response", 
            "test feedback"
        )
    
    def test_export_cognitive_knowledge(self):
        """Test exporting cognitive knowledge."""
        export_data = self.cognitive_llm.export_cognitive_knowledge()
        
        self.assertIsNotNone(export_data)
        self.assertIn("atomspace", export_data)
        self.assertIn("cognitive_state", export_data)
        self.assertIn("timestamp", export_data)
    
    def test_export_cognitive_knowledge_disabled(self):
        """Test exporting when OpenCog is disabled."""
        self.cognitive_llm.opencog_enabled = False
        
        export_data = self.cognitive_llm.export_cognitive_knowledge()
        
        self.assertIsNone(export_data)
    
    @patch('libre_chat.cognitive_llm.time.time', return_value=1234567890)
    def test_cognitive_query_processing(self, mock_time):
        """Test cognitive query processing."""
        # Mock the cognitive chain to return a result
        mock_result = {
            "result": "Cognitive response based on reasoning",
            "cognitive_state": {
                "atomspace_size": 10,
                "conclusions_found": 2,
                "focus_size": 3
            },
            "attention_summary": {
                "top_concepts": ["concept1", "concept2"]
            }
        }
        
        self.cognitive_llm.cognitive_chain.process_query = MagicMock(return_value=mock_result)
        
        result = self.cognitive_llm.query("What is artificial intelligence?")
        
        self.assertIn("result", result)
        self.assertIn("cognitive_state", result)
        self.assertIn("attention_summary", result)
        
        # Should have called cognitive processing
        self.cognitive_llm.cognitive_chain.process_query.assert_called_once()
    
    def test_hybrid_approach_decision(self):
        """Test decision to use hybrid cognitive+LLM approach."""
        # Result with few conclusions should trigger hybrid approach
        low_conclusion_result = {
            "result": "Limited cognitive response",
            "cognitive_state": {
                "conclusions_found": 1,
                "focus_size": 2
            }
        }
        
        should_use_hybrid = self.cognitive_llm._should_use_hybrid_approach(low_conclusion_result)
        self.assertTrue(should_use_hybrid)
        
        # Result with many conclusions should not trigger hybrid
        high_conclusion_result = {
            "result": "Rich cognitive response", 
            "cognitive_state": {
                "conclusions_found": 5,
                "focus_size": 8
            }
        }
        
        should_use_hybrid = self.cognitive_llm._should_use_hybrid_approach(high_conclusion_result)
        self.assertFalse(should_use_hybrid)
    
    def test_merge_results(self):
        """Test merging cognitive and LLM results."""
        cognitive_result = {
            "result": "Cognitive reasoning shows...",
            "cognitive_state": {"conclusions_found": 2},
            "attention_summary": {"top_concepts": ["AI"]}
        }
        
        llm_result = {
            "result": "Traditional LLM response...",
            "source_documents": [{"content": "doc1"}]
        }
        
        merged = self.cognitive_llm._merge_results(cognitive_result, llm_result)
        
        self.assertIn("Cognitive reasoning shows", merged["result"])
        self.assertIn("Traditional LLM response", merged["result"])
        self.assertEqual(merged["cognitive_state"], cognitive_result["cognitive_state"])
        self.assertEqual(merged["source_documents"], llm_result["source_documents"])
        self.assertEqual(merged["approach"], "hybrid_cognitive_llm")
    
    def test_initial_knowledge_population(self):
        """Test that initial knowledge is populated in AtomSpace."""
        # AtomSpace should have basic concepts
        concepts = self.cognitive_llm.atomspace.find_atoms("ConceptNode")
        
        # Should have at least some basic concepts
        self.assertGreater(len(concepts), 0)
        
        # Should have specific concepts like "language", "learning", etc.
        concept_names = [atom.name for atom in concepts]
        self.assertIn("language", concept_names)
        self.assertIn("learning", concept_names)
        self.assertIn("reasoning", concept_names)
    
    def test_moses_learning_objectives_setup(self):
        """Test that Moses learning objectives are properly set up."""
        # Should have added learning objectives
        behaviors = self.cognitive_llm.moses_engine.fitness_function.target_behaviors
        self.assertGreater(len(behaviors), 0)
        
        # Should have test cases
        test_cases = self.cognitive_llm.moses_engine.fitness_function.test_cases
        self.assertGreater(len(test_cases), 0)
    
    def test_shutdown(self):
        """Test proper shutdown of cognitive components."""
        # Mock the component shutdown methods
        self.cognitive_llm.attention_agent.stop = MagicMock()
        self.cognitive_llm.moses_engine.stop_evolution = MagicMock()
        
        self.cognitive_llm.shutdown()
        
        # Should have called shutdown on components
        self.cognitive_llm.attention_agent.stop.assert_called_once()
        self.cognitive_llm.moses_engine.stop_evolution.assert_called_once()


class TestCognitiveLlmErrorHandling(unittest.TestCase):
    """Test error handling in Cognitive LLM."""
    
    @patch('libre_chat.cognitive_llm.torch')
    def test_opencog_initialization_failure(self, mock_torch):
        """Test graceful handling of OpenCog initialization failure."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        
        # Create config with OpenCog enabled
        conf = ChatConf()
        conf.opencog.enabled = True
        
        # Mock AtomSpace to raise an exception
        with patch('libre_chat.cognitive_llm.AtomSpace') as mock_atomspace:
            mock_atomspace.side_effect = Exception("Mock initialization failure")
            
            with patch('libre_chat.cognitive_llm.parallel_download'):
                with patch('libre_chat.cognitive_llm.LlamaCpp'):
                    cognitive_llm = CognitiveLlm(conf=conf)
        
        # Should have fallen back to disabled state
        self.assertFalse(cognitive_llm.opencog_enabled)
        self.assertIsNone(cognitive_llm.atomspace)
    
    def test_query_fallback(self):
        """Test fallback to base LLM when cognitive processing fails."""
        with patch('libre_chat.cognitive_llm.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.device.return_value = "cpu"
            
            conf = ChatConf()
            conf.opencog.enabled = False  # Disabled OpenCog
            
            with patch('libre_chat.cognitive_llm.parallel_download'):
                with patch('libre_chat.cognitive_llm.LlamaCpp') as mock_llama:
                    mock_llama_instance = MagicMock()
                    mock_llama.return_value = mock_llama_instance
                    
                    cognitive_llm = CognitiveLlm(conf=conf)
        
        # Mock the base LLM query method
        base_result = {"result": "Base LLM response"}
        
        with patch.object(cognitive_llm.__class__.__bases__[0], 'query', return_value=base_result):
            result = cognitive_llm.query("Test query")
        
        self.assertEqual(result, base_result)


if __name__ == '__main__':
    unittest.main()