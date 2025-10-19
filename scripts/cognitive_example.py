#!/usr/bin/env python3
"""
Example script demonstrating OpenCog cognitive chatbot capabilities.

This script shows how to:
1. Initialize a cognitive chatbot with OpenCog components
2. Interact with the AtomSpace knowledge representation
3. Monitor attention and evolution processes
4. Use cognitive reasoning for queries
"""

import logging
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from libre_chat.conf import ChatConf, SettingsOpenCog
    from libre_chat.cognitive_llm import CognitiveLlm
    from libre_chat.opencog_integration import AtomSpace, ECANAttentionAgent
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please install libre-chat dependencies")
    exit(1)


def create_cognitive_config() -> ChatConf:
    """Create a configuration for the cognitive chatbot."""
    conf = ChatConf()
    
    # Enable OpenCog features
    conf.opencog = SettingsOpenCog()
    conf.opencog.enabled = True
    conf.opencog.attention_agent_enabled = True
    conf.opencog.moses_enabled = True
    conf.opencog.cognitive_reasoning_enabled = True
    
    # Smaller settings for demo
    conf.opencog.moses_population_size = 20
    conf.opencog.attention_bank = 500.0
    
    # Basic LLM settings (mock for demo)
    conf.llm.model_path = "./mock_model.gguf"  # Would be real model in practice
    conf.llm.temperature = 0.1
    conf.llm.max_new_tokens = 512
    
    logger.info("✅ Created cognitive configuration")
    return conf


def demonstrate_atomspace_operations(atomspace: AtomSpace) -> None:
    """Demonstrate basic AtomSpace operations."""
    logger.info("\n🧠 Demonstrating AtomSpace Operations")
    
    # Create some concepts
    ai = atomspace.create_concept("artificial_intelligence")
    ml = atomspace.create_concept("machine_learning")
    dl = atomspace.create_concept("deep_learning")
    
    # Create hierarchical relationships
    ai_ml_inheritance = atomspace.create_inheritance(ml, ai, strength=0.9)
    ml_dl_inheritance = atomspace.create_inheritance(dl, ml, strength=0.8)
    
    # Create some evaluations (facts)
    predicate = atomspace.add_atom("PredicateNode", "is_subset_of")
    ml_subset_eval = atomspace.create_evaluation(predicate, [ml, ai])
    ml_subset_eval.set_truth_value(0.95, 0.9)
    
    logger.info(f"Created {atomspace.size()} atoms in AtomSpace")
    
    # Show important concepts
    top_concepts = atomspace.get_atoms_by_importance(limit=5)
    logger.info("Top concepts by importance:")
    for atom in top_concepts:
        logger.info(f"  - {atom} (importance: {atom.importance:.2f})")


def demonstrate_attention_mechanisms(attention_agent: ECANAttentionAgent) -> None:
    """Demonstrate ECAN attention mechanisms."""
    logger.info("\n🎯 Demonstrating Attention Mechanisms")
    
    # Get initial attentional focus
    focus_atoms = attention_agent.get_attentional_focus()
    logger.info(f"Current attentional focus: {len(focus_atoms)} atoms")
    
    if focus_atoms:
        for atom in focus_atoms[:3]:  # Show first 3
            logger.info(f"  - {atom} (STI: {atom.sti:.2f})")
    
    # Simulate attention spread by accessing atoms
    atomspace = attention_agent.atomspace
    ai_concepts = atomspace.find_atoms("ConceptNode", "artificial_intelligence")
    if ai_concepts:
        ai_concept = ai_concepts[0]
        logger.info(f"Accessing concept: {ai_concept}")
        ai_concept.access()  # This boosts importance
        
        # Wait a moment for attention cycle to process
        time.sleep(2)
        
        new_focus = attention_agent.get_attentional_focus()
        logger.info(f"Attentional focus after access: {len(new_focus)} atoms")


def demonstrate_cognitive_reasoning(cognitive_llm: CognitiveLlm) -> None:
    """Demonstrate cognitive reasoning capabilities."""
    logger.info("\n🧠 Demonstrating Cognitive Reasoning")
    
    # Example queries that should trigger different reasoning mechanisms
    queries = [
        "What is the relationship between machine learning and AI?",
        "How are deep learning and machine learning connected?",
        "What can you infer about artificial intelligence?",
        "Tell me about the concepts you're currently focusing on."
    ]
    
    for query in queries:
        logger.info(f"\n📝 Query: {query}")
        
        try:
            # Process query through cognitive reasoning
            result = cognitive_llm.query(query)
            
            # Show results
            response = result.get("result", "No response")
            logger.info(f"Response: {response[:200]}...")
            
            # Show cognitive state
            cognitive_state = result.get("cognitive_state", {})
            if cognitive_state:
                logger.info(f"Cognitive State:")
                logger.info(f"  - AtomSpace size: {cognitive_state.get('atomspace_size', 0)}")
                logger.info(f"  - Focus size: {cognitive_state.get('focus_size', 0)}")
                logger.info(f"  - Conclusions found: {cognitive_state.get('conclusions_found', 0)}")
            
            # Show attention summary
            attention_summary = result.get("attention_summary", {})
            top_concepts = attention_summary.get("top_concepts", [])
            if top_concepts:
                logger.info(f"  - Current focus: {', '.join(top_concepts[:3])}")
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
        
        time.sleep(1)  # Brief pause between queries


def demonstrate_learning(cognitive_llm: CognitiveLlm) -> None:
    """Demonstrate learning from interactions."""
    logger.info("\n📚 Demonstrating Learning Capabilities")
    
    # Simulate learning from user feedback
    learning_examples = [
        {
            "query": "What is machine learning?",
            "response": "Machine learning is a subset of AI that enables computers to learn patterns from data.",
            "feedback": "good explanation"
        },
        {
            "query": "How does deep learning work?", 
            "response": "Deep learning uses neural networks with multiple layers to process information.",
            "feedback": "clear and helpful"
        }
    ]
    
    for example in learning_examples:
        cognitive_llm.learn_from_interaction(
            example["query"],
            example["response"], 
            example["feedback"]
        )
        logger.info(f"✅ Learned from: {example['query'][:40]}...")
    
    # Show updated AtomSpace state
    state = cognitive_llm.get_cognitive_state()
    atomspace_info = state.get("atomspace", {})
    logger.info(f"AtomSpace now contains {atomspace_info.get('size', 0)} atoms")


def demonstrate_evolution_monitoring(cognitive_llm: CognitiveLlm) -> None:
    """Demonstrate Moses evolution monitoring."""
    logger.info("\n🧬 Demonstrating Evolution Monitoring")
    
    if not cognitive_llm.moses_engine:
        logger.warning("Moses evolution engine not available")
        return
    
    # Get evolution status
    evolution_state = cognitive_llm.moses_engine.get_evolution_summary()
    
    logger.info(f"Evolution Status:")
    logger.info(f"  - Status: {evolution_state.get('status', 'unknown')}")
    logger.info(f"  - Generation: {evolution_state.get('generation', 0)}")
    logger.info(f"  - Population size: {evolution_state.get('population_size', 0)}")
    logger.info(f"  - Best fitness: {evolution_state.get('best_fitness', 0.0):.3f}")
    
    # Show top programs if available
    top_programs = evolution_state.get('top_programs', [])
    if top_programs:
        logger.info("Top evolved programs:")
        for i, program in enumerate(top_programs):
            expr = program.get('expression', '')[:50]
            fitness = program.get('fitness', 0.0)
            logger.info(f"  {i+1}. {expr}... (fitness: {fitness:.3f})")


def export_cognitive_knowledge(cognitive_llm: CognitiveLlm) -> None:
    """Demonstrate knowledge export capabilities."""
    logger.info("\n💾 Exporting Cognitive Knowledge")
    
    try:
        export_data = cognitive_llm.export_cognitive_knowledge()
        
        if export_data:
            atomspace_data = export_data.get("atomspace", {})
            logger.info(f"Exported AtomSpace with {atomspace_data.get('size', 0)} atoms")
            
            # Show some statistics
            atoms = atomspace_data.get("atoms", [])
            if atoms:
                atom_types = {}
                for atom in atoms:
                    atom_type = atom.get("type", "Unknown")
                    atom_types[atom_type] = atom_types.get(atom_type, 0) + 1
                
                logger.info("Atom type distribution:")
                for atom_type, count in sorted(atom_types.items()):
                    logger.info(f"  - {atom_type}: {count}")
        else:
            logger.warning("No cognitive data available for export")
            
    except Exception as e:
        logger.error(f"Error exporting cognitive knowledge: {e}")


def main():
    """Main demonstration function."""
    logger.info("🚀 Starting OpenCog Cognitive Chatbot Demo")
    
    try:
        # Create configuration
        conf = create_cognitive_config()
        
        # Note: In a real deployment, you would need actual model files
        # For this demo, we'll mock the LLM initialization
        logger.info("⚠️  Note: This demo mocks LLM initialization for demonstration purposes")
        logger.info("   In practice, you would need actual model files and full dependencies")
        
        # Initialize cognitive chatbot (this would fail without real model files)
        # cognitive_llm = CognitiveLlm(conf=conf)
        
        # Instead, let's demonstrate the individual components
        logger.info("\n🧩 Demonstrating Individual Components")
        
        # Create AtomSpace directly for demonstration
        atomspace = AtomSpace()
        demonstrate_atomspace_operations(atomspace)
        
        # Create attention agent
        attention_agent = ECANAttentionAgent(atomspace)
        attention_agent.start()
        
        try:
            demonstrate_attention_mechanisms(attention_agent)
            
            # Wait a bit for attention cycles
            logger.info("⏱️  Waiting for attention cycles...")
            time.sleep(3)
            
            # Show final state
            logger.info(f"\n📊 Final AtomSpace Statistics:")
            logger.info(f"  - Total atoms: {atomspace.size()}")
            
            focus_atoms = attention_agent.get_attentional_focus()
            logger.info(f"  - Atoms in focus: {len(focus_atoms)}")
            
            top_atoms = atomspace.get_atoms_by_importance(limit=3)
            logger.info(f"  - Most important atoms:")
            for atom in top_atoms:
                logger.info(f"    • {atom} (importance: {atom.importance:.2f})")
        
        finally:
            attention_agent.stop()
        
        logger.info("\n✅ Demo completed successfully!")
        logger.info("\n📖 To run with full cognitive features:")
        logger.info("   1. Install required dependencies (torch, langchain, etc.)")
        logger.info("   2. Download an LLM model (e.g., Mixtral)")
        logger.info("   3. Create a proper configuration file")
        logger.info("   4. Run: libre-chat start config/chat-opencog-cognitive.yml")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        logger.error("This is expected in environments without full dependencies")


if __name__ == "__main__":
    main()