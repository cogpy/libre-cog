# OpenCog Cognitive Chatbot Integration

This document describes the OpenCog integration in libre-cog, which transforms the chatbot into a cognitive AI system with advanced reasoning capabilities.

## Overview

The OpenCog integration adds the following cognitive capabilities to the libre-chat system:

- **AtomSpace**: Structured knowledge representation using atoms and links
- **ECAN Attention Allocation**: Economic attention mechanisms for focus management
- **Moses Evolution**: Self-improving program synthesis through evolutionary algorithms
- **Cognitive Reasoning**: Forward and backward chaining inference with pattern matching

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Cognitive Web Service                     │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Endpoints  │  Gradio UI  │  Cognitive API Routes   │
├─────────────────────────────────────────────────────────────┤
│                    CognitiveLlm                             │
├─────────────────────────────────────────────────────────────┤
│  AtomSpace Memory   │  ECAN Attention  │  Moses Evolution   │
│  Knowledge Repo     │  Focus Manager    │  Program Learning  │
├─────────────────────────────────────────────────────────────┤
│              Cognitive Reasoning Chain                       │
│     Pattern Matching │ Forward Chain │ Backward Chain      │
├─────────────────────────────────────────────────────────────┤
│  Traditional LLM (Fallback/Hybrid)  │  Vector Store (RAG)  │
└─────────────────────────────────────────────────────────────┘
```

## Components

### AtomSpace

The AtomSpace serves as the core knowledge representation system, storing concepts, relationships, and learned patterns as atoms with truth values and importance metrics.

**Key Features:**
- Concept nodes for entities and ideas
- Link types for relationships (inheritance, similarity, evaluation)
- Truth values (strength, confidence) for uncertain reasoning
- Importance values for attention allocation
- Thread-safe operations for concurrent access

**Example Usage:**
```python
# Create concepts
dog = atomspace.create_concept("dog")
animal = atomspace.create_concept("animal")

# Create relationships
inheritance = atomspace.create_inheritance(dog, animal, strength=0.9)

# Update importance for attention
dog.update_importance(5.0)
```

### ECAN Attention Agent

The Economic Cognitive Attention Network (ECAN) manages attention allocation across atoms in the AtomSpace, implementing Short-Term Importance (STI) and attention spreading.

**Key Features:**
- STI-based attentional focus boundary
- Attention spreading between connected atoms
- Importance decay over time
- Continuous attention cycles in background thread

**Configuration:**
```yaml
opencog:
  attention_agent_enabled: true
  attention_cycle_interval: 1.0      # Run every second
  attention_focus_boundary: 10.0     # STI threshold for focus
  attention_bank: 1000.0             # Total attention currency
  importance_decay_rate: 0.005       # Decay rate per cycle
```

### Moses Evolution Engine

Moses (Meta-Optimizing Semantic Evolutionary Search) evolves cognitive programs to improve the system's reasoning capabilities over time.

**Key Features:**
- Population-based evolution of reasoning programs
- Multi-objective fitness functions
- Tournament selection and genetic operators
- Integration with AtomSpace for knowledge-aware evolution
- Continuous learning objectives

**Configuration:**
```yaml
opencog:
  moses_enabled: true
  moses_population_size: 50          # Population size
  moses_mutation_rate: 0.1          # Mutation probability
  moses_crossover_rate: 0.7         # Crossover probability
  moses_elitism_rate: 0.2           # Elite preservation
  moses_evolution_interval: 5.0     # Evolution frequency
```

### Cognitive Reasoning Chain

The reasoning chain integrates AtomSpace knowledge with attention mechanisms to perform structured inference.

**Capabilities:**
- Forward chaining from premises to conclusions
- Backward chaining for goal-directed reasoning
- Pattern matching and unification
- Context-aware reasoning using attentional focus
- Learning from successful inference paths

## Configuration

### Basic Cognitive Setup

Create a configuration file `chat-cognitive.yml`:

```yaml
# Enable OpenCog cognitive features
opencog:
  enabled: true
  
  # AtomSpace settings
  atomspace_max_size: 100000
  
  # Attention settings
  attention_agent_enabled: true
  attention_cycle_interval: 1.0
  attention_focus_boundary: 10.0
  attention_bank: 1000.0
  importance_decay_rate: 0.005
  
  # Evolution settings  
  moses_enabled: true
  moses_population_size: 50
  moses_evolution_interval: 5.0
  
  # Reasoning settings
  cognitive_reasoning_enabled: true
  pattern_matching_enabled: true
  forward_chaining_enabled: true
  backward_chaining_enabled: true
  max_inference_depth: 3

# LLM configuration (traditional fallback)
llm:
  model_path: ./models/your-model.gguf
  temperature: 0.1
  max_new_tokens: 1024
  prompt_variables: [input, history, cognitive_context]
  prompt_template: |
    You are a cognitive AI with structured knowledge representation.
    Current cognitive context: {cognitive_context}
    
    {history}
    User: {input}
    Cognitive AI:

info:
  title: "Cognitive Chatbot"
  description: "OpenCog-powered cognitive AI with reasoning capabilities"
```

### Hybrid Mode (Cognitive + RAG)

Combine cognitive reasoning with document retrieval:

```yaml
opencog:
  enabled: true
  # ... cognitive settings ...

vector:
  vector_path: ./vectorstore/cognitive_db
  embeddings_path: ./embeddings/all-MiniLM-L6-v2
  documents_path: ./documents
  # ... vector settings ...
```

## API Endpoints

The cognitive integration adds several new API endpoints:

### GET `/cognitive/state`
Returns current cognitive system state:
```json
{
  "status": "enabled",
  "atomspace": {
    "size": 1547,
    "top_concepts": ["learning", "intelligence", "reasoning"]
  },
  "attention": {
    "focus_size": 8,
    "focus_concepts": ["current_query", "relevant_concept"]
  },
  "evolution": {
    "generation": 23,
    "best_fitness": 0.87,
    "population_size": 50
  }
}
```

### GET `/cognitive/attention`
Returns atoms in current attentional focus:
```json
{
  "focus_atoms": ["ConceptNode:machine_learning", "ConceptNode:algorithm"],
  "focus_size": 12,
  "attention_bank": 1000.0
}
```

### GET `/cognitive/atomspace`
Returns AtomSpace statistics:
```json
{
  "total_atoms": 1547,
  "top_concepts": [
    {
      "atom": "ConceptNode:artificial_intelligence",
      "importance": 15.3,
      "truth_value": [0.9, 0.8]
    }
  ],
  "atom_types": ["ConceptNode", "InheritanceLink", "SimilarityLink"]
}
```

### POST `/cognitive/learn`
Provide feedback for learning:
```json
{
  "query": "What is machine learning?",
  "response": "Machine learning is a subset of AI...",
  "feedback": "good explanation"
}
```

### GET `/cognitive/export`
Export cognitive knowledge for analysis.

## Usage Examples

### Starting the Cognitive Chatbot

```bash
# Using cognitive configuration
libre-chat start config/chat-opencog-cognitive.yml

# Check cognitive status
curl http://localhost:8000/cognitive/state
```

### Interacting with Cognitive Features

```python
import requests

# Query the cognitive chatbot
response = requests.post("http://localhost:8000/prompt", json={
    "prompt": "What relationships exist between learning and intelligence?"
})

print(response.json())

# Get current attentional focus
attention = requests.get("http://localhost:8000/cognitive/attention")
print("Currently focusing on:", attention.json()["focus_atoms"])

# Provide learning feedback
requests.post("http://localhost:8000/cognitive/learn", json={
    "query": "Explain neural networks",
    "response": "Neural networks are computational models...",
    "feedback": "clear and accurate"
})
```

### Cognitive Query Processing

The system processes queries through multiple reasoning stages:

1. **Query Analysis**: Create concept atoms for query terms
2. **Attention Allocation**: Focus on relevant concepts in AtomSpace
3. **Pattern Matching**: Find related atoms and relationships
4. **Forward Chaining**: Derive new conclusions from premises
5. **Backward Chaining**: Find proof paths for goals
6. **Response Generation**: Combine cognitive inferences with LLM output
7. **Learning**: Update AtomSpace with new relationships

Example cognitive response:
```
Based on my knowledge, I can infer:
- Learning processes improve knowledge representation
- Intelligence emerges from pattern recognition capabilities

I found reasoning paths for your query:
Path 1: learning -> knowledge_acquisition -> intelligence_development
Path 2: pattern_recognition -> abstraction -> intelligent_behavior

Current focus includes: learning, intelligence, cognition, neural_networks
AtomSpace contains 1,247 atoms with 15 in attentional focus.
```

## Performance Considerations

### Memory Usage
- AtomSpace grows with knowledge acquisition
- Configure `atomspace_max_size` to limit memory usage
- Importance decay prevents unbounded growth

### CPU Usage
- Attention cycles run continuously (configurable interval)
- Moses evolution runs in background (configurable frequency)
- Pattern matching complexity scales with AtomSpace size

### Optimization Tips
- Adjust attention cycle interval for performance vs. responsiveness
- Use smaller Moses populations for faster evolution
- Limit inference depth for real-time responses
- Enable hybrid mode for complex queries requiring both reasoning and retrieval

## Troubleshooting

### Common Issues

**OpenCog initialization fails:**
- Check that `opencog.enabled = true` in configuration
- Verify memory requirements for AtomSpace
- Check logs for specific initialization errors

**High CPU usage:**
- Reduce attention cycle frequency
- Decrease Moses population size
- Limit inference depth

**Memory growth:**
- Enable importance decay
- Set reasonable atomspace_max_size
- Monitor AtomSpace growth via `/cognitive/state`

**Slow responses:**
- Use hybrid mode with traditional LLM fallback
- Reduce pattern matching complexity
- Optimize attention focus boundary

### Debugging

Enable debug logging to trace cognitive operations:

```bash
libre-chat start --log-level debug config/cognitive.yml
```

Monitor cognitive state during operation:
```bash
# Watch attention focus changes
watch -n 1 "curl -s http://localhost:8000/cognitive/attention | jq '.focus_atoms'"

# Monitor evolution progress  
watch -n 5 "curl -s http://localhost:8000/cognitive/evolution | jq '.generation, .best_fitness'"
```

## Development

### Extending Cognitive Capabilities

To add new reasoning capabilities:

1. **New Atom Types**: Add to `AtomSpace.atom_types`
2. **Custom Inference Rules**: Extend `PatternMatcher` 
3. **Fitness Functions**: Add to `MosesEvolutionEngine`
4. **API Endpoints**: Add to cognitive router

### Testing

Run cognitive tests:
```bash
python -m pytest tests/test_opencog.py
python -m pytest tests/test_moses.py  
python -m pytest tests/test_cognitive_llm.py
```

### Contributing

The cognitive system is designed to be extensible. Key areas for contribution:

- Enhanced pattern matching algorithms
- Additional reasoning rule types
- Improved attention allocation strategies
- More sophisticated fitness functions
- Integration with external knowledge bases

## References

- [OpenCog Framework](https://opencog.org/)
- [ECAN Attention Allocation](https://wiki.opencog.org/w/Attention_Allocation) 
- [Moses Program Evolution](https://wiki.opencog.org/w/MOSES)
- [AtomSpace Architecture](https://wiki.opencog.org/w/AtomSpace)