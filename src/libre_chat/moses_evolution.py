"""Moses evolution engine integration for self-guided learning."""

import json
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from libre_chat.opencog_integration import Atom, AtomSpace
from libre_chat.utils import log


class Program:
    """
    Represents a program in the Moses evolution system.
    
    Programs are represented as expression trees that can be evolved.
    """
    
    def __init__(self, expression: str, atoms: List[Atom] = None):
        self.expression = expression
        self.atoms = atoms or []
        self.fitness = 0.0
        self.generation = 0
        self.parent_id = None
        self.id = f"prog_{hash(expression)}_{time.time()}"
        self.execution_count = 0
        self.success_count = 0
    
    def evaluate(self, fitness_function: Callable) -> float:
        """Evaluate program fitness using provided function."""
        try:
            self.fitness = fitness_function(self)
            self.execution_count += 1
            if self.fitness > 0.5:
                self.success_count += 1
            return self.fitness
        except Exception as e:
            log.error(f"Error evaluating program {self.id}: {e}")
            return 0.0
    
    def mutate(self) -> 'Program':
        """Create a mutated version of this program."""
        mutations = [
            self._add_random_operation,
            self._modify_parameter,
            self._reorder_operations,
            self._simplify_expression
        ]
        
        mutation_func = random.choice(mutations)
        mutated_expr = mutation_func()
        
        mutated_program = Program(mutated_expr, self.atoms.copy())
        mutated_program.generation = self.generation + 1
        mutated_program.parent_id = self.id
        
        return mutated_program
    
    def _add_random_operation(self) -> str:
        """Add a random operation to the expression."""
        operations = ["AND", "OR", "NOT", "IMPLIES", "SIMILAR"]
        op = random.choice(operations)
        
        # Simple string manipulation - could be more sophisticated
        if "AND" not in self.expression and len(self.expression) > 10:
            return f"({self.expression} AND {op}(x))"
        return self.expression
    
    def _modify_parameter(self) -> str:
        """Modify a parameter in the expression."""
        # Replace numeric values with random variations
        import re
        numbers = re.findall(r'\d+\.?\d*', self.expression)
        if numbers:
            old_num = random.choice(numbers)
            new_num = str(float(old_num) * random.uniform(0.8, 1.2))
            return self.expression.replace(old_num, new_num, 1)
        return self.expression
    
    def _reorder_operations(self) -> str:
        """Reorder operations in the expression."""
        # Simple reordering for commutative operations
        if " AND " in self.expression:
            parts = self.expression.split(" AND ")
            if len(parts) > 1:
                random.shuffle(parts)
                return " AND ".join(parts)
        return self.expression
    
    def _simplify_expression(self) -> str:
        """Simplify the expression by removing redundant parts."""
        # Remove duplicate operations
        simplified = self.expression
        simplified = simplified.replace("(( ", "(").replace(" ))", ")")
        return simplified
    
    def crossover(self, other: 'Program') -> Tuple['Program', 'Program']:
        """Perform crossover with another program to create offspring."""
        # Simple crossover - split expressions at random points
        if len(self.expression) < 5 or len(other.expression) < 5:
            return self.mutate(), other.mutate()
        
        split1 = random.randint(1, len(self.expression) - 1)
        split2 = random.randint(1, len(other.expression) - 1)
        
        child1_expr = self.expression[:split1] + other.expression[split2:]
        child2_expr = other.expression[:split2] + self.expression[split1:]
        
        child1 = Program(child1_expr, self.atoms + other.atoms)
        child2 = Program(child2_expr, other.atoms + self.atoms)
        
        child1.generation = max(self.generation, other.generation) + 1
        child2.generation = max(self.generation, other.generation) + 1
        
        child1.parent_id = f"{self.id}+{other.id}"
        child2.parent_id = f"{other.id}+{self.id}"
        
        return child1, child2
    
    def __str__(self):
        return f"Program(expr='{self.expression[:50]}...', fitness={self.fitness:.3f})"
    
    def __repr__(self):
        return self.__str__()


class FitnessFunction:
    """
    Fitness function for evaluating program performance in cognitive tasks.
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.test_cases = []
        self.target_behaviors = []
    
    def add_test_case(self, input_data: Dict[str, Any], expected_output: Any):
        """Add a test case for program evaluation."""
        self.test_cases.append({
            "input": input_data,
            "expected": expected_output
        })
    
    def add_target_behavior(self, behavior_description: str, weight: float = 1.0):
        """Add a target behavior with associated weight."""
        self.target_behaviors.append({
            "description": behavior_description,
            "weight": weight
        })
    
    def evaluate(self, program: Program) -> float:
        """Evaluate program fitness based on multiple criteria."""
        fitness_components = []
        
        # Test case performance
        if self.test_cases:
            test_fitness = self._evaluate_test_cases(program)
            fitness_components.append(test_fitness * 0.4)
        
        # Cognitive coherence
        coherence_fitness = self._evaluate_cognitive_coherence(program)
        fitness_components.append(coherence_fitness * 0.3)
        
        # Complexity penalty (prefer simpler solutions)
        complexity_fitness = self._evaluate_complexity(program)
        fitness_components.append(complexity_fitness * 0.2)
        
        # Novelty bonus
        novelty_fitness = self._evaluate_novelty(program)
        fitness_components.append(novelty_fitness * 0.1)
        
        total_fitness = sum(fitness_components)
        return max(0.0, min(1.0, total_fitness))
    
    def _evaluate_test_cases(self, program: Program) -> float:
        """Evaluate program against test cases."""
        if not self.test_cases:
            return 0.5  # Neutral score if no test cases
        
        correct_count = 0
        for test_case in self.test_cases:
            try:
                # Simulate program execution (simplified)
                result = self._simulate_execution(program, test_case["input"])
                if self._compare_results(result, test_case["expected"]):
                    correct_count += 1
            except Exception:
                continue
        
        return correct_count / len(self.test_cases)
    
    def _evaluate_cognitive_coherence(self, program: Program) -> float:
        """Evaluate how well program integrates with existing knowledge."""
        coherence_score = 0.0
        
        # Check if program uses relevant atoms from atomspace
        relevant_atoms = 0
        for atom in program.atoms:
            if atom.importance > 1.0:  # High importance atoms
                relevant_atoms += 1
                coherence_score += 0.1
        
        # Bonus for using highly connected atoms
        for atom in program.atoms:
            if len(atom.incoming) + len(atom.outgoing) > 2:
                coherence_score += 0.05
        
        return min(1.0, coherence_score)
    
    def _evaluate_complexity(self, program: Program) -> float:
        """Evaluate program complexity (lower is better)."""
        expr_length = len(program.expression)
        atom_count = len(program.atoms)
        
        # Normalize complexity measures
        length_score = max(0.0, 1.0 - expr_length / 200.0)
        atom_score = max(0.0, 1.0 - atom_count / 20.0)
        
        return (length_score + atom_score) / 2.0
    
    def _evaluate_novelty(self, program: Program) -> float:
        """Evaluate program novelty (encourage exploration)."""
        # Simple novelty measure based on expression uniqueness
        expr_hash = hash(program.expression)
        
        # In a full implementation, this would check against a database
        # of previously seen expressions
        novelty_score = min(1.0, abs(expr_hash) % 100 / 100.0)
        
        return novelty_score
    
    def _simulate_execution(self, program: Program, input_data: Dict[str, Any]) -> Any:
        """Simulate program execution with given input."""
        # Simplified execution simulation
        # In practice, this would execute the program logic
        
        if "reasoning" in program.expression.lower():
            return {"type": "reasoning_result", "confidence": 0.7}
        elif "pattern" in program.expression.lower():
            return {"type": "pattern_match", "matches": 3}
        else:
            return {"type": "default", "value": 0.5}
    
    def _compare_results(self, result: Any, expected: Any) -> bool:
        """Compare program result with expected output."""
        if isinstance(result, dict) and isinstance(expected, dict):
            return result.get("type") == expected.get("type")
        return result == expected


class MosesEvolutionEngine:
    """
    Moses (Meta-Optimizing Semantic Evolutionary Search) engine for program evolution.
    
    Evolves cognitive programs that improve the chatbot's reasoning capabilities.
    """
    
    def __init__(self, atomspace: AtomSpace, population_size: int = 50):
        self.atomspace = atomspace
        self.population_size = population_size
        self.population: List[Program] = []
        self.generation = 0
        self.best_program: Optional[Program] = None
        self.fitness_function = FitnessFunction(atomspace)
        self.evolution_history = []
        self.running = False
        self.thread = None
        
        # Evolution parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.elitism_rate = 0.2
        self.tournament_size = 5
        
        log.info(f"🧬 Moses Evolution Engine initialized with population size {population_size}")
    
    def initialize_population(self):
        """Initialize the population with seed programs."""
        log.info("🧬 Initializing Moses population...")
        
        # Create seed programs for different cognitive tasks
        seed_expressions = [
            "REASONING(query, context)",
            "PATTERN_MATCH(input, memory)",
            "ATTENTION_FOCUS(concepts)",
            "INFERENCE(premises, rules)",
            "SIMILARITY(concept1, concept2)",
            "PREDICTION(context, history)",
            "CLASSIFICATION(features, categories)",
            "ASSOCIATION(stimulus, response)",
            "ABSTRACTION(specific_cases)",
            "GENERALIZATION(examples, pattern)"
        ]
        
        for expr in seed_expressions:
            # Create multiple variations of each seed
            for i in range(self.population_size // len(seed_expressions)):
                variation = f"{expr}_v{i}"
                
                # Add relevant atoms from atomspace
                relevant_atoms = self.atomspace.get_atoms_by_importance(limit=5)
                
                program = Program(variation, relevant_atoms)
                program.generation = 0
                self.population.append(program)
        
        # Fill remaining slots with random combinations
        while len(self.population) < self.population_size:
            expr = f"COMPOSITE({random.choice(seed_expressions)}, {random.choice(seed_expressions)})"
            atoms = random.sample(list(self.atomspace.atoms.values()), 
                                min(3, len(self.atomspace.atoms)))
            program = Program(expr, atoms)
            self.population.append(program)
        
        log.info(f"🧬 Initialized population with {len(self.population)} programs")
    
    def start_evolution(self):
        """Start the evolution process in a background thread."""
        if not self.running:
            self.running = True
            if not self.population:
                self.initialize_population()
            
            self.thread = threading.Thread(target=self._evolution_loop)
            self.thread.daemon = True
            self.thread.start()
            log.info("🧬 Moses evolution started")
    
    def stop_evolution(self):
        """Stop the evolution process."""
        self.running = False
        if self.thread:
            self.thread.join()
        log.info("🧬 Moses evolution stopped")
    
    def _evolution_loop(self):
        """Main evolution loop."""
        while self.running:
            try:
                self._evolve_generation()
                time.sleep(5.0)  # Evolve every 5 seconds
            except Exception as e:
                log.error(f"Error in evolution loop: {e}")
    
    def _evolve_generation(self):
        """Evolve one generation of programs."""
        log.debug(f"🧬 Evolving generation {self.generation}")
        
        # Evaluate fitness for all programs
        for program in self.population:
            program.evaluate(self.fitness_function.evaluate)
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda p: p.fitness, reverse=True)
        
        # Update best program
        if not self.best_program or self.population[0].fitness > self.best_program.fitness:
            self.best_program = self.population[0]
            log.info(f"🧬 New best program found: fitness {self.best_program.fitness:.3f}")
        
        # Record evolution statistics
        self._record_generation_stats()
        
        # Create next generation
        next_population = []
        
        # Elitism: keep top performers
        elite_count = int(self.population_size * self.elitism_rate)
        next_population.extend(self.population[:elite_count])
        
        # Generate offspring through crossover and mutation
        while len(next_population) < self.population_size:
            if random.random() < self.crossover_rate and len(self.population) > 1:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child1, child2 = parent1.crossover(parent2)
                next_population.extend([child1, child2])
            else:
                # Mutation only
                parent = self._tournament_selection()
                child = parent.mutate()
                next_population.append(child)
        
        # Trim to exact population size
        self.population = next_population[:self.population_size]
        self.generation += 1
    
    def _tournament_selection(self) -> Program:
        """Select a program using tournament selection."""
        tournament_candidates = random.sample(self.population, 
                                           min(self.tournament_size, len(self.population)))
        return max(tournament_candidates, key=lambda p: p.fitness)
    
    def _record_generation_stats(self):
        """Record statistics for this generation."""
        fitnesses = [p.fitness for p in self.population]
        stats = {
            "generation": self.generation,
            "best_fitness": max(fitnesses),
            "average_fitness": sum(fitnesses) / len(fitnesses),
            "worst_fitness": min(fitnesses),
            "population_size": len(self.population)
        }
        
        self.evolution_history.append(stats)
        
        # Keep only recent history to prevent memory bloat
        if len(self.evolution_history) > 1000:
            self.evolution_history = self.evolution_history[-500:]
    
    def get_best_programs(self, count: int = 5) -> List[Program]:
        """Get the top performing programs."""
        if not self.population:
            return []
        
        sorted_pop = sorted(self.population, key=lambda p: p.fitness, reverse=True)
        return sorted_pop[:count]
    
    def add_learning_objective(self, description: str, test_cases: List[Dict[str, Any]] = None):
        """Add a new learning objective to guide evolution."""
        self.fitness_function.add_target_behavior(description)
        
        if test_cases:
            for test_case in test_cases:
                self.fitness_function.add_test_case(
                    test_case.get("input", {}),
                    test_case.get("expected")
                )
        
        log.info(f"🧬 Added learning objective: {description}")
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution progress."""
        if not self.evolution_history:
            return {"status": "not_started"}
        
        recent_stats = self.evolution_history[-10:] if len(self.evolution_history) >= 10 else self.evolution_history
        
        return {
            "status": "running" if self.running else "stopped",
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.best_program.fitness if self.best_program else 0.0,
            "recent_progress": [stats["best_fitness"] for stats in recent_stats],
            "total_generations": len(self.evolution_history),
            "top_programs": [
                {
                    "expression": p.expression[:100],
                    "fitness": p.fitness,
                    "generation": p.generation
                }
                for p in self.get_best_programs(3)
            ]
        }
    
    def export_best_program(self) -> Optional[Dict[str, Any]]:
        """Export the best program for external use."""
        if not self.best_program:
            return None
        
        return {
            "id": self.best_program.id,
            "expression": self.best_program.expression,
            "fitness": self.best_program.fitness,
            "generation": self.best_program.generation,
            "atom_count": len(self.best_program.atoms),
            "execution_count": self.best_program.execution_count,
            "success_rate": (self.best_program.success_count / 
                           max(1, self.best_program.execution_count))
        }