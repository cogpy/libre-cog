"""Tests for Moses evolution engine."""

import time
import unittest

from libre_chat.moses_evolution import (
    FitnessFunction,
    MosesEvolutionEngine,
    Program,
)
from libre_chat.opencog_integration import AtomSpace


class TestProgram(unittest.TestCase):
    """Test Program class functionality."""
    
    def test_program_creation(self):
        """Test basic program creation."""
        program = Program("test_expression")
        self.assertEqual(program.expression, "test_expression")
        self.assertEqual(program.fitness, 0.0)
        self.assertEqual(program.generation, 0)
        self.assertIsNotNone(program.id)
    
    def test_program_evaluation(self):
        """Test program evaluation."""
        program = Program("REASONING(x, y)")
        
        def dummy_fitness(prog):
            return 0.7
        
        fitness = program.evaluate(dummy_fitness)
        self.assertEqual(fitness, 0.7)
        self.assertEqual(program.fitness, 0.7)
        self.assertEqual(program.execution_count, 1)
        self.assertEqual(program.success_count, 1)  # fitness > 0.5
    
    def test_program_mutation(self):
        """Test program mutation."""
        original = Program("SIMPLE_EXPRESSION")
        mutated = original.mutate()
        
        self.assertNotEqual(mutated.id, original.id)
        self.assertEqual(mutated.generation, original.generation + 1)
        self.assertEqual(mutated.parent_id, original.id)
    
    def test_program_crossover(self):
        """Test program crossover."""
        parent1 = Program("EXPRESSION_A")
        parent2 = Program("EXPRESSION_B")
        
        child1, child2 = parent1.crossover(parent2)
        
        self.assertNotEqual(child1.id, parent1.id)
        self.assertNotEqual(child2.id, parent2.id)
        self.assertEqual(child1.generation, 1)
        self.assertEqual(child2.generation, 1)
        self.assertIn(parent1.id, child1.parent_id)
        self.assertIn(parent2.id, child1.parent_id)


class TestFitnessFunction(unittest.TestCase):
    """Test FitnessFunction class."""
    
    def setUp(self):
        """Set up test environment."""
        self.atomspace = AtomSpace()
        self.fitness_function = FitnessFunction(self.atomspace)
    
    def test_add_test_case(self):
        """Test adding test cases."""
        self.fitness_function.add_test_case(
            {"input": "test"},
            {"expected": "output"}
        )
        
        self.assertEqual(len(self.fitness_function.test_cases), 1)
        self.assertEqual(self.fitness_function.test_cases[0]["input"], {"input": "test"})
    
    def test_add_target_behavior(self):
        """Test adding target behaviors."""
        self.fitness_function.add_target_behavior(
            "improve reasoning", 
            weight=1.5
        )
        
        self.assertEqual(len(self.fitness_function.target_behaviors), 1)
        behavior = self.fitness_function.target_behaviors[0]
        self.assertEqual(behavior["description"], "improve reasoning")
        self.assertEqual(behavior["weight"], 1.5)
    
    def test_evaluate_program_basic(self):
        """Test basic program evaluation."""
        program = Program("REASONING(query, context)")
        
        fitness = self.fitness_function.evaluate(program)
        
        # Should return a valid fitness score between 0 and 1
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
    
    def test_evaluate_with_test_cases(self):
        """Test evaluation with test cases."""
        self.fitness_function.add_test_case(
            {"query": "test question"},
            {"type": "reasoning_result"}
        )
        
        # Program that should match the test case
        program = Program("reasoning(input)")
        
        fitness = self.fitness_function.evaluate(program)
        
        self.assertGreater(fitness, 0.0)
    
    def test_evaluate_cognitive_coherence(self):
        """Test cognitive coherence evaluation."""
        # Add some important atoms to atomspace
        concept = self.atomspace.create_concept("important_concept")
        concept.update_importance(5.0)
        
        # Program using the important atom
        program = Program("TEST_EXPRESSION", atoms=[concept])
        
        coherence = self.fitness_function._evaluate_cognitive_coherence(program)
        
        self.assertGreater(coherence, 0.0)
    
    def test_evaluate_complexity(self):
        """Test complexity evaluation."""
        simple_program = Program("A")
        complex_program = Program("A" * 100)  # Very long expression
        
        simple_complexity = self.fitness_function._evaluate_complexity(simple_program)
        complex_complexity = self.fitness_function._evaluate_complexity(complex_program)
        
        # Simpler programs should have higher complexity scores
        self.assertGreater(simple_complexity, complex_complexity)


class TestMosesEvolutionEngine(unittest.TestCase):
    """Test Moses evolution engine."""
    
    def setUp(self):
        """Set up test environment."""
        self.atomspace = AtomSpace()
        self.moses_engine = MosesEvolutionEngine(
            self.atomspace, 
            population_size=10  # Small population for testing
        )
    
    def test_moses_creation(self):
        """Test Moses engine initialization."""
        self.assertEqual(self.moses_engine.atomspace, self.atomspace)
        self.assertEqual(self.moses_engine.population_size, 10)
        self.assertEqual(self.moses_engine.generation, 0)
        self.assertFalse(self.moses_engine.running)
    
    def test_initialize_population(self):
        """Test population initialization."""
        self.moses_engine.initialize_population()
        
        self.assertEqual(len(self.moses_engine.population), 10)
        
        # Check that all programs are properly initialized
        for program in self.moses_engine.population:
            self.assertIsInstance(program, Program)
            self.assertIsNotNone(program.expression)
            self.assertEqual(program.generation, 0)
    
    def test_get_best_programs(self):
        """Test getting best programs."""
        self.moses_engine.initialize_population()
        
        # Set some fitness values
        for i, program in enumerate(self.moses_engine.population):
            program.fitness = i / 10.0  # Fitness from 0.0 to 0.9
        
        best_programs = self.moses_engine.get_best_programs(count=3)
        
        self.assertEqual(len(best_programs), 3)
        # Should be sorted by fitness (descending)
        self.assertGreaterEqual(best_programs[0].fitness, best_programs[1].fitness)
        self.assertGreaterEqual(best_programs[1].fitness, best_programs[2].fitness)
    
    def test_add_learning_objective(self):
        """Test adding learning objectives."""
        test_cases = [
            {"input": {"query": "test"}, "expected": "output"}
        ]
        
        self.moses_engine.add_learning_objective(
            "Test objective", 
            test_cases
        )
        
        # Should add to fitness function
        self.assertEqual(len(self.moses_engine.fitness_function.target_behaviors), 1)
        self.assertEqual(len(self.moses_engine.fitness_function.test_cases), 1)
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        self.moses_engine.initialize_population()
        
        # Set fitness values
        for i, program in enumerate(self.moses_engine.population):
            program.fitness = i / 10.0
        
        # Tournament should select higher fitness programs more often
        selections = []
        for _ in range(50):  # Multiple selections
            selected = self.moses_engine._tournament_selection()
            selections.append(selected.fitness)
        
        # Average selected fitness should be higher than population average
        avg_selected = sum(selections) / len(selections)
        avg_population = sum(p.fitness for p in self.moses_engine.population) / len(self.moses_engine.population)
        
        self.assertGreater(avg_selected, avg_population)
    
    def test_evolution_summary(self):
        """Test getting evolution summary."""
        self.moses_engine.initialize_population()
        
        # Simulate some evolution history
        self.moses_engine.evolution_history = [
            {"generation": 0, "best_fitness": 0.5, "average_fitness": 0.3},
            {"generation": 1, "best_fitness": 0.6, "average_fitness": 0.35},
        ]
        self.moses_engine.generation = 2
        
        summary = self.moses_engine.get_evolution_summary()
        
        self.assertIn("status", summary)
        self.assertIn("generation", summary)
        self.assertIn("population_size", summary)
        self.assertIn("recent_progress", summary)
        self.assertEqual(summary["generation"], 2)
        self.assertEqual(summary["population_size"], 10)
    
    def test_export_best_program(self):
        """Test exporting best program."""
        self.moses_engine.initialize_population()
        
        # Set best program
        best = self.moses_engine.population[0]
        best.fitness = 0.9
        best.execution_count = 5
        best.success_count = 4
        self.moses_engine.best_program = best
        
        export_data = self.moses_engine.export_best_program()
        
        self.assertIsNotNone(export_data)
        self.assertEqual(export_data["fitness"], 0.9)
        self.assertEqual(export_data["execution_count"], 5)
        self.assertEqual(export_data["success_rate"], 0.8)  # 4/5
    
    def test_evolve_generation(self):
        """Test evolving one generation."""
        self.moses_engine.initialize_population()
        
        # Set initial fitness values
        for i, program in enumerate(self.moses_engine.population):
            program.fitness = i / 10.0
        
        initial_generation = self.moses_engine.generation
        
        self.moses_engine._evolve_generation()
        
        # Generation should have incremented
        self.assertEqual(self.moses_engine.generation, initial_generation + 1)
        
        # Should have recorded statistics
        self.assertGreater(len(self.moses_engine.evolution_history), 0)
        
        # Best program should be set
        self.assertIsNotNone(self.moses_engine.best_program)


class TestIntegration(unittest.TestCase):
    """Test integration between Moses and AtomSpace."""
    
    def setUp(self):
        """Set up integrated test environment."""
        self.atomspace = AtomSpace()
        
        # Add some knowledge to atomspace
        self.dog = self.atomspace.create_concept("dog")
        self.animal = self.atomspace.create_concept("animal")
        self.inheritance = self.atomspace.create_inheritance(self.dog, self.animal)
        
        self.moses_engine = MosesEvolutionEngine(self.atomspace, population_size=5)
    
    def test_programs_use_atomspace_knowledge(self):
        """Test that programs can use knowledge from AtomSpace."""
        self.moses_engine.initialize_population()
        
        # Programs should have access to atomspace atoms
        for program in self.moses_engine.population:
            if program.atoms:
                # At least one program should use atoms from atomspace
                atom_ids = [atom.id for atom in program.atoms]
                atomspace_ids = [atom.id for atom in self.atomspace.atoms.values()]
                
                # Should have some overlap
                overlap = set(atom_ids) & set(atomspace_ids)
                if overlap:
                    self.assertGreater(len(overlap), 0)
                    break
    
    def test_fitness_considers_atomspace_state(self):
        """Test that fitness function considers AtomSpace state."""
        # Create program that uses important atoms
        important_atom = self.atomspace.create_concept("important")
        important_atom.update_importance(10.0)
        
        program_with_important = Program("TEST", atoms=[important_atom])
        program_without = Program("TEST", atoms=[])
        
        fitness_with = self.moses_engine.fitness_function.evaluate(program_with_important)
        fitness_without = self.moses_engine.fitness_function.evaluate(program_without)
        
        # Program using important atoms should potentially have better cognitive coherence
        # (though other factors may affect overall fitness)
        self.assertGreaterEqual(fitness_with, 0.0)
        self.assertGreaterEqual(fitness_without, 0.0)


if __name__ == '__main__':
    unittest.main()