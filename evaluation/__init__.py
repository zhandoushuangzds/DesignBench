"""
Evaluation module for benchcore.

Provides task-specific evaluators that assume standardized input format.
"""
from evaluation.base import BaseEvaluator
from evaluation.motif_bench import MotifBenchEvaluator

# Factory function to get evaluator by task type
def get_evaluator(task_type: str, config) -> BaseEvaluator:
    """
    Factory function to get evaluator by task type.
    
    Args:
        task_type: Type of evaluation task (e.g., "motif_scaffolding", "pbp")
        config: Configuration object
        
    Returns:
        Appropriate evaluator instance
    """
    evaluators = {
        "motif_scaffolding": MotifBenchEvaluator,
        "motif_bench": MotifBenchEvaluator,  # Alias
    }
    
    evaluator_class = evaluators.get(task_type)
    if evaluator_class is None:
        raise ValueError(
            f"Unknown task type: {task_type}. "
            f"Available tasks: {list(evaluators.keys())}"
        )
    
    return evaluator_class(config)

__all__ = [
    'BaseEvaluator',
    'MotifBenchEvaluator',
    'get_evaluator',
]
