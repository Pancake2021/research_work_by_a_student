"""Lightweight evaluation package exports."""

from src.evaluation.ueba_metrics import evaluate_ueba_predictions

__all__ = ["analyze_errors", "compare_methods", "evaluate_checkpoint", "evaluate_ueba_predictions"]


def __getattr__(name):
    if name in {"evaluate_checkpoint", "compare_methods"}:
        from src.evaluation.evaluator import compare_methods, evaluate_checkpoint

        return {"evaluate_checkpoint": evaluate_checkpoint, "compare_methods": compare_methods}[name]
    if name == "analyze_errors":
        from src.evaluation.error_analysis import analyze_errors

        return analyze_errors
    raise AttributeError(f"module 'src.evaluation' has no attribute {name!r}")
