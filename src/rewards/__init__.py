"""
rewards/__init__.py
===================
Реестр всех reward functions.

Использование:
    from src.rewards import get_reward_fn, REWARD_REGISTRY

    reward_fn = get_reward_fn("reasoning")
"""

from src.rewards.reward_accuracy   import reward_accuracy
from src.rewards.reward_reasoning  import reward_with_reasoning
from src.rewards.reward_binary     import reward_binary
from src.rewards.reward_entropy    import reward_with_entropy_bonus
from src.rewards.reward_lambda_grpo import reward_length_weighted

REWARD_REGISTRY = {
    "accuracy":    reward_accuracy,
    "reasoning":   reward_with_reasoning,
    "binary":      reward_binary,
    "entropy":     reward_with_entropy_bonus,
    "lambda_grpo": reward_length_weighted,
}


def get_reward_fn(name: str):
    """Возвращает reward function по имени."""
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Неизвестная reward function: '{name}'. "
            f"Доступны: {list(REWARD_REGISTRY.keys())}"
        )
    return REWARD_REGISTRY[name]


__all__ = [
    "reward_accuracy",
    "reward_with_reasoning",
    "reward_binary",
    "reward_with_entropy_bonus",
    "reward_length_weighted",
    "REWARD_REGISTRY",
    "get_reward_fn",
]
