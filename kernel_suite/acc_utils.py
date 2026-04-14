from __future__ import annotations

from typing import Any

import torch


def clone_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {k: clone_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_value(v) for v in value)
    return value


def assert_close_tree(actual: Any, expected: Any, atol: float = 1e-3, rtol: float = 1e-3) -> None:
    if torch.is_tensor(actual) and torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=False):
            assert_close_tree(a, e, atol=atol, rtol=rtol)
        return
    if actual is None or expected is None:
        assert actual is None and expected is None
        return
    raise TypeError(f"Unsupported comparison types: {type(actual)} vs {type(expected)}")


def run_accuracy_cases(module) -> None:
    print(f"Running accuracy cases for {module.KERNEL_NAME}")
    for case in module.CASES:
        module.run_accuracy_case(case)
        print(f"[OK] kernel={module.KERNEL_NAME} case={case['name']} tags={','.join(case.get('tags', []))}")
