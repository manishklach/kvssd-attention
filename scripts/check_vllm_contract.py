"""Fail when the pinned vLLM source no longer exposes the integration contracts we use."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def class_methods(path: Path, class_name: str) -> tuple[set[str], ast.ClassDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
            return methods, node
    raise AssertionError(f"{class_name} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_source", type=Path)
    args = parser.parse_args()
    root = args.vllm_source / "vllm" / "v1" / "kv_offload"
    secondary, _ = class_methods(root / "tiering" / "base.py", "SecondaryTierManager")
    required = {
        "lookup",
        "submit_store",
        "submit_load",
        "get_finished_jobs",
        "on_new_request",
        "drain_jobs",
        "shutdown",
    }
    assert required <= secondary, f"SecondaryTierManager contract changed: {required - secondary}"
    factory, _ = class_methods(root / "tiering" / "factory.py", "SecondaryTierFactory")
    assert {"register_tier", "create_secondary_tier", "get_tier_class"} <= factory
    spec, node = class_methods(root / "tiering" / "spec.py", "TieringOffloadingSpec")
    assert {"__init__", "get_manager", "create_worker"} <= spec
    assert any(
        isinstance(base, ast.Name) and base.id == "CPUOffloadingSpec" for base in node.bases
    ), "TieringOffloadingSpec no longer extends CPUOffloadingSpec"
    print("vLLM source contract is compatible")


if __name__ == "__main__":
    main()
