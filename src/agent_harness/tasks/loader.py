"""Load a strict task suite without silently discarding unknown fields."""

from pathlib import Path

import yaml

from agent_harness.tasks.models import TaskSuite


def load_task_suite(path: Path) -> TaskSuite:
    return TaskSuite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
