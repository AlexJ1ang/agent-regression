"""Readable task evidence, with no inferred score for missing observations."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agent_harness.tasks.models import CheckResult, TaskRun


def _escape(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    return html.escape(value)


def _checks(items: list[CheckResult]) -> str:
    if not items:
        return "<p>No checks configured for this section.</p>"
    rows = "".join(
        f"<tr><td>{_escape(item.name)}</td><td>{_escape(item.status)}</td>"
        f"<td><pre>{_escape(item.expected)}</pre></td>"
        f"<td><pre>{_escape(item.actual)}</pre></td><td>{_escape(item.reason)}</td></tr>"
        for item in items
    )
    return "<table><tr><th>Check</th><th>Status</th><th>Expected</th><th>Actual</th><th>Reason</th></tr>" + rows + "</table>"


def render_report(run: TaskRun, output_dir: Path) -> Path:
    """Write a self-contained HTML report alongside run.json."""
    counts = Counter(result.status for result in run.results)
    summary = " | ".join(
        f"{status}: {counts[status]}"
        for status in ("passed", "failed", "review", "error", "invalid")
    )
    sections: list[str] = []
    for result in run.results:
        turns: list[str] = []
        for turn in result.turns:
            evidence = "<p>No reply recorded.</p>"
            if turn.reply is not None:
                reply = turn.reply
                evidence = (
                    f"<h4>Agent</h4><pre>{_escape(reply.answer)}</pre>"
                    f"<p>Conversation: {_escape(reply.conversation_id)}</p>"
                    f"<p>Tools observed: {_escape(reply.tools_observed)}</p>"
                    f"<pre>{_escape([call.model_dump(mode='json') for call in reply.tool_calls])}</pre>"
                    f"<h4>Observed state</h4><pre>{_escape(reply.state)}</pre>"
                )
            turns.append(
                f"<section><h3>Turn {turn.index}</h3><h4>User</h4><pre>{_escape(turn.message)}</pre>"
                f"{evidence}<p>End-to-end latency: {turn.latency_ms:.3f} ms</p>"
                f"<p>{_escape(turn.error)}</p>{_checks(turn.checks)}</section>"
            )
        sections.append(
            f"<article><h2>{_escape(result.task_id)} — {_escape(result.status)}</h2>"
            f"<p>Completed: {_escape(result.completed)}</p><p>{_escape(result.error)}</p>"
            + "".join(turns) + "<h3>Task checks</h3>" + _checks(result.checks) + "</article>"
        )
    document = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Agent Regression task report</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "article{border:1px solid #bbb;margin:1.5rem 0;padding:1rem}"
        "section{border-top:1px solid #ddd}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ddd;padding:.5rem;text-align:left;vertical-align:top}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "</style><body><h1>Agent Regression task report</h1>"
        f"<p>Suite: {_escape(run.suite_id)}</p><p>{summary}</p>"
        f"<p>Configuration fingerprint: {_escape(run.fingerprint)}</p>"
        "<p>Review means evidence is insufficient. Execution errors are shown separately.</p>"
        "<p><a href='run.json'>Full JSON evidence</a></p>"
        + "".join(sections) + "</body></html>"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
