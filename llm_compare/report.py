"""Render a comparison as a readable Markdown report."""

from __future__ import annotations

from .comparison import Comparison
from .rubric import Rubric


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(comparison: Comparison, rubric: Rubric) -> str:
    lines = [
        "# LLM Response Comparison",
        "",
        f"- Date: {comparison.created_at}",
        f"- Rubric: {comparison.rubric_name}",
        "",
        "## Prompt",
        "",
        "```text",
        comparison.prompt,
        "```",
        "",
    ]

    if comparison.is_scored:
        criteria = [c.name for c in rubric.criteria]
        header = ["Response", "Model", *criteria, "Total"]
        lines += [
            "## Scores",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header),
        ]
        for entry in comparison.entries:
            ev = entry.evaluation
            assert ev is not None
            row = [
                entry.label,
                entry.response.model,
                *[str(ev.scores.get(name, "")) for name in criteria],
                f"{ev.total}/{rubric.max_total}",
            ]
            lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
        lines.append("")

        if comparison.preferred_label:
            winner = comparison.entry(comparison.preferred_label)
            lines += [
                "## Preference",
                "",
                f"**Response {winner.label}** ({winner.response.model}) was preferred.",
                "",
                f"Reason: {comparison.preference_reason}",
                "",
            ]
    else:
        lines += ["_Not scored yet._", ""]

    lines += ["## Responses", ""]
    for entry in comparison.entries:
        resp = entry.response
        stats = [f"{resp.seconds}s"]
        if resp.output_tokens is not None:
            stats.append(f"{resp.output_tokens} output tokens")
        lines += [f"### Response {entry.label}: {resp.model} ({', '.join(stats)})", ""]
        if entry.evaluation and entry.evaluation.note:
            lines += [f"> Evaluator note: {entry.evaluation.note}", ""]
        lines += [resp.text, ""]

    return "\n".join(lines)
