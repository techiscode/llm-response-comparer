"""Interactive, blind scoring in the terminal."""

from __future__ import annotations

from collections.abc import Callable

from .comparison import Comparison, Evaluation
from .rubric import Rubric

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def ask_score(rubric: Rubric, criterion_name: str, input_fn: InputFn, output: OutputFn) -> int:
    prompt = f"  {criterion_name} ({rubric.min_score}-{rubric.max_score}): "
    while True:
        raw = input_fn(prompt).strip()
        try:
            return rubric.validate_score(int(raw))
        except ValueError:
            output(
                f"    Please enter a whole number from {rubric.min_score} to {rubric.max_score}."
            )


def ask_text(prompt: str, input_fn: InputFn, output: OutputFn, required: bool = False) -> str:
    while True:
        value = input_fn(prompt).strip()
        if value or not required:
            return value
        output("    This one is required. A short sentence is enough.")


def score_comparison(
    comparison: Comparison,
    rubric: Rubric,
    input_fn: InputFn = input,
    output: OutputFn = print,
) -> Comparison:
    """Show each response by label only, collect scores and notes, then a preference.

    Model names stay hidden until scoring is finished.
    """
    output(f"\nPROMPT\n{'-' * 60}\n{comparison.prompt}\n")
    for entry in comparison.entries:
        output(f"RESPONSE {entry.label}\n{'-' * 60}\n{entry.response.text}\n")

    output("Rubric:")
    for criterion in rubric.criteria:
        output(f"  {criterion.name}: {criterion.description}")

    for entry in comparison.entries:
        output(f"\nScore response {entry.label}")
        scores = {c.name: ask_score(rubric, c.name, input_fn, output) for c in rubric.criteria}
        note = ask_text("  Note (what was good or wrong): ", input_fn, output)
        entry.evaluation = Evaluation(scores=scores, note=note)

    labels = [entry.label for entry in comparison.entries]
    while True:
        choice = input_fn(f"\nWhich response is better overall? ({'/'.join(labels)}): ")
        choice = choice.strip().upper()
        if choice in labels:
            break
        output(f"    Please choose one of: {', '.join(labels)}")
    comparison.preferred_label = choice
    comparison.preference_reason = ask_text("Why is it better? ", input_fn, output, required=True)

    output("\nRevealing models:")
    for entry in comparison.entries:
        total = entry.evaluation.total if entry.evaluation else 0
        marker = "  <- preferred" if entry.label == choice else ""
        output(f"  {entry.label} = {entry.response.model}  ({total}/{rubric.max_total}){marker}")
    return comparison
