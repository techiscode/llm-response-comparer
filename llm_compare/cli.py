"""Command-line interface.

python -m llm_compare run "Explain recursion" -m anthropic:MODEL -m openai:MODEL
python -m llm_compare score results/2026-11-19_101500.json
python -m llm_compare report results/2026-11-19_101500.json
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .comparison import build_comparison, load, save
from .providers import ModelResponse, ModelSpec, ProviderError, ask, parse_model_spec
from .report import render_markdown
from .rubric import Rubric, RubricError, load_rubric
from .scoring import InputFn, score_comparison

AskFn = Callable[[ModelSpec, str, int], ModelResponse]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm_compare",
        description="Send one prompt to several LLMs and score the answers blind.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Ask the models, then score their answers")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("prompt", nargs="?", help="The prompt text")
    source.add_argument("--prompt-file", help="Read the prompt from a text file")
    run.add_argument(
        "-m",
        "--model",
        action="append",
        required=True,
        dest="models",
        help="provider:model, repeat for each model (at least two)",
    )
    run.add_argument("--rubric", help="Path to a rubric JSON file (default: built-in)")
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--out-dir", default="results", help="Where to save results")
    run.add_argument("--no-score", action="store_true", help="Save answers and score later")

    score = sub.add_parser("score", help="Score a saved comparison")
    score.add_argument("file", help="Path to a saved comparison JSON file")
    score.add_argument("--rubric", help="Path to a rubric JSON file (default: built-in)")

    report = sub.add_parser("report", help="Write a Markdown report for a saved comparison")
    report.add_argument("file", help="Path to a saved comparison JSON file")
    report.add_argument("--rubric", help="Path to a rubric JSON file (default: built-in)")

    return parser


def _write_outputs(comparison, rubric: Rubric, json_path: Path) -> None:
    save(comparison, json_path)
    md_path = json_path.with_suffix(".md")
    md_path.write_text(render_markdown(comparison, rubric), encoding="utf-8")
    print(f"\nSaved {json_path} and {md_path}")


def cmd_run(args, rubric: Rubric, ask_fn: AskFn, input_fn: InputFn) -> int:
    try:
        specs = [parse_model_spec(value) for value in args.models]
    except ProviderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if len(specs) < 2:
        print("Error: pass at least two models with -m to compare.", file=sys.stderr)
        return 2

    if args.prompt_file:
        try:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            print(f"Error: prompt file not found: {args.prompt_file}", file=sys.stderr)
            return 2
    else:
        prompt = args.prompt.strip()
    if not prompt:
        print("Error: the prompt is empty.", file=sys.stderr)
        return 2

    responses = []
    for spec in specs:
        print(f"Asking {spec.name}...")
        try:
            responses.append(ask_fn(spec, prompt, args.max_tokens))
        except ProviderError as exc:
            print(f"Error from {spec.name}: {exc}", file=sys.stderr)
            return 1

    comparison = build_comparison(prompt, responses, rubric)
    json_path = Path(args.out_dir) / f"{datetime.now():%Y-%m-%d_%H%M%S}.json"

    if not args.no_score:
        score_comparison(comparison, rubric, input_fn=input_fn)
    _write_outputs(comparison, rubric, json_path)
    return 0


def cmd_score(args, rubric: Rubric, input_fn: InputFn) -> int:
    path = Path(args.file)
    try:
        comparison = load(path)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 2
    score_comparison(comparison, rubric, input_fn=input_fn)
    _write_outputs(comparison, rubric, path)
    return 0


def cmd_report(args, rubric: Rubric) -> int:
    path = Path(args.file)
    try:
        comparison = load(path)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 2
    md_path = path.with_suffix(".md")
    md_path.write_text(render_markdown(comparison, rubric), encoding="utf-8")
    print(f"Wrote {md_path}")
    return 0


def run(
    argv: Sequence[str] | None = None,
    ask_fn: AskFn | None = None,
    input_fn: InputFn = input,
) -> int:
    """Run the tool and return an exit code. ask_fn and input_fn can be swapped in tests."""
    load_dotenv()
    args = build_parser().parse_args(argv)

    try:
        rubric = load_rubric(args.rubric)
    except RubricError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "run":
            return cmd_run(args, rubric, ask_fn or ask, input_fn)
        if args.command == "score":
            return cmd_score(args, rubric, input_fn)
        return cmd_report(args, rubric)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Nothing was saved for this step.", file=sys.stderr)
        return 130


def main() -> None:
    sys.exit(run())
