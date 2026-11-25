import random

from llm_compare.comparison import build_comparison
from llm_compare.report import render_markdown
from llm_compare.rubric import DEFAULT_RUBRIC
from llm_compare.scoring import score_comparison

# accuracy, completeness, clarity, note for A; then the same for B; then preference.
HAPPY_PATH = [
    "5",
    "4",
    "4",
    "Correct and complete",
    "2",
    "3",
    "5",
    "Wrong date",
    "A",
    "Fewer errors",
]


def make(responses):
    return build_comparison("When was X?", responses, DEFAULT_RUBRIC, rng=random.Random(0))


def test_scoring_records_scores_notes_and_preference(responses, scripted_input):
    comparison = make(responses)
    out = []

    score_comparison(
        comparison, DEFAULT_RUBRIC, input_fn=scripted_input(HAPPY_PATH), output=out.append
    )

    a, b = comparison.entries
    assert a.evaluation.scores == {"accuracy": 5, "completeness": 4, "clarity": 4}
    assert a.evaluation.note == "Correct and complete"
    assert b.evaluation.total == 10
    assert comparison.preferred_label == "A"
    assert comparison.preference_reason == "Fewer errors"
    assert comparison.is_scored


def test_model_names_hidden_until_reveal(responses, scripted_input):
    comparison = make(responses)
    out = []

    score_comparison(
        comparison, DEFAULT_RUBRIC, input_fn=scripted_input(HAPPY_PATH), output=out.append
    )

    reveal_index = out.index("\nRevealing models:")
    before = "\n".join(out[:reveal_index])
    after = "\n".join(out[reveal_index:])
    assert "model-a" not in before and "model-b" not in before
    assert "model-a" in after and "model-b" in after


def test_scoring_reprompts_on_invalid_input(responses, scripted_input):
    comparison = make(responses)
    answers = [
        "9",
        "abc",
        "5",
        "4",
        "4",
        "",  # A: two bad scores, empty note is allowed
        "3",
        "3",
        "3",
        "",
        "C",
        "b",  # bad label, then lowercase is accepted
        "",
        "Clearer",  # empty reason is rejected, then accepted
    ]
    out = []

    score_comparison(
        comparison, DEFAULT_RUBRIC, input_fn=scripted_input(answers), output=out.append
    )

    assert comparison.entries[0].evaluation.scores["accuracy"] == 5
    assert comparison.preferred_label == "B"
    assert comparison.preference_reason == "Clearer"
    assert sum("whole number" in line for line in out) == 2
    assert any("required" in line for line in out)


def test_report_for_scored_comparison(responses, scripted_input):
    comparison = make(responses)
    score_comparison(
        comparison, DEFAULT_RUBRIC, input_fn=scripted_input(HAPPY_PATH), output=lambda _: None
    )

    report = render_markdown(comparison, DEFAULT_RUBRIC)

    assert "| Response | Model | accuracy | completeness | clarity | Total |" in report
    assert "13/15" in report
    assert "was preferred" in report
    assert "Reason: Fewer errors" in report
    assert "> Evaluator note: Wrong date" in report


def test_report_for_unscored_comparison(responses):
    report = render_markdown(make(responses), DEFAULT_RUBRIC)
    assert "_Not scored yet._" in report
    assert "## Scores" not in report
    assert "Answer from A" in report and "Answer from B" in report


def test_report_escapes_pipes_in_table(responses, scripted_input):
    responses[0].model = "openai:weird|name"
    comparison = make(responses)
    score_comparison(
        comparison, DEFAULT_RUBRIC, input_fn=scripted_input(HAPPY_PATH), output=lambda _: None
    )
    assert "weird\\|name" in render_markdown(comparison, DEFAULT_RUBRIC)
