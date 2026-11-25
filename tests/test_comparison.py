import random

from llm_compare.comparison import Evaluation, build_comparison, load, save
from llm_compare.rubric import DEFAULT_RUBRIC


def test_build_comparison_labels_in_order(responses):
    comparison = build_comparison("Q?", responses, DEFAULT_RUBRIC, rng=random.Random(0))
    assert [e.label for e in comparison.entries] == ["A", "B"]
    assert {e.response.model for e in comparison.entries} == {
        "anthropic:model-a",
        "openai:model-b",
    }
    assert not comparison.is_scored


def test_shuffle_actually_changes_order_for_some_seeds(responses):
    first_models = {
        build_comparison("Q?", responses, DEFAULT_RUBRIC, rng=random.Random(seed))
        .entries[0]
        .response.model
        for seed in range(20)
    }
    assert first_models == {"anthropic:model-a", "openai:model-b"}


def test_build_comparison_does_not_mutate_input(responses):
    original = list(responses)
    build_comparison("Q?", responses, DEFAULT_RUBRIC, rng=random.Random(1))
    assert responses == original


def test_evaluation_total():
    assert Evaluation(scores={"a": 4, "b": 3, "c": 5}).total == 12


def test_save_and_load_round_trip(tmp_path, responses):
    comparison = build_comparison("Q?", responses, DEFAULT_RUBRIC, rng=random.Random(0))
    comparison.entries[0].evaluation = Evaluation({"accuracy": 4}, note="solid")
    comparison.entries[1].evaluation = Evaluation({"accuracy": 2})
    comparison.preferred_label = "A"
    comparison.preference_reason = "More accurate"

    path = save(comparison, tmp_path / "nested" / "run.json")
    loaded = load(path)

    assert loaded == comparison
    assert loaded.is_scored
    assert loaded.entry("A").evaluation.note == "solid"
