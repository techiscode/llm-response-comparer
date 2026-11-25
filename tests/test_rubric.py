import json

import pytest

from llm_compare.rubric import DEFAULT_RUBRIC, RubricError, load_rubric


def write(tmp_path, data):
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return path


def test_default_rubric():
    rubric = load_rubric(None)
    assert rubric is DEFAULT_RUBRIC
    assert [c.name for c in rubric.criteria] == ["accuracy", "completeness", "clarity"]
    assert rubric.max_total == 15


def test_load_custom_rubric(tmp_path):
    path = write(
        tmp_path,
        {
            "name": "Code review",
            "max_score": 10,
            "criteria": [
                {"name": "Correctness", "description": "Code runs and is right"},
                {"name": "security"},
            ],
        },
    )
    rubric = load_rubric(path)
    assert rubric.name == "Code review"
    assert [c.name for c in rubric.criteria] == ["correctness", "security"]
    assert rubric.max_total == 20


@pytest.mark.parametrize(
    "data, message",
    [
        ({"criteria": []}, "non-empty 'criteria'"),
        ({"criteria": [{"description": "no name"}]}, "needs a 'name'"),
        ({"criteria": [{"name": "a"}, {"name": "A"}]}, "Duplicate criterion"),
        ({"criteria": [{"name": "a"}], "min_score": 5, "max_score": 5}, "min_score"),
        ("{not json", "not valid JSON"),
    ],
)
def test_invalid_rubrics(tmp_path, data, message):
    with pytest.raises(RubricError, match=message):
        load_rubric(write(tmp_path, data))


def test_missing_rubric_file(tmp_path):
    with pytest.raises(RubricError, match="not found"):
        load_rubric(tmp_path / "missing.json")


def test_validate_score_range():
    assert DEFAULT_RUBRIC.validate_score(5) == 5
    with pytest.raises(ValueError):
        DEFAULT_RUBRIC.validate_score(0)
    with pytest.raises(ValueError):
        DEFAULT_RUBRIC.validate_score(6)


def test_bundled_example_rubrics_are_valid():
    from pathlib import Path

    rubric_dir = Path(__file__).parent.parent / "rubrics"
    files = sorted(rubric_dir.glob("*.json"))
    assert files, "expected example rubrics in rubrics/"
    for path in files:
        load_rubric(path)
