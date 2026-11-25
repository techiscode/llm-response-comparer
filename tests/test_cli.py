import json

import pytest

from llm_compare.cli import run
from llm_compare.comparison import load
from llm_compare.providers import ModelResponse, ProviderError

SCORES = ["5", "5", "5", "", "3", "3", "3", "", "A", "More detail"]


def fake_ask(spec, prompt, max_tokens):
    return ModelResponse(model=spec.name, text=f"{spec.model} says hi", seconds=0.1)


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    # Keep any real .env out of the tests.
    monkeypatch.chdir(tmp_path)


def models():
    return ["-m", "anthropic:alpha", "-m", "openai:beta"]


def only_result(out_dir):
    files = sorted(out_dir.glob("*.json"))
    assert len(files) == 1
    return files[0]


def test_run_scores_and_saves_json_and_markdown(tmp_path, scripted_input, capsys):
    out_dir = tmp_path / "results"
    code = run(
        ["run", "Say hi", *models(), "--out-dir", str(out_dir)],
        ask_fn=fake_ask,
        input_fn=scripted_input(SCORES),
    )

    assert code == 0
    path = only_result(out_dir)
    comparison = load(path)
    assert comparison.prompt == "Say hi"
    assert comparison.is_scored
    assert path.with_suffix(".md").read_text().startswith("# LLM Response Comparison")


def test_no_score_then_score_later(tmp_path, scripted_input):
    out_dir = tmp_path / "results"
    assert (
        run(["run", "Say hi", *models(), "--out-dir", str(out_dir), "--no-score"], ask_fn=fake_ask)
        == 0
    )
    path = only_result(out_dir)
    assert not load(path).is_scored

    assert run(["score", str(path)], input_fn=scripted_input(SCORES)) == 0
    assert load(path).is_scored


def test_prompt_file(tmp_path, scripted_input):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("  Explain recursion  \n")
    out_dir = tmp_path / "results"

    run(
        [
            "run",
            "--prompt-file",
            str(prompt_file),
            *models(),
            "--out-dir",
            str(out_dir),
            "--no-score",
        ],
        ask_fn=fake_ask,
    )

    assert load(only_result(out_dir)).prompt == "Explain recursion"


def test_report_command(tmp_path):
    out_dir = tmp_path / "results"
    run(["run", "Say hi", *models(), "--out-dir", str(out_dir), "--no-score"], ask_fn=fake_ask)
    path = only_result(out_dir)
    path.with_suffix(".md").unlink()

    assert run(["report", str(path)]) == 0
    assert path.with_suffix(".md").exists()


def test_custom_rubric_is_used(tmp_path, scripted_input):
    rubric = tmp_path / "r.json"
    rubric.write_text(json.dumps({"name": "Tiny", "criteria": [{"name": "correctness"}]}))
    out_dir = tmp_path / "results"

    run(
        ["run", "Q", *models(), "--rubric", str(rubric), "--out-dir", str(out_dir)],
        ask_fn=fake_ask,
        input_fn=scripted_input(["4", "", "2", "", "A", "Correct"]),
    )

    comparison = load(only_result(out_dir))
    assert comparison.rubric_name == "Tiny"
    assert comparison.entries[0].evaluation.scores == {"correctness": 4}


def test_needs_at_least_two_models(capsys):
    code = run(["run", "Q", "-m", "anthropic:alpha"], ask_fn=fake_ask)
    assert code == 2
    assert "at least two models" in capsys.readouterr().err


def test_bad_model_spec(capsys):
    code = run(["run", "Q", "-m", "alpha", "-m", "openai:beta"], ask_fn=fake_ask)
    assert code == 2
    assert "provider:model" in capsys.readouterr().err


def test_provider_error_stops_without_saving(tmp_path, capsys):
    def failing_ask(spec, prompt, max_tokens):
        raise ProviderError("API error 401: bad key")

    out_dir = tmp_path / "results"
    code = run(["run", "Q", *models(), "--out-dir", str(out_dir)], ask_fn=failing_ask)

    assert code == 1
    assert "bad key" in capsys.readouterr().err
    assert not out_dir.exists()


def test_ctrl_c_during_scoring_exits_cleanly(tmp_path, capsys):
    def interrupt(prompt=""):
        raise KeyboardInterrupt

    out_dir = tmp_path / "results"
    code = run(
        ["run", "Q", *models(), "--out-dir", str(out_dir)], ask_fn=fake_ask, input_fn=interrupt
    )

    assert code == 130
    assert not out_dir.exists()


def test_missing_file_for_score(capsys):
    assert run(["score", "nope.json"]) == 2
    assert "not found" in capsys.readouterr().err
