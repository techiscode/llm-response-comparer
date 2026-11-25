import pytest

from llm_compare.providers import ModelResponse


@pytest.fixture
def responses():
    return [
        ModelResponse(
            model="anthropic:model-a", text="Answer from A", seconds=1.2, output_tokens=40
        ),
        ModelResponse(model="openai:model-b", text="Answer from B", seconds=0.9, output_tokens=55),
    ]


@pytest.fixture
def scripted_input():
    """Return a fake input() that replays answers in order and records prompts."""

    def _make(answers):
        answers = list(answers)
        prompts = []

        def fake_input(prompt=""):
            prompts.append(prompt)
            if not answers:
                raise AssertionError(f"Ran out of scripted answers at prompt: {prompt!r}")
            return answers.pop(0)

        fake_input.prompts = prompts
        return fake_input

    return _make
