from unittest.mock import MagicMock

import pytest
import requests

from llm_compare.providers import (
    ANTHROPIC_URL,
    ModelSpec,
    ProviderError,
    ask,
    parse_model_spec,
)


def fake_response(status_code=200, json_data=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data or {}
    response.text = str(json_data)
    return response


@pytest.fixture
def session():
    return MagicMock(spec=requests.Session)


@pytest.fixture(autouse=True)
def api_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def test_parse_model_spec():
    spec = parse_model_spec(" Anthropic : claude-model ")
    assert spec == ModelSpec("anthropic", "claude-model")
    assert spec.name == "anthropic:claude-model"


@pytest.mark.parametrize("value", ["claude-model", "anthropic:", ":model", ""])
def test_parse_model_spec_rejects_bad_format(value):
    with pytest.raises(ProviderError, match="provider:model"):
        parse_model_spec(value)


def test_parse_model_spec_rejects_unknown_provider():
    with pytest.raises(ProviderError, match="Unknown provider 'mistral'"):
        parse_model_spec("mistral:large")


def test_anthropic_request_and_parsing(session):
    session.post.return_value = fake_response(
        json_data={
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "tool_use", "id": "x"},
                {"type": "text", "text": "world"},
            ],
            "usage": {"input_tokens": 12, "output_tokens": 3},
        }
    )

    result = ask(ModelSpec("anthropic", "claude-model"), "Hi", max_tokens=50, session=session)

    assert result.text == "Hello world"
    assert result.model == "anthropic:claude-model"
    assert (result.input_tokens, result.output_tokens) == (12, 3)

    url = session.post.call_args.args[0]
    kwargs = session.post.call_args.kwargs
    assert url == ANTHROPIC_URL
    assert kwargs["headers"]["x-api-key"] == "test-anthropic"
    assert kwargs["headers"]["anthropic-version"] == "2023-06-01"
    assert kwargs["json"] == {
        "model": "claude-model",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Hi"}],
    }


def test_openai_request_and_parsing(session):
    session.post.return_value = fake_response(
        json_data={
            "choices": [{"message": {"content": "Hi there"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
    )

    result = ask(ModelSpec("openai", "some-model"), "Hi", max_tokens=20, session=session)

    assert result.text == "Hi there"
    assert (result.input_tokens, result.output_tokens) == (5, 2)
    url = session.post.call_args.args[0]
    kwargs = session.post.call_args.kwargs
    assert url == "https://api.openai.com/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer test-openai"
    assert kwargs["json"]["max_completion_tokens"] == 20


def test_openai_base_url_can_point_to_compatible_api(session, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1/")
    session.post.return_value = fake_response(
        json_data={"choices": [{"message": {"content": "ok"}}]}
    )

    ask(ModelSpec("openai", "local-model"), "Hi", session=session)

    assert session.post.call_args.args[0] == "http://localhost:11434/v1/chat/completions"


def test_missing_api_key_raises(session, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY is not set"):
        ask(ModelSpec("anthropic", "m"), "Hi", session=session)
    session.post.assert_not_called()


def test_api_error_message_is_surfaced(session):
    session.post.return_value = fake_response(
        400, {"error": {"type": "invalid_request_error", "message": "model not found"}}
    )
    with pytest.raises(ProviderError, match="400: model not found"):
        ask(ModelSpec("anthropic", "nope"), "Hi", session=session)


def test_network_error_is_wrapped(session):
    session.post.side_effect = requests.Timeout("too slow")
    with pytest.raises(ProviderError, match="Could not reach"):
        ask(ModelSpec("openai", "m"), "Hi", session=session)


def test_unexpected_openai_shape_raises(session):
    session.post.return_value = fake_response(json_data={"choices": []})
    with pytest.raises(ProviderError, match="Unexpected response format"):
        ask(ModelSpec("openai", "m"), "Hi", session=session)
