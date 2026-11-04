from flock.results import safe_error


def test_safe_error_redacts_environment_and_runtime_credentials(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-provider-token")
    monkeypatch.setenv("service_secret", "test-service-secret")
    monkeypatch.setenv("SERVICE_API_KEY", "test-api-key")
    monkeypatch.setenv("EMPTY_TOKEN", "")
    error = RuntimeError(
        "Failed: test-provider-token test-service-secret test-api-key test-board-token"
    )

    result = safe_error(error, ["test-board-token", ""])

    assert result == "RuntimeError: Failed: [redacted] [redacted] [redacted] [redacted]"


def test_safe_error_redacts_longer_credentials_before_shared_prefixes(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token-with-private-suffix")

    result = safe_error(ValueError("test-token-with-private-suffix"), ["test-token"])

    assert result == "ValueError: [redacted]"
