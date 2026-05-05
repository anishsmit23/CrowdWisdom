"""UT-03: retry_with_backoff decorator and direct-call tests.

Tests:
- Function that fails twice then succeeds is called 3 times total.
- Final return value is 'ok'.
- time.sleep is mocked to avoid actual delays.
"""

import pytest
from unittest.mock import MagicMock

from cwt_ads_agent.utils.retry import retry_with_backoff, AgentError


class TestRetryWithBackoff:
    def test_fails_twice_then_succeeds(self, mocker):
        """Function raises Exception twice, returns 'ok' on 3rd call."""
        mock_sleep = mocker.patch("cwt_ads_agent.utils.retry.time.sleep")

        call_count = 0

        @retry_with_backoff(max_retries=2, backoff=[1, 2])
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient failure")
            return "ok"

        result = flaky()

        assert result == "ok"
        assert call_count == 3
        assert mock_sleep.call_count == 2
        # Verify backoff values
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    def test_direct_call_mode(self, mocker):
        """Backward-compatible fn=... mode works."""
        mocker.patch("cwt_ads_agent.utils.retry.time.sleep")

        call_count = 0

        def flaky_fn(*, value):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise RuntimeError("one-off")
            return f"got {value}"

        result = retry_with_backoff(
            fn=flaky_fn,
            max_retries=2,
            backoff=[0.1],
            value="hello",
        )

        assert result == "got hello"
        assert call_count == 2

    def test_exhausts_retries_raises_agent_error(self, mocker):
        """All retries fail → AgentError raised."""
        mocker.patch("cwt_ads_agent.utils.retry.time.sleep")

        @retry_with_backoff(max_retries=2, backoff=[1, 1])
        def always_fails():
            raise ValueError("permanent")

        with pytest.raises(AgentError, match="permanent"):
            always_fails()

    def test_only_retries_specified_exceptions(self, mocker):
        """Non-matching exception types are not retried."""
        mocker.patch("cwt_ads_agent.utils.retry.time.sleep")

        @retry_with_backoff(
            max_retries=3,
            backoff=[0.1],
            exceptions=(ValueError,),
        )
        def raises_type_error():
            raise TypeError("wrong type")

        with pytest.raises(TypeError, match="wrong type"):
            raises_type_error()

    def test_success_on_first_call_no_sleep(self, mocker):
        """No retries needed → no sleep calls."""
        mock_sleep = mocker.patch("cwt_ads_agent.utils.retry.time.sleep")

        @retry_with_backoff(max_retries=3, backoff=[1, 2, 3])
        def instant_success():
            return 42

        assert instant_success() == 42
        mock_sleep.assert_not_called()


class TestAgentError:
    def test_stores_context(self):
        err = AgentError("test", context={"tool": "Apify"})
        assert str(err) == "test"
        assert err.context == {"tool": "Apify"}

    def test_default_empty_context(self):
        err = AgentError("bare")
        assert err.context == {}

    def test_is_runtime_error(self):
        assert issubclass(AgentError, RuntimeError)
