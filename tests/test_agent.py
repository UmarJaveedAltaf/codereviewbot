from unittest.mock import MagicMock, patch

import pytest

from agent.reviewer import FileFinding, _review_file


@pytest.fixture()
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="[WARNING] math_utils.py:2 — Division by zero risk\nNo guard against b=0.")
    return llm


@pytest.fixture()
def sample_cf():
    from parser.diff_extractor import ChangedFile, HunkRange
    return ChangedFile(
        filename="math_utils.py",
        language="python",
        patch="@@ -1,3 +1,5 @@\n+def divide(a, b):\n+    return a / b\n",
        added_lines=["def divide(a, b):", "    return a / b"],
        removed_lines=[],
        hunks=[HunkRange(start=1, length=5)],
    )


@patch("agent.reviewer.query_similar", return_value=[])
@patch("agent.reviewer.extract_functions", return_value=[])
def test_review_file_calls_llm(mock_extract, mock_query, mock_llm, sample_cf):
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = MagicMock(content="[WARNING] divide — no zero guard")
    with patch("agent.reviewer.review_prompt") as mock_prompt:
        mock_prompt.__or__ = MagicMock(return_value=chain_mock)
        result = _review_file(mock_llm, "org/repo", 42, "Add divide", sample_cf)
    assert isinstance(result, str)


@patch("agent.reviewer.query_similar", return_value=[
    {"repo": "org/repo", "pr_number": 1, "comment": "Always guard division", "distance": 0.1}
])
@patch("agent.reviewer.extract_functions", return_value=[])
def test_review_file_includes_past_conventions(mock_extract, mock_query, mock_llm, sample_cf):
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = MagicMock(content="see past conventions")
    with patch("agent.reviewer.review_prompt") as mock_prompt:
        mock_prompt.__or__ = MagicMock(return_value=chain_mock)
        _review_file(mock_llm, "org/repo", 42, "title", sample_cf)
    call_kwargs = chain_mock.invoke.call_args[0][0]
    assert "Always guard division" in call_kwargs["past_conventions"]
