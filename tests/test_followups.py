import pytest

from backend.agent.nodes import WorkflowNodes


def answered_turn(question, query=None):
    return {"question": question, "response": {
        "citations": [{"number": 1}], "trace": {"rewritten_query": query or question},
    }}


@pytest.mark.parametrize("question", ["explain again", "explain agfain", "Explain again!", "simplify that", "give an example"])
def test_explanation_resolves_last_answered_topic(question):
    history = [answered_turn("explain Section 310"),
               {"question": "my name is akins", "response": {"citations": []}},
               {"question": "explain agfain", "response": {"citations": []}}]
    result = WorkflowNodes(None, None, None, None).rewrite({"question": question, "history": history})
    assert result["query"].startswith("explain Section 310\nFollow-up:")
    assert "simpler terms" in result["query"]


def test_repeated_followup_does_not_chain_previous_requests():
    history = [answered_turn("explain again", "explain Section 310\nFollow-up: Explain again")]
    result = WorkflowNodes(None, None, None, None).rewrite({"question": "explain again", "history": history})
    assert result["query"].count("Follow-up:") == 1
    assert "Section 310" in result["query"]


@pytest.mark.parametrize("question", ["explain Section 43A", "explain a gain in turnover", "my name is akins"])
def test_new_topic_is_not_overwritten(question):
    result = WorkflowNodes(None, None, None, None).rewrite({"question": question, "history": [answered_turn("Section 310")]})
    assert result["query"] == question


def test_empty_session_cannot_borrow_context():
    result = WorkflowNodes(None, None, None, None).rewrite({"question": "explain again", "history": []})
    assert result["query"] == "explain again"