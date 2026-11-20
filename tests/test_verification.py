from backend.agent.gemini import Claim, Draft, Support, Verdict, Verification, assess_claims, generate_claims
from backend.agent.nodes import WorkflowNodes


def test_unsupported_missing_and_fabricated_quotes_are_rejected():
    citations = [{"number": 1, "text": "Employees receive 20 days leave."}]
    draft = Draft(answer="20 days leave.", claims=[
        Claim(text="20 days leave.", citation_numbers=[1]),
        Claim(text="30 days leave.", citation_numbers=[1]),
        Claim(text="Missing citation.", citation_numbers=[9]),
        Claim(text="No review.", citation_numbers=[1]),
    ])
    review = Verification(evidence_support_percent=25, question_fully_answered=True, missing_information=[], verdicts=[
        Verdict(claim_index=0, supported=True, reason="Entailed", evidence=[Support(citation_number=1, quote="20 days leave")]),
        Verdict(claim_index=1, supported=True, reason="Wrong quote", evidence=[Support(citation_number=1, quote="30 days leave")]),
        Verdict(claim_index=2, supported=True, reason="Invalid source", evidence=[Support(citation_number=9, quote="20 days leave")]),
    ])
    assert [claim["supported"] for claim in assess_claims(draft, review, citations)] == [True, False, False, False]


def test_duplicate_verdicts_fail_closed():
    draft = Draft(answer="20 days.", claims=[Claim(text="20 days.", citation_numbers=[1])])
    verdict = Verdict(claim_index=0, supported=True, reason="ok", evidence=[Support(citation_number=1, quote="20 days")])
    result = assess_claims(draft, Verification(evidence_support_percent=0, question_fully_answered=True, missing_information=[], verdicts=[verdict, verdict]), [{"number": 1, "text": "20 days"}])
    assert result[0]["supported"] is False


def test_pdf_line_breaks_match_whitespace_normalized_quotes():
    text = (
        "4C. Section 43A-Private company to become public company in certain cases. -  \n"
        "The average annual turnover, for the purposes of sub-section (1A) of section 43A of the \n"
        "Act, shall not be less than rupees twenty-five crores."
    )
    draft = Draft(answer="The threshold is twenty-five crores.", claims=[Claim(text="The threshold is twenty-five crores.", citation_numbers=[1])])
    verdict = Verdict(claim_index=0, supported=True, reason="Entailed", evidence=[
        Support(citation_number=1, quote=text.replace("\n", " ")),
    ])
    result = assess_claims(draft, Verification(evidence_support_percent=100, question_fully_answered=True, missing_information=[], verdicts=[verdict]), [{"number": 1, "text": text}])[0]
    assert result["supported"] is True
    assert result["quote_check_passed"] is True
    verdict.evidence[0].quote = verdict.evidence[0].quote.replace("twenty-five", "thirty-five")
    rejected = assess_claims(draft, Verification(evidence_support_percent=0, question_fully_answered=True, missing_information=[], verdicts=[verdict]), [{"number": 1, "text": text}])[0]
    assert rejected["supported"] is False
    assert rejected["verifier_supported"] is True
    assert "Local validation failed" in rejected["reason"]


def test_answer_is_returned_even_when_verification_score_is_low(monkeypatch):
    draft = Draft(answer="20 days leave. [1] 30 days leave. [1]", claims=[
        Claim(text="20 days leave.", citation_numbers=[1]),
        Claim(text="30 days leave.", citation_numbers=[1]),
    ])
    review = Verification(evidence_support_percent=50, question_fully_answered=True, missing_information=[], verdicts=[
        Verdict(claim_index=0, supported=True, reason="Supported", evidence=[Support(citation_number=1, quote="20 days leave")]),
        Verdict(claim_index=1, supported=False, reason="Contradicted", evidence=[]),
    ])
    monkeypatch.setattr("backend.agent.nodes.verify_claims", lambda *args: review)
    nodes = WorkflowNodes(None, None, None, None)
    result = nodes.verify({"query": "Leave?", "draft": draft, "response": {
        "answer": draft.answer,
        "citations": [{"number": 1, "text": "20 days leave"}], "trace": {},
    }})["response"]
    assert "20 days leave. [1]" in result["answer"]
    assert "30 days leave. [1]" in result["answer"]
    assert result["confidence"] == 0.5
    assert result["evidence_support_percent"] == 50
    assert result["trace"]["verification"]["unsupported_withheld"] == 1


def test_supported_partial_claim_is_returned_with_missing_detail_warning(monkeypatch):
    draft = Draft(answer="Section 601 concerns fees for registering documents by foreign companies. [1]", claims=[Claim(
        text="Section 601 concerns fees for registering documents by foreign companies.",
        citation_numbers=[1],
    )])
    review = Verification(
		evidence_support_percent=35,
        question_fully_answered=False,
        missing_information=["The fee amount is Rs. 1,000."],
        verdicts=[Verdict(claim_index=0, supported=True, reason="The description is supported.", evidence=[
            Support(citation_number=1, quote="Section 601 - Fee for registration of document by foreign company")
        ])],
    )
    monkeypatch.setattr("backend.agent.nodes.verify_claims", lambda *args: review)
    nodes = WorkflowNodes(None, None, None, None)
    result = nodes.verify({"query": "What is the fee to register a foreign company?", "draft": draft, "response": {
        "citations": [{"number": 1, "text": "Section 601 - Fee for registration of document by foreign company. The fee shall be Rs. 1000/-"}],
        "trace": {},
    }})["response"]
    assert result["answer"].startswith("Section 601 concerns fees")
    assert "Evidence review" in result["answer"]
    assert "Rs. 1,000" in result["answer"]
    assert result["confidence"] == 0.35
    assert result["evidence_support_percent"] == 35
    assert result["citations"][0]["number"] == 1
    assert result["trace"]["verification"]["question_fully_answered"] is False


def test_empty_draft_gets_one_focused_generation_retry(monkeypatch):
    responses = [Draft(answer="", claims=[]), Draft(answer="The fee is Rs. 1,000.", claims=[
        Claim(text="The fee is Rs. 1,000.", citation_numbers=[1]),
    ])]
    calls = []
    def generate(question, citations, **kwargs):
        calls.append(kwargs)
        return responses.pop(0)
    monkeypatch.setattr("backend.agent.nodes.generate_claims", generate)
    nodes = WorkflowNodes(None, None, None, None)
    result = nodes.answer({
        "query": "What is the registration fee?", "session_id": "session", "mode": "hybrid",
        "evidence": [{"chunk_id": "chunk:0", "text": "The fee is Rs. 1,000."}],
        "retrieved": [],
    })
    assert len(calls) == 2
    assert calls == [{}, {"retry": True}]
    assert result["draft"].claims[0].text == "The fee is Rs. 1,000."
    assert result["response"]["trace"]["draft_retried"] is True


def test_answer_node_keeps_fee_at_end_of_retrieved_chunk(monkeypatch):
    text = "Filler text. " * 85 + "Section 601 fee for foreign company is Rs. 1000/-"
    captured = {}
    def generate(question, citations, **kwargs):
        captured["citations"] = citations
        return Draft(answer="The fee is Rs. 1,000. [1]", claims=[
            Claim(text="The fee is Rs. 1,000.", citation_numbers=[1]),
        ])
    monkeypatch.setattr("backend.agent.nodes.generate_claims", generate)
    nodes = WorkflowNodes(None, None, None, None)
    result = nodes.answer({
        "query": "What is the fee for registering a foreign company?", "session_id": "session",
        "mode": "hybrid", "retrieved": [],
        "evidence": [{"chunk_id": "fee:0", "text": text}],
    })
    assert "Rs. 1000/-" in captured["citations"][0]["text"]
    assert len(captured["citations"][0]["text"]) == len(text)
    assert "Rs. 1,000" in result["response"]["answer"]


def test_generation_and_retry_prompts_only_use_retrieved_evidence(monkeypatch):
    prompts = []
    def call(schema, instruction, payload):
        prompts.append(instruction.lower())
        return Draft(answer="Fee is Rs. 1,000. [1]", claims=[])
    monkeypatch.setattr("backend.agent.gemini.structured_call", call)
    generate_claims("What is the fee?", [{"number": 1, "text": "Fee is Rs. 1,000."}])
    generate_claims("What is the fee?", [{"number": 1, "text": "Fee is Rs. 1,000."}], retry=True)
    assert len(prompts) == 2
    assert all("only the numbered evidence" in prompt or "using only those passages" in prompt for prompt in prompts)
    assert all("do not use general knowledge" in prompt for prompt in prompts)


def test_no_evidence_refuses_without_calling_gemini():
    nodes = WorkflowNodes(None, None, None, None)
    result = nodes.verify({"draft": None, "response": {"citations": [], "trace": {}}})["response"]
    assert result["confidence"] == 0
    assert result["citations"] == []
    assert "couldn't generate" in result["answer"]