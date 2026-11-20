import pytest

from backend.agent.gemini import Claim, Draft, Support, Verdict, Verification


@pytest.fixture
def fake_gemini(monkeypatch):
    def generate(question, citations):
        claims = [
            Claim(text=item["text"], citation_numbers=[item["number"]]) for item in citations
        ]
        answer = "\n".join(f"{claim.text} [{claim.citation_numbers[0]}]" for claim in claims)
        return Draft(answer=answer, claims=claims)

    def verify(question, draft, citations):
        return Verification(evidence_support_percent=100, question_fully_answered=True, missing_information=[], verdicts=[Verdict(
            claim_index=index, supported=True, reason="Verbatim evidence",
            evidence=[Support(citation_number=claim.citation_numbers[0], quote=claim.text)],
        ) for index, claim in enumerate(draft.claims)])

    monkeypatch.setattr("backend.agent.nodes.generate_claims", generate)
    monkeypatch.setattr("backend.agent.nodes.verify_claims", verify)