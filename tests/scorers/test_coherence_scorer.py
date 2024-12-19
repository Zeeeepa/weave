import pytest
from unittest.mock import MagicMock

import weave
from weave.scorers.llm_utils import download_model
from weave.scorers.coherence_scorer import CoherenceScorer
from tests.scorers.test_utils import TINY_MODEL_PATHS

@pytest.fixture
def coherence_scorer():
    """Fixture to return a CoherenceScorer instance."""
    tiny_model_path = download_model(TINY_MODEL_PATHS["coherence_scorer"])
    scorer = CoherenceScorer(
        model_name_or_path=tiny_model_path,
        device="cpu",
    )
    return scorer

def test_score_messages(coherence_scorer):
    """Test score_messages with a coherent response."""
    prompt = "This is a test prompt."
    output = "This is a coherent response."
    result = coherence_scorer.score_messages(prompt, output)
    # Now we check the updated payload structure
    assert result["flagged"] is True
    assert result["extras"]["coherence_label"] == 'A Little Incoherent'


@pytest.mark.asyncio
async def test_score_with_chat_history(coherence_scorer):
    """Test the async .score method with chat history."""
    prompt = "This is a test prompt."
    output = "This is a coherent response."
    chat_history = [
        {"role": "user", "text": "Hello"},
        {"role": "assistant", "text": "Hi"},
    ]
    result = await coherence_scorer.score(prompt, output, chat_history=chat_history)
    assert result["flagged"]
    assert result["extras"]["coherence_label"] == 'A Little Incoherent'

@pytest.mark.asyncio
async def test_score_with_context(coherence_scorer):
    """Test the async .score method with additional context."""
    prompt = "This is a test prompt."
    output = "This is a coherent response."
    context = "This is additional context."
    result = await coherence_scorer.score(prompt, output, context=context)
    assert result["flagged"]
    assert result["extras"]["coherence_label"] == 'A Little Incoherent'