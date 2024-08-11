import pytest
from models.object_extraction import summarize_text

def test_summarize_text():
    text = "This is a sample text to test the summarization model."
    summary = summarize_text(text)
    assert isinstance(summary, str) and len(summary) > 0
