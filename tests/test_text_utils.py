from vgj_chat.utils.text import drop_last_incomplete_paragraph, strip_metadata


def test_strip_metadata_removes_sections():
    raw = (
        "Dinner spot recommendations.\n" "Reference: [1] source\n" "URL: example.com\n"
    )
    assert strip_metadata(raw) == "Dinner spot recommendations."


def test_strip_metadata_handles_question_blocks():
    raw = "Great food available.\nQuestion: where?\nAnswer: here"
    assert strip_metadata(raw) == "Great food available."


def test_drop_last_incomplete_paragraph_removes_truncated():
    text = "First paragraph.\n\nSecond paragraph unfinished"
    assert drop_last_incomplete_paragraph(text) == "First paragraph."


def test_drop_last_incomplete_paragraph_keeps_complete():
    text = "First paragraph.\n\nSecond paragraph finished."
    assert drop_last_incomplete_paragraph(text) == text
