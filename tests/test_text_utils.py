from vgj_chat.utils.text import strip_metadata


def test_strip_metadata_removes_sections():
    raw = (
        "Dinner spot recommendations.\n" "Reference: [1] source\n" "URL: example.com\n"
    )
    assert strip_metadata(raw) == "Dinner spot recommendations."


def test_strip_metadata_handles_question_blocks():
    raw = "Great food available.\nQuestion: where?\nAnswer: here"
    assert strip_metadata(raw) == "Great food available."
