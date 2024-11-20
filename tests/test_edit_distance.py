from wyoming_rhasspy_speech.edit_distance import edit_distance


def test_edit_distance() -> None:
    # Delete
    assert edit_distance(["this", "is", "a", "test"], ["this", "is", "test"]) == 1

    # Add
    assert edit_distance(["this", "is"], ["this", "is", "a", "test"]) == 2

    # Skip words
    assert (
        edit_distance(
            ["this", "is", "test"], ["this", "is", "a", "test"], skip_words={"a"}
        )
        == 0
    )
