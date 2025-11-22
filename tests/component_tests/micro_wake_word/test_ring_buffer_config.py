"""Tests for micro_wake_word ring_buffer_duration configuration."""


def test_ring_buffer_duration_custom(generate_main):
    """
    When ring_buffer_duration is set to 480ms, it should generate the setter call.
    """
    # When
    main_cpp = generate_main(
        "tests/component_tests/micro_wake_word/test_micro_wake_word.yaml"
    )

    # Then
    # Should have call to set_ring_buffer_duration with custom value (480ms = 480)
    assert "->set_ring_buffer_duration(480)" in main_cpp
