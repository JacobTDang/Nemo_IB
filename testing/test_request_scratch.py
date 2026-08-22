"""Per-request scratch space that always cleans up.

No tool in the image writes a file today -- verified, none of the five servers
accepts a file path. But that is a property of the current tool set, not a
guarantee, and the safety net that used to catch this is gone: with stdio and
--rm, anything a tool left behind died with the container. A long-lived HTTP
server keeps it.

Cleanup must survive the error path. That is where leaks actually come from --
the happy path is the one people remember to clean up.
"""
import pathlib

import pytest

from tools.mcp_http import request_scratch


def test_directory_exists_inside_the_block():
    with request_scratch() as scratch:
        assert scratch.is_dir()
        (scratch / "work.txt").write_text("x")
        assert (scratch / "work.txt").exists()


def test_directory_is_removed_afterwards():
    with request_scratch() as scratch:
        (scratch / "work.txt").write_text("x")
        captured = scratch
    assert not captured.exists()


def test_directory_is_removed_when_the_body_raises():
    """The case that matters. A tool that fails halfway must not leave its
    partial output behind on a server that never restarts."""
    captured = None
    with pytest.raises(ValueError):
        with request_scratch() as scratch:
            captured = scratch
            (scratch / "partial.bin").write_bytes(b"half a file")
            raise ValueError("tool failed mid-write")
    assert captured is not None
    assert not captured.exists(), "scratch survived an exception"


def test_nested_content_is_removed_too():
    with request_scratch() as scratch:
        (scratch / "a" / "b").mkdir(parents=True)
        (scratch / "a" / "b" / "deep.txt").write_text("y")
        captured = scratch
    assert not captured.exists()


def test_each_request_gets_its_own_directory():
    with request_scratch() as first:
        with request_scratch() as second:
            assert first != second


def test_removal_is_idempotent():
    """A tool that cleans up after itself must not break the context manager."""
    import shutil
    with request_scratch() as scratch:
        shutil.rmtree(scratch)
    # exiting the block over an already-removed directory must not raise
