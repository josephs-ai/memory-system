"""Pytest session-scoped fixtures for memory-index test suite.

Ensures database connection pools are cleanly shut down after all tests
complete, eliminating the 'couldn't stop thread pool-1-worker-X' warnings.
"""

import pytest


@pytest.fixture(autouse=True, scope="session")
def _cleanup_pools():
    """Yield control to the test session, then close all DB pools on teardown."""
    yield
    # Close memory_db pool if it was imported
    try:
        from memory_db import close_pool as close_memory_pool
        close_memory_pool()
    except Exception:
        pass
    # Close checkpoint_db pool if it was imported
    try:
        from checkpoint_db import close_pool as close_checkpoint_pool
        close_checkpoint_pool()
    except Exception:
        pass
