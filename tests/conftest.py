import pytest


def _cuda_available() -> bool:
    try:
        import warp as wp

        wp.config.quiet = True
        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    gpu_items = [item for item in items if "gpu" in item.keywords]
    if gpu_items and not _cuda_available():
        skip = pytest.mark.skip(reason="no CUDA device")
        for item in gpu_items:
            item.add_marker(skip)


@pytest.fixture
def device():
    return "cuda:0"
