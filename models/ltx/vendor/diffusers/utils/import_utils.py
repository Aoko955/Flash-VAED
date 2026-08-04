def is_accelerate_available() -> bool:
    try:
        import accelerate
        return True
    except Exception:
        return False


def is_torch_npu_available() -> bool:
    return False


def is_torch_version(operation: str, version: str) -> bool:

    return True
