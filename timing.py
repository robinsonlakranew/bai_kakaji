
import time
def timed(fn):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        out["time_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        return out
    return wrapper
