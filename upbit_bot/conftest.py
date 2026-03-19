"""
conftest.py — pytest 전역 설정 + numba mock (Python 3.14 호환)

pandas-ta는 numba를 선택적으로 사용하지만 import 시 로드함.
Python 3.14에서 numba 미지원 → 테스트용 stub 제공.
실제 운영 환경(Python 3.11)에서는 numba 정상 설치됨.
"""
import sys
import types
from unittest.mock import MagicMock

# ------------------------------------------------------------------
# numba stub — Python 3.14 테스트 환경 전용
# ------------------------------------------------------------------

def _make_numba_stub() -> types.ModuleType:
    """numba의 njit, jit 데코레이터를 no-op으로 대체."""
    stub = types.ModuleType("numba")

    def _passthrough(func=None, **kwargs):
        """@njit, @jit 데코레이터 → 원본 함수 그대로 반환."""
        if func is not None:
            return func
        def decorator(f):
            return f
        return decorator

    stub.njit = _passthrough
    stub.jit = _passthrough
    stub.prange = range
    stub.float64 = float
    stub.int64 = int
    stub.boolean = bool
    stub.types = MagicMock()
    stub.typed = MagicMock()
    stub.core = MagicMock()
    return stub


if "numba" not in sys.modules:
    sys.modules["numba"] = _make_numba_stub()

# pyupbit stub — Python 3.14에서 numba 의존성으로 설치 불가
if "pyupbit" not in sys.modules:
    pyupbit_stub = types.ModuleType("pyupbit")
    pyupbit_stub.get_ohlcv = MagicMock(return_value=None)
    pyupbit_stub.get_current_price = MagicMock(return_value=None)
    sys.modules["pyupbit"] = pyupbit_stub
