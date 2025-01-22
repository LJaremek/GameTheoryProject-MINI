from numba import jit
import numpy as np
from functools import lru_cache
from tools import compute_sprague_grundy as compute_sprague_grundy_slow, find_pre_and_period_length

@jit(nopython=True)
def mex(reachable: np.ndarray, size: int) -> int:
    """
    Find the minimum excludant (smallest non-negative integer not in the set).
    Uses a boolean array instead of a set for Numba compatibility.
    """
    # We know mex can't be larger than the size of reachable set + 1
    for i in range(size + 1):
        if not reachable[i]:
            return i
    return size + 1

@jit(nopython=True)
def compute_sprague_grundy_numba(
        S: np.ndarray,
        max_n: int
        ) -> np.ndarray:
    grundy = np.zeros(max_n + 1, dtype=np.int32)
    reachable = np.zeros(len(S)+1, dtype=np.bool_)
    for n in range(1, max_n + 1):
        reachable.fill(False)
        max_reachable = 0
        for move in S:
            if n - move >= 0:
                g_value = grundy[n - move]
                reachable[g_value] = True
                max_reachable = max(max_reachable, g_value)        
        grundy[n] = mex(reachable, max_reachable)
    return grundy


@jit(nopython=True)
def my_mex(reachable: np.ndarray) -> int:
    size = len(reachable)
    reachable = np.sort(reachable)
    start_idx = 0
    while reachable[start_idx] == -1:
        start_idx += 1
    if reachable[start_idx] != 0:
        return 0
    start_idx += 1
    for i in range(start_idx, size):
        if reachable[i] != reachable[i-1] and reachable[i] != reachable[i-1] + 1:
            return reachable[i-1] + 1
    return reachable[size-1] + 1
    
    
def compute_sprague_grundy(
        S: list[int],
        max_n: int
        ) -> list[int]:
    
    S_array = np.array(sorted(S), dtype=np.int32)
    result = compute_sprague_grundy_numba(S_array, max_n)
    return result.tolist()


def find_period(grundy_values: list[int]) -> tuple[int, int | None]:
    n = len(grundy_values)

    for pre_period in range(n):
        for period in range(1, n - pre_period):
            is_periodic = True
            for i in range(pre_period, n - period):
                if grundy_values[i] != grundy_values[i + period]:
                    is_periodic = False
                    break

            if is_periodic:
                return pre_period, period

    return len(grundy_values), None


def analyze_periodicity(
        grundy_values: list[int]
        ) -> tuple[int, int | None, int | None]:
    n = len(grundy_values)

    for pre_period in range(n):
        for period in range(1, n - pre_period):
            if (
                grundy_values[pre_period:pre_period + period]
                == grundy_values[pre_period + period:pre_period + 2 * period]
            ):
                saltus = None
                is_arithmetic = True

                for i in range(pre_period, n - period):
                    if (
                        (grundy_values[i] + grundy_values[i + period])
                        != grundy_values[i]
                    ):
                        is_arithmetic = False
                        break

                if is_arithmetic:
                    saltus = (
                        grundy_values[pre_period + period]
                        - grundy_values[pre_period]
                    )

                return pre_period, period, saltus

    return len(grundy_values), None, None



@jit(nopython=True)
def find_period_fast(grundy_values: np.ndarray, max_subtraction: int) -> tuple:
    n = len(grundy_values)

    if n < 2 * max_subtraction:
        return (-1, -1)
    for l in range(n - max_subtraction):
        for p in range(1, n - l - max_subtraction + 1):
            is_periodic = True
            for offset in range(max_subtraction):
                if l + offset >= n or l + offset + p >= n:
                    is_periodic = False
                    break
                    
                if grundy_values[l + offset] != grundy_values[l + offset + p]:
                    is_periodic = False
                    break
            
            if is_periodic:
                return (l, p)
                
    return (-1, -1)


from multiprocessing import Pool, cpu_count
from itertools import product
from tqdm import tqdm

def compute_single_pair(args):
    """
    Compute period for a single (a,b) pair
    
    Args:
        args: Tuple containing (a, b, S_base, max_n)
    Returns:
        Tuple of (a, b, period)
    """
    a, b, S_base, max_n = args
    S = S_base | {a, b}
    max_S = max(S)
    k = len(S)
    
    # Compute Sprague-Grundy sequence
    g = compute_sprague_grundy(S, max_n)
    
    
    # Find period
    pre_period, period = find_period_fast(np.array(g), max_S)
    
    return (a, b, period)

def parallel_period_computation(S_base, m, n, max_n, num_processes=None):
    """
    Parallel computation of period map
    
    Args:
        S_base: Base set of numbers
        m, n: Dimensions of output matrix
        max_n: Maximum n for Sprague-Grundy computation
        num_processes: Number of processes to use (defaults to CPU count)
    
    Returns:
        period_map: n x m matrix of periods
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    # Create argument list for all pairs
    args_list = []
    for a in range(1, n):
        for b in range(a+1, m):
            if a in S_base or b in S_base:
                continue
            args_list.append((a, b, S_base, max_n))
    
    # Create process pool and run computations
    period_map = np.zeros((n, m), dtype=float)
    
    with Pool(processes=num_processes) as pool:
        # Use tqdm to show progress bar
        results = list(tqdm(
            pool.imap(compute_single_pair, args_list),
            total=len(args_list),
            desc="Computing periods"
        ))
        
        # Fill in results matrix
        for a, b, period in results:
            period_map[a, b] = period
    
    # Add symmetric entries
    period_map += period_map.T
    
    return period_map