from functools import lru_cache


def mex(reachable: list[int]) -> int:
    """
    Compute the minimum excludant (mex) of a set of numbers.

    :param reachable: set of integers
    :return: int, minimum non-negative integer not in the set
    """
    mex_value = 0
    while mex_value in reachable:
        mex_value += 1
    return mex_value


@lru_cache(None)
def compute_sprague_grundy(
        game_type: str,
        S: list[int],
        max_n: int
        ) -> list[int]:
    """
    Compute Sprague-Grundy values for a given game type (SUBTRACTION or ALLBUT)
        and set S up to max_n.

    :param game_type: str, either "SUBTRACTION" or "ALLBUT"
    :param S: set of allowed moves
    :param max_n: int, maximum number to compute values for
    :return: list of Sprague-Grundy values from 0 to max_n
    """
    grundy = [0] * (max_n + 1)

    for n in range(1, max_n + 1):
        reachable = set()

        if game_type == "SUBTRACTION":
            reachable = {
                grundy[n - move]
                for move in S
                if n - move >= 0
                }
        elif game_type == "ALLBUT":
            reachable = {
                grundy[n - move]
                for move in range(1, n + 1)
                if move not in S and n - move >= 0
                }
        else:
            msg = "Invalid game type. Choose 'SUBTRACTION' or 'ALLBUT'."
            raise ValueError(msg)

        grundy[n] = mex(reachable)

    return grundy


@lru_cache(None)
def find_pre_and_period_length(sequence: list[int]) -> tuple[int, int]:
    """
    Finds the length of the pre-sequence (pre-period) and the period in a seq.

    Args:
        sequence (list[int]): List of integers to analyze.

    Returns:
        tuple[int, int]: Length of the pre-sequence and length of the
            periodic sequence.
    """
    n = len(sequence)

    for pre_length in range(n):
        for period_length in range(1, (n - pre_length) // 2 + 1):
            is_periodic = True
            for i in range(pre_length, n):
                if (
                    (i + period_length < n)
                    and sequence[i] != sequence[i + period_length]
                ):
                    is_periodic = False
                    break

            if is_periodic:
                return pre_length, period_length

    return n, 0
