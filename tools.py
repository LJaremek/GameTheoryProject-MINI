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


def find_period(grundy_values: list[int]) -> tuple[int, int | None]:
    """
    Find the period and pre-period of the Sprague-Grundy values.

    :param grundy_values: list of integers representing the Sprague-Grundy vals
    :return: tuple (pre_period, period) where:
             - pre_period is the length before periodicity starts
             - period is the length of the repeating cycle
    """
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
    """
    Analyze the periodicity and pre-period of Sprague-Grundy values.

    :param grundy_values: list of Sprague-Grundy values
    :return: tuple (pre_period, period, saltus), where:
             - pre_period is the length before periodicity starts
             - period is the length of the repeating cycle
             - saltus is the arithmetic difference in the periodic sequence
                (if applicable)
    """
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
