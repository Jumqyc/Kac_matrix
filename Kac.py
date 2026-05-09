"""Kac matrix (Gram matrix) for the Virasoro algebra Verma module."""

import sympy as sp
from itertools import product
from typing import Iterator, Tuple
import functools


def partitions(n: int) -> Iterator[Tuple[int, ...]]:
    """Generate all integer partitions of n in descending (non‑increasing) order."""
    if n == 0:
        yield ()
        return
    for p in range(1, n + 1):
        for sub in partitions(n - p):
            if not sub or p >= sub[0]:
                yield (p,) + sub


# Symbols
h = sp.symbols('h', positive=True)
c = sp.symbols('c', positive=True)


@functools.lru_cache(maxsize=None)
def _gram_coeff(I: Tuple[int, ...]) -> sp.Expr:
    """
    Compute the coefficient <h| L_{i_1} ... L_{i_k} |h> for a tuple of mode indices,
    using the Virasoro algebra commutation relations.  Returns 0 if the total
    level does not vanish or the ordering is invalid.
    """
    # Base cases
    if not I:
        return sp.Integer(1)
    if I[-1] > 0:          # annihilation operator on the far right → 0
        return sp.Integer(0)
    if I[0] < 0:           # creation operator on the far left → 0
        return sp.Integer(0)
    if sum(I) != 0:        # level mismatch
        return sp.Integer(0)
    if I[-1] == 0:         # L_0 → h
        return h * _gram_coeff(I[:-1])

    # Find the first adjacent pair (L_a L_b) with a > 0 and b ≤ 0
    for p in range(len(I) - 1):
        a = I[p]
        b = I[p + 1]
        if a <= 0 or b > 0:
            continue

        # Swap: L_a L_b → L_b L_a
        swapped = I[:p] + (b, a) + I[p + 2:]
        term1 = _gram_coeff(swapped)

        # Commutator: (a - b) L_{a+b}
        new_seq = I[:p] + (a + b,) + I[p + 2:]
        term2 = (a - b) * _gram_coeff(new_seq)

        # Central term: (c/12) a(a^2 - 1) δ_{a+b,0}
        if a + b == 0:
            center_seq = I[:p] + I[p + 2:]
            term3 = (c * (a**3 - a) / 12) * _gram_coeff(center_seq)
            return sp.simplify(term1 + term2 + term3)
        else:
            return sp.simplify(term1 + term2)

    # No crossing found (should not happen for valid input)
    return sp.Integer(0)


def kac_matrix(level: int) -> sp.Matrix:
    """
    Return the symbolic Kac (Gram) matrix of the Virasoro Verma module
    at the given level.
    """
    basis = list(partitions(level))
    n = len(basis)
    mat = sp.zeros(n, n)

    for i in range(n):
        p1 = basis[i]
        # Positive modes in ascending order (creation operators on the bra side)
        pos_tuple = p1[::-1]
        for j in range(i, n):
            p2 = basis[j]
            # String: L_{pos} L_{-p2}
            I = pos_tuple + tuple(-x for x in p2)
            coeff = _gram_coeff(I)
            mat[i, j] = coeff
            if i != j:
                mat[j, i] = coeff   # the Gram matrix is symmetric

    return sp.simplify(mat)


def kac_matrix_str(level: int) -> str:
    """Return a pretty-printed string of the Kac matrix."""
    return sp.pretty(kac_matrix(level), wrap_line=False)


def kac_matrix_strings(level: int) -> list[list[str]]:
    """Return the Kac matrix as a 2D list of strings (using '^' and no '*')."""
    M = kac_matrix(level)
    rows = []
    for i in range(M.rows):
        row = []
        for j in range(M.cols):
            s = str(M[i, j])
            s = s.replace('**', '^').replace('*', '')
            row.append(s)
        rows.append(row)
    return rows


if __name__ == "__main__":
    import sys
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"Kac matrix for level {level}:")
    sp.pprint(kac_matrix(level))
    print("\nString matrix (with '^' and no '*'):")
    for row in kac_matrix_strings(level):
        print(row)