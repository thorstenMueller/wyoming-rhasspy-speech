# Based on: https://github.com/nltk/nltk
from collections.abc import Collection, Sequence
from typing import Dict, Iterable, List, Optional, Set


def edit_distance(
    words1: Sequence[str],
    words2: Sequence[str],
    substitution_cost: int = 1,
    skip_words: Optional[Collection[str]] = None,
) -> int:
    """
    Calculate the Levenshtein edit-distance between two lists of words.  The
    edit distance is the number of words that need to be substituted, inserted,
    or deleted, to transform words1 into words2. "Skip" words do not have any
    cost.

    Allows specifying the cost of substitution edits (e.g., "a" -> "b"),
    because sometimes it makes sense to assign greater penalties to
    substitutions.
    """
    # set up a 2-D array
    len1 = len(words1)
    len2 = len(words2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # retrieve alphabet
    sigma: Set[str] = set()
    sigma.update(words1)
    sigma.update(words2)

    # set up table to remember positions of last seen occurrence in words1
    last_left_t = _last_left_t_init(sigma)

    # iterate over the array
    # i and j start from 1 and not 0 to stay close to the wikipedia pseudo-code
    # see https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
    for i in range(1, len1 + 1):
        last_right_buf = 0
        for j in range(1, len2 + 1):
            last_left = last_left_t[words2[j - 1]]
            last_right = last_right_buf
            if words1[i - 1] == words2[j - 1]:
                last_right_buf = j
            _edit_dist_step(
                lev,
                i,
                j,
                words1,
                words2,
                last_left,
                last_right,
                substitution_cost=substitution_cost,
                skip_words=skip_words,
            )
        last_left_t[words1[i - 1]] = i
    return lev[len1][len2]


def _edit_dist_step(
    lev: List[List[int]],
    i: int,
    j: int,
    words1: Sequence[str],
    words2: Sequence[str],
    last_left: int,
    last_right: int,
    substitution_cost: int = 1,
    skip_words: Optional[Collection[str]] = None,
):
    c1 = words1[i - 1]
    c2 = words2[j - 1]

    c1_cost = 1
    c2_cost = 1

    if skip_words:
        # No cost
        if c1 in skip_words:
            c1_cost = 0
            substitution_cost = 0

        if c2 in skip_words:
            c2_cost = 0
            substitution_cost = 0

    # skipping a character in s1
    a = lev[i - 1][j] + c1_cost
    # skipping a character in s2
    b = lev[i][j - 1] + c2_cost
    # substitution
    c = lev[i - 1][j - 1] + (substitution_cost if c1 != c2 else 0)

    # pick the cheapest
    lev[i][j] = min(a, b, c)


def _edit_dist_init(len1: int, len2: int) -> List[List[int]]:
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i  # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j  # row 0: 0,1,2,3,4,...
    return lev


def _last_left_t_init(sigma: Iterable[str]) -> Dict[str, int]:
    return {c: 0 for c in sigma}
