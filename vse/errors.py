from __future__ import annotations


class VectorSearchError(Exception):
    """Base for everything this package raises."""


class ConfigError(VectorSearchError):
    """A parameter that cannot mean anything.

    A negative neighbour count, a probe count larger than the partition count, a subspace count
    that does not divide the dimension. These are caught where the value arrives rather than
    where it eventually produces an empty result, because an index that silently returns fewer
    neighbours than asked for is the hardest kind of bug to notice.
    """


class DataError(VectorSearchError):
    """Vectors that do not fit the index they are being handed to."""


class IndexStateError(VectorSearchError):
    """An index used in a state it is not in.

    Searched before it was built, inserted into after it was frozen, asked for a vector it does
    not hold. Not called IndexError, because the builtin of that name means something entirely
    different and shadowing it inside this package would make every traceback ambiguous.
    """


class BuildError(VectorSearchError):
    """A construction that could not finish.

    Clustering that could not fill its partitions, a graph that came out disconnected, a
    codebook trained on fewer points than it has centroids. Each of these produces an index that
    works and answers badly, which is worse than one that refuses.
    """
