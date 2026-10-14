"""Finite log-relaxation cells and the manuscript's prescribed binary hierarchy."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid:
    t1: np.ndarray
    t2: np.ndarray

    def __post_init__(self):
        for key in ('t1', 't2'):
            value = np.asarray(getattr(self, key), dtype=np.float64).copy()
            if value.ndim != 1 or value.size < 2 or not np.all(np.isfinite(value)) or np.any(np.diff(value) <= 0):
                raise ValueError('Grid edges must be finite, strictly increasing log10 seconds.')
            value.setflags(write=False)
            object.__setattr__(self, key, value)

    @property
    def shape(self):
        return len(self.t1) - 1, len(self.t2) - 1

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def areas(self):
        return np.outer(np.diff(self.t1), np.diff(self.t2)).ravel()

    @property
    def centers(self):
        a, b = np.meshgrid((self.t1[:-1] + self.t1[1:]) / 2,
                           (self.t2[:-1] + self.t2[1:]) / 2, indexing='ij')
        return np.column_stack((a.ravel(), b.ravel()))

    def uniform_partition(self, shape):
        n, m = self.shape
        p, q = map(int, shape)
        if p < 1 or q < 1 or n % p or m % q:
            raise ValueError('Partition dimensions must divide the reference grid.')
        return Partition(self, tuple((i, i+n//p, j, j+m//q)
                         for i in range(0, n, n//p) for j in range(0, m, m//q)))


@dataclass(frozen=True)
class Partition:
    grid: Grid
    # Half-open integer rectangles on the fixed reference grid.
    cells: tuple

    def __post_init__(self):
        coverage = np.zeros(self.grid.shape, dtype=int)
        for i, k, j, l in self.cells:
            if not (0 <= i < k <= self.grid.shape[0] and 0 <= j < l <= self.grid.shape[1]):
                raise ValueError('Invalid partition rectangle.')
            coverage[i:k, j:l] += 1
        if not np.all(coverage == 1):
            raise ValueError('Cells must cover each reference cell exactly once.')

    @property
    def size(self):
        return len(self.cells)

    @property
    def groups(self):
        index = np.arange(self.grid.size).reshape(self.grid.shape)
        return tuple(index[i:k, j:l].ravel() for i, k, j, l in self.cells)

    @property
    def bounds(self):
        return np.array([(self.grid.t1[i], self.grid.t1[k], self.grid.t2[j], self.grid.t2[l])
                         for i, k, j, l in self.cells])

    def aggregate(self, reference_values):
        x = np.asarray(reference_values)
        if x.shape[-1] != self.grid.size:
            raise ValueError('Last axis must index the fixed reference cells.')
        return np.stack([x[..., g].sum(axis=-1) for g in self.groups], axis=-1)

    def split(self, index, axis):
        i, k, j, l = self.cells[index]
        if axis == 0 and (k-i) % 2 == 0:
            mid = (i+k)//2
            children = ((i, mid, j, l), (mid, k, j, l))
        elif axis == 1 and (l-j) % 2 == 0:
            mid = (j+l)//2
            children = ((i, k, j, mid), (i, k, mid, l))
        else:
            raise ValueError('Binary refinement requires an even cell count along that axis.')
        return Partition(self.grid, self.cells[:index] + children + self.cells[index+1:])

    def to_reference(self, mass):
        if self.size != self.grid.size:
            raise ValueError('Only a fully refined state can be returned on the reference grid.')
        out = np.empty_like(mass)
        for i, group in enumerate(self.groups):
            out[..., group[0]] = mass[..., i]
        return out


def aggregation_matrix(fine, coarse):
    if fine.grid.shape != coarse.grid.shape or not np.array_equal(fine.grid.t1, coarse.grid.t1) or not np.array_equal(fine.grid.t2, coarse.grid.t2):
        raise ValueError('Partitions must use the same reference grid.')
    matrix = np.zeros((coarse.size, fine.size))
    for j, (a, b, c, d) in enumerate(fine.cells):
        owners = [i for i, (e, f, g, h) in enumerate(coarse.cells)
                  if e <= a < b <= f and g <= c < d <= h]
        if len(owners) != 1:
            raise ValueError('The requested partitions are not nested.')
        matrix[owners[0], j] = 1
    return matrix


def refinement_stage(partition):
    """Yield (before, after, parent index), T1 then both T2 children per parent."""
    current = partition
    for parent in sorted(partition.cells):
        index = current.cells.index(parent)
        following = current.split(index, 0)
        yield current, following, index
        current = following
        # After splitting the first T1 child in T2, the second is at index+2.
        for child in (index, index+2):
            following = current.split(child, 1)
            yield current, following, child
            current = following


def hierarchy(grid, shapes):
    shapes = [tuple(s) for s in shapes]
    if not shapes or shapes[-1] != grid.shape:
        raise ValueError('The schedule must finish at the reference grid.')
    current = grid.uniform_partition(shapes[0])
    stages = []
    for old, new in zip(shapes, shapes[1:]):
        if new != (old[0]*2, old[1]*2):
            raise ValueError('Each prescribed stage must double both dimensions.')
        stage = list(refinement_stage(current))
        stages.append(stage)
        current = stage[-1][1]
    return grid.uniform_partition(shapes[0]), stages
