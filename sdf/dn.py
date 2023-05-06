import itertools
import numpy as np

_min = np.minimum
_max = np.maximum


def union(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, "_k", None)
            if K is None:
                d1 = _min(d1, d2)
            else:
                h = np.clip(0.5 + 0.5 * (d2 - d1) / K, 0, 1)
                m = d2 + (d1 - d2) * h
                d1 = m - K * h * (1 - h)
        return d1

    return f


def difference(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, "_k", None)
            if K is None:
                d1 = _max(d1, -d2)
            else:
                h = np.clip(0.5 - 0.5 * (d2 + d1) / K, 0, 1)
                m = d1 + (-d2 - d1) * h
                d1 = m + K * h * (1 - h)
        return d1

    return f


def intersection(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, "_k", None)
            if K is None:
                d1 = _max(d1, d2)
            else:
                h = np.clip(0.5 - 0.5 * (d2 - d1) / K, 0, 1)
                m = d2 + (d1 - d2) * h
                d1 = m + K * h * (1 - h)
        return d1

    return f


def blend(a, *bs, k=0.5):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, "_k", None)
            d1 = K * d2 + (1 - K) * d1
        return d1

    return f


def negate(other):
    def f(p):
        return -other(p)

    return f


def dilate(other, r):
    def f(p):
        return other(p) - r

    return f


def erode(other, r):
    def f(p):
        return other(p) + r

    return f


def shell(other, thickness=1, type="center"):
    """
    Keep only a margin of a given thickness around the object's boundary.

    Args:
        thickness (float): the resulting thickness
        type (str): what kind of shell to generate.

            ``"center"`` (default)
                shell is spaced symmetrically around boundary
            ``"outer"``
                the resulting shell will be ``thickness`` larger than before
            ``"inner"``
                the resulting shell will be as large as before
    """
    return dict(
        center=lambda p: np.abs(other(p)) - thickness / 2,
        inner=other - other.erode(thickness),
        outer=other.dilate(thickness) - other,
    )[type]


def mirror(other, direction, at=0):
    """
    Mirror around a given plane defined by ``origin`` reference point and
    ``direction``.

    Args:
        direction (3D vector): direction to mirror to (e.g. :any:`X` to mirror along X axis)
        at (3D vector): point to mirror at. Default is the origin.
    """
    direction = direction / np.linalg.norm(direction)

    def f(p):
        projdir = np.expand_dims((p - at) @ direction, axis=1) * direction
        # mirrored point:
        # - project 'p' onto 'direction' (result goes into 'projdir' direction)
        # - projected point is at   'at + projdir'
        # - remember direction from projected point to the original point (p - (at + projdir))
        # - from origin 'at' go backwards the projected direction (at - projdir)
        # - from that target, move along the remembered direction (p - (at + projdir))
        # - pmirr = at - projdir + (p - (at + projdir))
        # - the 'at' cancels out, the projdir is subtracted twice from the point
        return other(p - 2 * projdir)

    return f


def repeat(other, spacing, count=None, padding=0):
    count = np.array(count) if count is not None else None
    spacing = np.array(spacing)

    def neighbors(dim, padding, spacing):
        try:
            padding = [padding[i] for i in range(dim)]
        except Exception:
            padding = [padding] * dim
        try:
            spacing = [spacing[i] for i in range(dim)]
        except Exception:
            spacing = [spacing] * dim
        for i, s in enumerate(spacing):
            if s == 0:
                padding[i] = 0
        axes = [list(range(-p, p + 1)) for p in padding]
        return list(itertools.product(*axes))

    def f(p):
        q = np.divide(p, spacing, out=np.zeros_like(p), where=spacing != 0)
        if count is None:
            index = np.round(q)
        else:
            index = np.clip(np.round(q), -count, count)

        indexes = [index + n for n in neighbors(p.shape[-1], padding, spacing)]
        A = [other(p - spacing * i) for i in indexes]
        a = A[0]
        for b in A[1:]:
            a = _min(a, b)
        return a

    return f
