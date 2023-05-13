import itertools
import numpy as np
from . import ease

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


def modulate_between(sdf, a, b, e=ease.in_out_cubic):
    """
    Apply a distance offset transition between two control points
    (e.g. make a rod thicker or thinner at some point or add a bump)

    Args:
        a, b (vectors): the two control points
        e (scalar function): the distance offset function, will be called with
            values between 0 (at control point ``a``) and 1 (at control point
            ``b``).  Its result will be subtracted from the given SDF, thus
            enlarging the object by that value.
    """

    # unit vector from control point a to b
    ab = (ab := b - a) / (L := np.linalg.norm(ab))

    def f(p):
        # project current point onto control direction, clip and apply easing
        offset = e(np.clip((p - a) @ ab / L, 0, 1))
        return (dist := sdf(p)) - offset.reshape(dist.shape)

    return f


def stretch(sdf, a, b, e=ease.linear, k=None):
    """
    Grab the object at point ``a`` and stretch the entire plane to ``b``.

    Args:
        a, b (point vectors): the control points
        k (pos float or None): not the usual k, here it limits the
            surrounding positions around a and b that can be reached
            during stretching. Use this if stretching results
            in weird parts missing.
    """
    ab = (ab := b - a) / (L := np.linalg.norm(ab))
    if k is not None:
        k_ = k / L

    def f(p):
        # s = ”how far are we between a and b as fraction?”
        s = np.clip((p - a) @ ab / L, 0, 1)
        s_ = e(s)
        if k is not None:
            s_ = np.clip(s, s - k_, s + k_)
        return sdf(p - (s_ * L * ab[:, np.newaxis]).T)

    return f


def mirror(other, direction, at=0):
    """
    Mirror around a given plane defined by ``origin`` reference point and
    ``direction``.

    Args:
        direction (vector): direction to mirror to (e.g. :any:`X` to mirror along X axis)
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
