"""Helper classes and functions for 3D FOV Visualization."""
# TODO: test Z-primary functions
# TODO: test different eye levels for DZDP functions (for X and Y axis)
# TODO: Get rid of slope functions and tests (using better way now)

import math
from enum import Enum
from typing import Optional, Self, Tuple

#    ######   ##           ##      ######    ######   ########   ######
#   ##    ##  ##         ##  ##   ##        ##        ##        ##
#   ##        ##        ##    ##   ######    ######   ######     ######
#   ##    ##  ##        ########        ##        ##  ##              ##
#    ######   ########  ##    ##  #######   #######   ########  #######


class Axis(Enum):
    """Primary axes for 3D FOV."""

    X = 1
    Y = 2
    Z = 4

    def __iter__(self):
        return iter((self.X, self.Y, self.Z))


class Coords:
    """3D map integer coordinates."""

    __slots__ = "x", "y", "z"

    def __init__(self, x: int, y: int, z: int) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __repr__(self) -> str:
        return f"{self.x, self.y, self.z}"

    def as_tuple(self):
        return (self.x, self.y, self.z)


class Octant(Enum):
    """Octants for use in 3D FOV. Octant 1 is (+++)."""

    O1 = 1
    O2 = 2
    O3 = 3
    O4 = 4
    O5 = 5
    O6 = 6
    O7 = 7
    O8 = 8

    def __iter__(self):
        return iter(
            (self.O1, self.O2, self.O3, self.O4, self.O5, self.O6, self.O7, self.O8)
        )


class FovSide(Enum):
    """Side facing for an FovRect (X, Y, or Z)."""

    Xfacing = 1  # 'X-facing', Y aligned. Normal's X-value points toward 0
    Yfacing = 2  # 'Y-facing', X aligned. Normal's Y-value points toward 0
    Zfacing = 3  # 'Z-facing', parallel to XY plane. Normal's Z-value points toward 0

    def __iter__(self):
        return iter((self.Xfacing, self.Yfacing, self.Zfacing))


class Line:
    """3D line segment."""

    __slots__ = "x1", "y1", "z1", "x2", "y2", "z2"

    def __init__(
        self, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float
    ) -> None:
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.x2 = x2
        self.y2 = y2
        self.z2 = z2

    def __iter__(self):
        return iter((self.x1, self.y1, self.z1, self.x2, self.y2, self.z2))

    def __repr__(self) -> str:
        return f"Line {self.x1, self.y1, self.z1, self.x2, self.y2, self.z2}"

    def as_tuple(self):
        return (self.x1, self.y1, self.z1, self.x2, self.y2, self.z2)

    def as_ray(self):
        """Returns line in (Point, Vector) Ray form."""
        rvx = self.x2 - self.x1
        rvy = self.y2 - self.y1
        rvz = self.z2 - self.z1

        return Ray(self.x1, self.y1, self.z1, rvx, rvy, rvz)


class Point:
    """3D floating point coordinates."""

    __slots__ = "x", "y", "z"

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __repr__(self) -> str:
        return f"P{self.x, self.y, self.z}"

    def as_tuple(self):
        return (self.x, self.y, self.z)

    def add_vector(self, v) -> Self:
        return Point(self.x + v.x, self.y + v.y, self.z + v.z)

    def distance(self, other: Self) -> float:
        """Returns distance between self and other."""
        dx_abs = (other.x - self.x) ** 2
        dy_abs = (other.y - self.y) ** 2
        dz_abs = (other.z - self.z) ** 2

        return math.sqrt(dx_abs + dy_abs + dz_abs)

    def rounded(self) -> Self:
        return Point(round(self.x, 3), round(self.y, 3), round(self.z, 3))


class Vector:
    """3D floating point vector. Used to differentiate functionality from `Point`."""

    __slots__ = "x", "y", "z"

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: Self):
        return Vector(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __repr__(self) -> str:
        return f"V{self.x, self.y, self.z}"

    def __sub__(self, other: Self):
        return Vector(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def as_tuple(self):
        return (self.x, self.y, self.z)

    def cross(self, other: Self) -> Self:
        """Returns cross product of self and other vector."""
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def dot(self, other: Self) -> float:
        """Returns dot product of self and other vector."""
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def multiplied(self, factor: float):
        """Returns new vector as self multiplied by factor."""
        return Vector(self.x * factor, self.y * factor, self.z * factor)

    def multiply(self, factor: float):
        """Multiplies self by factor."""
        self.x *= factor
        self.y *= factor
        self.z *= factor

    def unit(self) -> Self:
        """Returns unit vector of self."""
        magnitude = self.dot(self)
        return Vector(self.x / magnitude, self.y / magnitude, self.z / magnitude)


class FovRect:
    """3D axis-aligned rectangle specifically made for FOV calculations.

    Reference point is closest to origin (0,0) - width and height are added to it.
    Side vector s1 is from p0 to p1 (width); side vector s2 is from p0 to p2 (height).

    Width and height are in cell distance (0.0 to 1.0).

    `p0`: Point
        Reference point. Always closest to origin.
    `s1`, `s2`: Vector
        Side vectors defining width and height. Needed for intersections.
    `s1_abs_mag`, `s2_abs_mag`: float
        Absolute magnitude (no square root) of side vectors s1 and s2. Effectively
        width squared or height squared.
    `normal`: Vector
        Defines normal vector to the rectangle plane. Always points toward origin. For
        side 'A', normal points toward x=0. For side 'B', it points toward y=0. For
        side 'C', it points toward z=0. Will be normalized (in unit form), but does
        not need to be.
    """

    __slots__ = "p0", "s1", "s2", "s1_abs_mag", "s2_abs_mag", "normal"

    def __init__(
        self,
        p0: Point,
        s1: Vector,
        s2: Vector,
        s1_abs_mag: float,
        s2_abs_mag: float,
        normal: Vector,
    ) -> None:
        self.p0 = p0
        self.s1 = s1
        self.s2 = s2
        self.normal = normal
        self.s1_abs_mag = s1_abs_mag
        self.s2_abs_mag = s2_abs_mag

    def __repr__(self) -> str:
        p0x, p0y, p0z = self.p0
        side1 = Line(
            p0x,
            p0y,
            p0z,
            self.p0.x + self.s1.x,
            self.p0.y + self.s1.y,
            self.p0.z + self.s1.z,
        )
        side2 = Line(
            p0x,
            p0y,
            p0z,
            self.p0.x + self.s2.x,
            self.p0.y + self.s2.y,
            self.p0.z + self.s2.z,
        )
        return f"Rect: s1 {side1}, s2 {side2}"

    def shifted(self, dx: float, dy: float, dz: float) -> Self:
        """Returns a new `FovRect` whose reference point p0 is shifted from original."""
        x, y, z = self.p0
        p0 = Point(x + dx, y + dy, z + dz)
        s1 = self.s1
        s2 = self.s2
        s1_abs_mag = self.s1_abs_mag
        s2_abs_mag = self.s2_abs_mag
        normal = self.normal
        return FovRect(p0, s1, s2, s1_abs_mag, s2_abs_mag, normal)

    @staticmethod
    def new(p0: Point, width: float, height: float, octant: Octant, side: FovSide):
        """Makes a new `FovRect` instance based on octant and X/Y/Z facing.

        p0 is always closest to the origin.

        For Z-facing side, width is along X-axis and height is along Y-axis.
        """
        s1, s2, normal = (
            Vector(0.0, 0.0, 0.0),
            Vector(0.0, 0.0, 0.0),
            Vector(0.0, 0.0, 0.0),
        )

        s1_abs_mag = width * width
        s2_abs_mag = height * height

        match octant, side:
            # (+x, +y, +z)
            case Octant.O1, FovSide.Xfacing:
                s1 = Vector(0.0, width, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(-1.0, 0.0, 0.0)
            case Octant.O1, FovSide.Yfacing:
                s1 = Vector(width, 0.0, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(0.0, -1.0, 0.0)
            case Octant.O1, FovSide.Zfacing:
                s1 = Vector(width, 0.0, 0.0)
                s2 = Vector(0.0, height, 0.0)
                normal = Vector(0.0, 0.0, -1.0)
            # (-x, +y, +z)
            case Octant.O2, FovSide.Xfacing:
                s1 = Vector(0.0, width, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(1.0, 0.0, 0.0)
            case Octant.O2, FovSide.Yfacing:
                s1 = Vector(-width, 0.0, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(0.0, -1.0, 0.0)
            case Octant.O2, FovSide.Zfacing:
                s1 = Vector(-width, 0.0, 0.0)
                s2 = Vector(0.0, height, 0.0)
                normal = Vector(0.0, 0.0, -1.0)
            # (-x, -y, +z)
            case Octant.O3, FovSide.Xfacing:
                s1 = Vector(0.0, -width, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(1.0, 0.0, 0.0)
            case Octant.O3, FovSide.Yfacing:
                s1 = Vector(-width, 0.0, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(0.0, 1.0, 0.0)
            case Octant.O3, FovSide.Zfacing:
                s1 = Vector(-width, 0.0, 0.0)
                s2 = Vector(0.0, -height, 0.0)
                normal = Vector(0.0, 0.0, -1.0)
            # (+x, -y, +z)
            case Octant.O4, FovSide.Xfacing:
                s1 = Vector(0.0, -width, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(-1.0, 0.0, 0.0)
            case Octant.O4, FovSide.Yfacing:
                s1 = Vector(width, 0.0, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(0.0, 1.0, 0.0)
            case Octant.O4, FovSide.Zfacing:
                s1 = Vector(width, 0.0, 0.0)
                s2 = Vector(0.0, -height, 0.0)
                normal = Vector(0.0, 0.0, -1.0)
            # (+x, +y, -z)
            case Octant.O5, FovSide.Xfacing:
                s1 = Vector(0.0, width, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(-1.0, 0.0, 0.0)
            case Octant.O5, FovSide.Yfacing:
                s1 = Vector(width, 0.0, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(0.0, -1.0, 0.0)
            case Octant.O5, FovSide.Zfacing:
                s1 = Vector(width, 0.0, 0.0)
                s2 = Vector(0.0, height, 0.0)
                normal = Vector(0.0, 0.0, 1.0)
            # (-x, +y, -z)
            case Octant.O6, FovSide.Xfacing:
                s1 = Vector(0.0, width, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(1.0, 0.0, 0.0)
            case Octant.O6, FovSide.Yfacing:
                s1 = Vector(-width, 0.0, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(0.0, -1.0, 0.0)
            case Octant.O6, FovSide.Zfacing:
                s1 = Vector(-width, 0.0, 0.0)
                s2 = Vector(0.0, height, 0.0)
                normal = Vector(0.0, 0.0, 1.0)
            # (+x, -y, -z)
            case Octant.O7, FovSide.Xfacing:
                s1 = Vector(0.0, -width, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(1.0, 0.0, 0.0)
            case Octant.O7, FovSide.Yfacing:
                s1 = Vector(-width, 0.0, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(0.0, 1.0, 0.0)
            case Octant.O7, FovSide.Zfacing:
                s1 = Vector(-width, 0.0, 0.0)
                s2 = Vector(0.0, -height, 0.0)
                normal = Vector(0.0, 0.0, 1.0)
            # (+x, -y, -z)
            case Octant.O8, FovSide.Xfacing:
                s1 = Vector(0.0, -width, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(-1.0, 0.0, 0.0)
            case Octant.O8, FovSide.Yfacing:
                s1 = Vector(width, 0.0, 0.0)
                s2 = Vector(0.0, 0.0, height)
                normal = Vector(0.0, 1.0, 0.0)
            case Octant.O8, FovSide.Zfacing:
                s1 = Vector(width, 0.0, 0.0)
                s2 = Vector(0.0, -height, 0.0)
                normal = Vector(0.0, 0.0, 1.0)

        return FovRect(p0, s1, s2, s1_abs_mag, s2_abs_mag, normal)


class QBits(Enum):
    """Number of Q bits used for quantized slopes and angle ranges."""

    Q32 = 32  # Least granular
    Q64 = 64
    Q128 = 128
    Q256 = 256  # Most granular


class Ray:
    """3D ray defined by ray origin r0 and ray vector rv."""

    __slots__ = "r0", "rv"

    def __init__(self, r0x, r0y, r0z, rvx, rvy, rvz) -> None:
        self.r0 = Point(r0x, r0y, r0z)
        self.rv = Vector(rvx, rvy, rvz)

    def __repr__(self) -> str:
        return (
            f"Ray {self.r0.x, self.r0.y, self.r0.z}->{self.rv.x, self.rv.y, self.rv.z}"
        )


#   ########  ##    ##  ##    ##   ######   ########  ########   ######   ##    ##
#   ##        ##    ##  ####  ##  ##    ##     ##        ##     ##    ##  ####  ##
#   ######    ##    ##  ## ## ##  ##           ##        ##     ##    ##  ## ## ##
#   ##        ##    ##  ##  ####  ##    ##     ##        ##     ##    ##  ##  ####
#   ##         ######   ##    ##   ######      ##     ########   ######   ##    ##


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamps floating point value between min and max."""
    if min_value > max_value:
        raise ValueError("minimum value cannot be larger than maximum!")

    return max(min(value, max_value), min_value)


def octant_to_relative(
    dpri: int, dsec: int, dter: int, octant: Octant, axis: Axis
) -> Coords:
    """Converts coordinates from FovCell's Octant form to relative CellMap form.

    X primary: X is pri, Y is sec, Z is ter
    Y primary: Y is pri, X is sec, Z is ter
    Z primary: Z is pri, X is sec, Y is ter
    """
    match octant:
        case Octant.O1:
            match axis:
                case Axis.X:
                    rx, ry, rz = dpri, dsec, dter
                case Axis.Y:
                    rx, ry, rz = dsec, dpri, dter
                case Axis.Z:
                    rx, ry, rz = dsec, dter, dpri
        case Octant.O2:
            match axis:
                case Axis.X:
                    rx, ry, rz = -dpri, dsec, dter
                case Axis.Y:
                    rx, ry, rz = -dsec, dpri, dter
                case Axis.Z:
                    rx, ry, rz = -dsec, dter, dpri
        case Octant.O3:
            match axis:
                case Axis.X:
                    rx, ry, rz = -dpri, -dsec, dter
                case Axis.Y:
                    rx, ry, rz = -dsec, -dpri, dter
                case Axis.Z:
                    rx, ry, rz = -dsec, -dter, dpri
        case Octant.O4:
            match axis:
                case Axis.X:
                    rx, ry, rz = dpri, -dsec, dter
                case Axis.Y:
                    rx, ry, rz = dsec, -dpri, dter
                case Axis.Z:
                    rx, ry, rz = dsec, -dter, dpri
        case Octant.O5:
            match axis:
                case Axis.X:
                    rx, ry, rz = dpri, dsec, -dter
                case Axis.Y:
                    rx, ry, rz = dsec, dpri, -dter
                case Axis.Z:
                    rx, ry, rz = dsec, dter, -dpri
        case Octant.O6:
            match axis:
                case Axis.X:
                    rx, ry, rz = -dpri, dsec, -dter
                case Axis.Y:
                    rx, ry, rz = -dsec, dpri, -dter
                case Axis.Z:
                    rx, ry, rz = -dsec, dter, -dpri
        case Octant.O7:
            match axis:
                case Axis.X:
                    rx, ry, rz = -dpri, -dsec, -dter
                case Axis.Y:
                    rx, ry, rz = -dsec, -dpri, -dter
                case Axis.Z:
                    rx, ry, rz = -dsec, -dter, -dpri
        case Octant.O8:
            match axis:
                case Axis.X:
                    rx, ry, rz = dpri, -dsec, -dter
                case Axis.Y:
                    rx, ry, rz = dsec, -dpri, -dter
                case Axis.Z:
                    rx, ry, rz = dsec, -dter, -dpri

    return Coords(rx, ry, rz)


def to_cell_id(x: int, y: int, z: int, xdims: int, ydims: int):
    """
    Takes 3D cell (x,y,z) coordinates and converts them into a cell ID.

    Where:
        - cell_id = x * x_shift + y * y_shift + z * z_shift
        - x_shift = 1
        - y_shift = x_dims
        - z_shift = y_dims * x_dims

    Parameters
    ----------
    x, y, z : int
        (x,y,z) coordinates of the cell.
    `xdims` : int
        number of x dimensions.
    `ydims` : int
        number of y dimensions.
    """
    y_shift = xdims
    z_shift = ydims * xdims

    return x + y * y_shift + z * z_shift


#    ######   ##         ######   #######   ########
#   ##        ##        ##    ##  ##    ##  ##
#    ######   ##        ##    ##  #######   ######
#         ##  ##        ##    ##  ##        ##
#   #######   ########   ######   ##        ########


def dsdp_slopes_cell(dpri: int, dsec: int) -> Tuple[float, float]:
    """Gets `dsec/dpri` slope range to entire cell range for all Octants.

    This covers both structure and floor ranges.
    """
    if dpri == 0:
        return (0.0, 1.0)
    if dsec == 0:
        s1 = clamp((dsec - 0.5) / (dpri - 0.5), 0.0, 1.0)
        s2 = clamp((dsec + 0.5) / (dpri - 0.5), 0.0, 1.0)
    else:
        s1 = clamp((dsec - 0.5) / (dpri + 0.5), 0.0, 1.0)
        s2 = clamp((dsec + 0.5) / (dpri - 0.5), 0.0, 1.0)

    return min(s1, s2), max(s1, s2)


def dsdp_slopes_north(
    dpri: int, dsec: int, octant: Octant, axis: Axis
) -> Tuple[float, float]:
    """Gets `dsec/dpri` slope range to cell's North side based on Octant."""
    match axis:
        case Axis.X:
            match octant:
                case Octant.O1 | Octant.O2 | Octant.O5 | Octant.O6:
                    if dpri == 0:
                        return (0.0, 0.0)
                    s1 = clamp((dsec - 0.5) / (dpri + 0.5), 0.0, 1.0)
                    s2 = clamp((dsec - 0.5) / (dpri - 0.5), 0.0, 1.0)
                case Octant.O3 | Octant.O4 | Octant.O7 | Octant.O8:
                    if dpri == 0:
                        return (1.0, 1.0)
                    s1 = clamp((dsec + 0.5) / (dpri + 0.5), 0.0, 1.0)
                    s2 = clamp((dsec + 0.5) / (dpri - 0.5), 0.0, 1.0)
        case Axis.Y:
            match octant:
                case Octant.O1 | Octant.O2 | Octant.O5 | Octant.O6:
                    if dpri == 0:
                        return (0.0, 0.0)
                    s1 = clamp((dsec - 0.5) / (dpri - 0.5), 0.0, 1.0)
                    s2 = clamp((dsec + 0.5) / (dpri - 0.5), 0.0, 1.0)
                case Octant.O3 | Octant.O4 | Octant.O7 | Octant.O8:
                    s1 = clamp((dsec - 0.5) / (dpri + 0.5), 0.0, 1.0)
                    s2 = clamp((dsec + 0.5) / (dpri + 0.5), 0.0, 1.0)
        case Axis.Z:
            raise ValueError("Axis Z calculations not yet complete!")

    return min(s1, s2), max(s1, s2)


def dsdp_slopes_west(
    dpri: int, dsec: int, octant: Octant, axis: Axis
) -> Tuple[float, float]:
    """Gets `dsec/dpri` slope range to cell's West side based on Octant."""

    match axis:
        case Axis.X:
            match octant:
                case Octant.O1 | Octant.O4 | Octant.O5 | Octant.O8:
                    if dpri == 0:
                        return (0.0, 0.0)
                    s1 = clamp((dsec - 0.5) / (dpri - 0.5), 0.0, 1.0)
                    s2 = clamp((dsec + 0.5) / (dpri - 0.5), 0.0, 1.0)
                case Octant.O2 | Octant.O3 | Octant.O6 | Octant.O7:
                    s1 = clamp((dsec - 0.5) / (dpri + 0.5), 0.0, 1.0)
                    s2 = clamp((dsec + 0.5) / (dpri + 0.5), 0.0, 1.0)
        case Axis.Y:
            match octant:
                case Octant.O1 | Octant.O4 | Octant.O5 | Octant.O8:
                    if dpri == 0:
                        return (0.0, 0.0)
                    s1 = clamp((dsec - 0.5) / (dpri + 0.5), 0.0, 1.0)
                    s2 = clamp((dsec - 0.5) / (dpri - 0.5), 0.0, 1.0)
                case Octant.O2 | Octant.O3 | Octant.O6 | Octant.O7:
                    s1 = clamp((dsec + 0.5) / (dpri + 0.5), 0.0, 1.0)
                    s2 = clamp((dsec + 0.5) / (dpri - 0.5), 0.0, 1.0)
        case Axis.Z:
            raise ValueError("Axis Z calculations not yet complete!")

    return min(s1, s2), max(s1, s2)


def dzdp_slopes_cell(
    dpri: int, dz: int, wall_ht: float, eye_ht: float, octant: Octant, axis: Axis
) -> Tuple[float, float]:
    """Gets `dz/dpri` slope range based on octant, for an entire cell (structure).

    `eye_ht` is FOV level as % of cell height. If max cell ht is 8 and eye height
    is 6 (standing), eye_ht height is 0.75. `wall_ht` is similar (1.0 = full ht),
    representing the height of the structure (rather than a wall).

    If dz >= 0, the "high" slope reaches the upper part of the wall.  If dz < 0,
    the "high" slope reaches the bottom of the cell (the floor).
    """
    if axis == Axis.Z:
        raise ValueError("dzdp slopes do not apply to the Z axis!")

    match octant:
        case Octant.O1 | Octant.O2 | Octant.O3 | Octant.O4:
            s1 = clamp((dz - eye_ht) / (dpri - 0.5), 0.0, 1.0)
            s2 = clamp((dz + wall_ht - eye_ht) / (dpri - 0.5), 0.0, 1.0)
        case Octant.O5 | Octant.O6 | Octant.O7 | Octant.O8:
            if dpri == 0:
                return (0.0, 1.0)
            s1 = clamp((dz + eye_ht - wall_ht) / (dpri + 0.5), 0.0, 1.0)
            s2 = clamp((dz + eye_ht) / (dpri - 0.5), 0.0, 1.0)

    return min(s1, s2), max(s1, s2)


def dzdp_slopes_floor(
    dpri: int, dz: int, eye_ht: float, octant: Octant, axis: Axis
) -> Tuple[float, float]:
    """Gets `dz/dpri` slope range based on octant, for all walls.

    `eye_ht` is FOV level as % of cell height. If max cell ht is 8 and eye height
    is 6 (standing), eye_ht height is 0.75.
    """
    if axis == Axis.Z:
        raise ValueError("dzdp slopes do not apply to the Z axis!")

    match octant:
        case Octant.O1 | Octant.O2 | Octant.O3 | Octant.O4:
            # Own floor doesn't block when looking up
            if dpri == 0:
                return (0.0, 0.0)
            s1 = clamp((dz - eye_ht) / (dpri + 0.5), 0.0, 1.0)
            s2 = clamp((dz - eye_ht) / (dpri - 0.5), 0.0, 1.0)
        case Octant.O5 | Octant.O6 | Octant.O7 | Octant.O8:
            # Own floor blocks only diagonal corner when looking down
            if dpri == 0:
                return (1.0, 1.0)
            s1 = clamp((dz + eye_ht) / (dpri + 0.5), 0.0, 1.0)
            s2 = clamp((dz + eye_ht) / (dpri - 0.5), 0.0, 1.0)

    return min(s1, s2), max(s1, s2)


def dzdp_slopes_north(
    dpri: int, dz: int, wall_ht: float, eye_ht: float, octant: Octant, axis: Axis
) -> Tuple[float, float]:
    """Gets `dz/dpri` slope range based on octant, to the North wall.

    `eye_ht` is FOV level as % of cell height. If max cell ht is 8 and eye height
    is 6 (standing), eye_ht height is 0.75. `wall_ht` is similar (1.0 = full ht).

    If dz >= 0, the "high" slope reaches the upper part of the wall.  If dz < 0,
    the "high" slope reaches the bottom of the cell (the floor).

    For dzdp, choose the primary distance that gives the most accurate Z-slope range.
    To visualize, try using a side view.
    """
    # NOTE: X: 1,2,3,4 should all use
    o = octant

    match octant, axis:
        case (_, Axis.Z):
            raise ValueError("dzdp slopes do not apply to the Z Axis!")
        case (o.O3, Axis.Y) | (o.O4, Axis.Y):
            # Upper Octants far from North wall
            s1 = clamp((dz - eye_ht) / (dpri + 0.5), 0.0, 1.0)
            s2 = clamp((dz + wall_ht - eye_ht) / (dpri + 0.5), 0.0, 1.0)
        case (o.O7, Axis.Y) | (o.O8, Axis.Y):
            # Lower Octants far from North wall
            s1 = clamp((dz + eye_ht - wall_ht) / (dpri + 0.5), 0.0, 1.0)
            s2 = clamp((dz + eye_ht) / (dpri + 0.5), 0.0, 1.0)
        case (o.O1, _) | (o.O2, _) | (o.O3, Axis.X) | (o.O4, Axis.X):
            # Upper Octants close to North wall
            s1 = clamp((dz - eye_ht) / (dpri - 0.5), 0.0, 1.0)
            s2 = clamp((dz + wall_ht - eye_ht) / (dpri - 0.5), 0.0, 1.0)
        case (o.O5, _) | (o.O6, _) | (o.O7, Axis.X) | (o.O8, Axis.X):
            # Lower Octants close to North wall
            s1 = clamp((dz + eye_ht - wall_ht) / (dpri - 0.5), 0.0, 1.0)
            s2 = clamp((dz + eye_ht) / (dpri - 0.5), 0.0, 1.0)
        case _:
            raise ValueError("Logic error! Function doesn't cover all cases!")

    return min(s1, s2), max(s1, s2)


def dzdp_slopes_west(
    dpri: int, dz: int, wall_ht: float, eye_ht: float, octant: Octant, axis: Axis
) -> Tuple[float, float]:
    """Gets `dz/dpri` slope range based on octant, to the West wall.

    `eye_ht` is FOV level as % of cell height. If max cell ht is 8 and eye height
    is 6 (standing), eye_ht height is 0.75. `wall_ht` is similar (1.0 = full ht).

    If dz >= 0, the "high" slope reaches the upper part of the wall.  If dz < 0,
    the "high" slope reaches the bottom of the cell (the floor).
    """
    o = octant

    match octant, axis:
        case (_, Axis.Z):
            raise ValueError("dzdp doesn't apply for Z Axis!")
        case (o.O2, Axis.X) | (o.O3, Axis.X):
            # Upper Octants far from West wall
            s1 = clamp((dz - eye_ht) / (dpri + 0.5), 0.0, 1.0)
            s2 = clamp((dz + wall_ht - eye_ht) / (dpri + 0.5), 0.0, 1.0)
        case (o.O6, Axis.X) | (o.O7, Axis.X):
            # Lower Octants far from West wall
            s1 = clamp((dz + eye_ht - wall_ht) / (dpri + 0.5), 0.0, 1.0)
            s2 = clamp((dz + eye_ht) / (dpri + 0.5), 0.0, 1.0)
        case (o.O1, _) | (o.O4, _) | (o.O2, Axis.Y) | (o.O3, Axis.Y):
            # Upper Octants close to West wall
            s1 = clamp((dz - eye_ht) / (dpri - 0.5), 0.0, 1.0)
            s2 = clamp((dz + wall_ht - eye_ht) / (dpri - 0.5), 0.0, 1.0)
        case (o.O5, _) | (o.O8, _) | (o.O6, Axis.Y) | (o.O7, Axis.Y):
            # Lower Octants close to West wall
            s1 = clamp((dz + eye_ht - wall_ht) / (dpri - 0.5), 0.0, 1.0)
            s2 = clamp((dz + eye_ht) / (dpri - 0.5), 0.0, 1.0)
        case _:
            raise ValueError("Logic error! Function doesn't cover all cases!")

    return min(s1, s2), max(s1, s2)


def octant_slope_summary(dpri: int, dsec: int, dz: int, wall_ht: float, eye_ht: float):
    """Prints a summary of (low,high) slopes for given octant coordinates."""
    for octant in [
        Octant.O1,
        Octant.O2,
        Octant.O3,
        Octant.O4,
        Octant.O5,
        Octant.O6,
        Octant.O7,
        Octant.O8,
    ]:
        for axis in [Axis.X, Axis.Y, Axis.Z]:
            cell = dsdp_slopes_cell(dpri, dsec)
            north = dsdp_slopes_north(dpri, dsec, octant, axis)
            west = dsdp_slopes_west(dpri, dsec, octant, axis)
            zwest = dzdp_slopes_west(dpri, dz, wall_ht, eye_ht, octant, axis)
            znorth = dzdp_slopes_north(dpri, dz, wall_ht, eye_ht, octant, axis)
            print(
                f"{octant}[{axis}] {dpri, dsec}\n  cell: {cell}\n  north: {north}\n  west: {west}\n  zwest: {zwest}\n  znorth: {znorth}"
            )


def ray_rect_intersection(
    r0: Point, rv: Vector, rect: FovRect, epsilon=1e-6
) -> Optional[Point]:
    """Checks if a ray intersects a rectangle, returning the point if it does.

    The 'dfac' variable is the distance factor - helps find the point between r0 -> r1
    where the ray intersects the given plane (ranges from 0.0 - 1.0).

    If 'dfac' is between (0.0 - 1.0) the point intersects with the segment. Otherwise:
        dfac < 0.0: intersection point falls short of r0.
        dfac > 1.0: intersection point falls beyond r1.

    Absolute magnitude data for each side is precalculated to avoid having to
    recalculate it each time the function is called.

    For the intersection point to be within the rectangle, the following must hold
    true for the test vector (tv):
        1.) dot product of test vector onto s1 cannot be:
                < 0, OR
                > absolute magnitude of s1
        2.) dot product of test vector onto s2 cannot be:
                < 0, OR
                > absolute magnitude of s2

    ### Parameters

    r0, : Point, Vector
        Origin of the ray.
    rv : Vector
        Vector emanating from the ray origin r0. The line is defined by r0 + rv.
    rect : FovRect
        Rectangle against which ray intersection is checked.  Comprised of side
        vectors s1 and s2 and a normal vector, all emerging from origin point p0.
    epsilon : float
        floating point error tolerance value.
    """
    p0x, p0y, p0z = rect.p0
    dot = rect.normal.dot(rv)

    if abs(dot) > epsilon:
        w = Vector(r0.x - p0x, r0.y - p0y, r0.z - p0z)

        dfac = -(rect.normal.dot(w)) / dot
        if dfac < 0.0 or dfac > 1.0 + epsilon:
            return None

        # Ray/plane intersection point
        rv_exp = rv.multiplied(dfac)
        result = r0.add_vector(rv_exp)

        # Test vector to see if intersection falls within rectangle
        tv = Vector(result.x - p0x, result.y - p0y, result.z - p0z)
        projection_1 = tv.dot(rect.s1)
        projection_2 = tv.dot(rect.s2)

        if projection_1 < 0 or projection_1 > rect.s1_abs_mag:
            return None
        if projection_2 < 0 or projection_2 > rect.s2_abs_mag:
            return None

        # Intersecton is within rectangle
        return result
    else:
        # The ray is parallel to the plane
        return None


def xyz_boundary_distance(
    src: Coords, dims: Coords, octant: Octant, radius: int
) -> Coords:
    """Get the maximum (x,y,z) FOV radius to the (x,y,z) boundaries.

    Octants 1-4 are in upper half, 5-8 are in lower half.
    """
    ox, oy, oz = src
    xdims, ydims, zdims = dims

    match octant:
        # +++
        case Octant.O1:
            xdist = min(xdims - ox - 1, radius)
            ydist = min(ydims - oy - 1, radius)
            zdist = min(zdims - oz - 1, radius)
            print(f"x/y/z bounardy distance: {xdist, ydist, zdist}")
        # -++
        case Octant.O2:
            xdist = min(ox, radius)
            ydist = min(ydims - oy - 1, radius)
            zdist = min(zdims - oz - 1, radius)
        # --+
        case Octant.O3:
            xdist = min(ox, radius)
            ydist = min(oy, radius)
            zdist = min(zdims - oz - 1, radius)
        # +-+
        case Octant.O4:
            xdist = min(xdims - ox - 1, radius)
            ydist = min(oy, radius)
            zdist = min(zdims - oz - 1, radius)
        # ++-
        case Octant.O5:
            xdist = min(xdims - ox - 1, radius)
            ydist = min(ydims - oy - 1, radius)
            zdist = min(oz, radius)
        # -+-
        case Octant.O6:
            xdist = min(ox, radius)
            ydist = min(ydims - oy - 1, radius)
            zdist = min(oz, radius)
        # ---
        case Octant.O7:
            xdist = min(ox, radius)
            ydist = min(oy, radius)
            zdist = min(oz, radius)
        # +--
        case Octant.O8:
            xdist = min(xdims - ox - 1, radius)
            ydist = min(oy, radius)
            zdist = min(oz, radius)

    return Coords(xdist, ydist, zdist)


#   ########  ########   ######   ########
#      ##     ##        ##           ##
#      ##     ######     ######      ##
#      ##     ##              ##     ##
#      ##     ########  #######      ##


def test_dsdp_slopes_cell():
    """Test dsdp slopes to the entire Cell with all octants and axes."""
    cells = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
    actual = [dsdp_slopes_cell(dp, ds) for (dp, ds, _) in cells]
    expected = [(0.0, 1.0), (0.0, 1.0), (1 / 3, 1.0), (0.0, 1.0), (1 / 3, 1.0)]

    assert actual == expected


def test_dsdp_slopes_north():
    """Test dsdp slopes to the North wall with all octants and axes.

    Expected: octants (1,2,5,6) and (3,4,7,8) should have same slope ranges.
    """
    cells = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]

    o1aX = [dsdp_slopes_north(dp, ds, Octant.O1, Axis.X) for (dp, ds, _) in cells]
    o1aY = [dsdp_slopes_north(dp, ds, Octant.O1, Axis.Y) for (dp, ds, _) in cells]
    o2aX = [dsdp_slopes_north(dp, ds, Octant.O2, Axis.X) for (dp, ds, _) in cells]
    o2aY = [dsdp_slopes_north(dp, ds, Octant.O2, Axis.Y) for (dp, ds, _) in cells]
    o3aX = [dsdp_slopes_north(dp, ds, Octant.O3, Axis.X) for (dp, ds, _) in cells]
    o3aY = [dsdp_slopes_north(dp, ds, Octant.O3, Axis.Y) for (dp, ds, _) in cells]
    o4aX = [dsdp_slopes_north(dp, ds, Octant.O4, Axis.X) for (dp, ds, _) in cells]
    o4aY = [dsdp_slopes_north(dp, ds, Octant.O4, Axis.Y) for (dp, ds, _) in cells]

    o5aX = [dsdp_slopes_north(dp, ds, Octant.O5, Axis.X) for (dp, ds, _) in cells]
    o5aY = [dsdp_slopes_north(dp, ds, Octant.O5, Axis.Y) for (dp, ds, _) in cells]
    o6aX = [dsdp_slopes_north(dp, ds, Octant.O6, Axis.X) for (dp, ds, _) in cells]
    o6aY = [dsdp_slopes_north(dp, ds, Octant.O6, Axis.Y) for (dp, ds, _) in cells]
    o7aX = [dsdp_slopes_north(dp, ds, Octant.O7, Axis.X) for (dp, ds, _) in cells]
    o7aY = [dsdp_slopes_north(dp, ds, Octant.O7, Axis.Y) for (dp, ds, _) in cells]
    o8aX = [dsdp_slopes_north(dp, ds, Octant.O8, Axis.X) for (dp, ds, _) in cells]
    o8aY = [dsdp_slopes_north(dp, ds, Octant.O8, Axis.Y) for (dp, ds, _) in cells]

    expected_o1aX = [(0.0, 0.0), (0.0, 0.0), (1 / 3, 1.0), (0.0, 0.0), (1 / 3, 1.0)]
    expected_o1aY = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 1.0), (1.0, 1.0)]
    expected_o2aX = [(0.0, 0.0), (0.0, 0.0), (1 / 3, 1.0), (0.0, 0.0), (1 / 3, 1.0)]
    expected_o2aY = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 1.0), (1.0, 1.0)]
    expected_o3aX = [(1.0, 1.0), (1 / 3, 1.0), (1.0, 1.0), (1 / 3, 1.0), (1.0, 1.0)]
    expected_o3aY = [(0.0, 1.0), (0.0, 1 / 3), (1 / 3, 1.0), (0.0, 1 / 3), (1 / 3, 1.0)]
    expected_o4aX = [(1.0, 1.0), (1 / 3, 1.0), (1.0, 1.0), (1 / 3, 1.0), (1.0, 1.0)]
    expected_o4aY = [(0.0, 1.0), (0.0, 1 / 3), (1 / 3, 1.0), (0.0, 1 / 3), (1 / 3, 1.0)]

    expected_o5aX = [(0.0, 0.0), (0.0, 0.0), (1 / 3, 1.0), (0.0, 0.0), (1 / 3, 1.0)]
    expected_o5aY = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 1.0), (1.0, 1.0)]
    expected_o6aX = [(0.0, 0.0), (0.0, 0.0), (1 / 3, 1.0), (0.0, 0.0), (1 / 3, 1.0)]
    expected_o6aY = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 1.0), (1.0, 1.0)]
    expected_o7aX = [(1.0, 1.0), (1 / 3, 1.0), (1.0, 1.0), (1 / 3, 1.0), (1.0, 1.0)]
    expected_o7aY = [(0.0, 1.0), (0.0, 1 / 3), (1 / 3, 1.0), (0.0, 1 / 3), (1 / 3, 1.0)]
    expected_o8aX = [(1.0, 1.0), (1 / 3, 1.0), (1.0, 1.0), (1 / 3, 1.0), (1.0, 1.0)]
    expected_o8aY = [(0.0, 1.0), (0.0, 1 / 3), (1 / 3, 1.0), (0.0, 1 / 3), (1 / 3, 1.0)]

    assert o1aX == expected_o1aX
    assert o1aY == expected_o1aY
    assert o2aX == expected_o2aX
    assert o2aY == expected_o2aY
    assert o3aX == expected_o3aX
    assert o3aY == expected_o3aY
    assert o4aX == expected_o4aX
    assert o4aY == expected_o4aY

    assert o5aX == expected_o5aX
    assert o5aY == expected_o5aY
    assert o6aX == expected_o6aX
    assert o6aY == expected_o6aY
    assert o7aX == expected_o7aX
    assert o7aY == expected_o7aY
    assert o8aX == expected_o8aX
    assert o8aY == expected_o8aY


def test_dsdp_slopes_west():
    """Test dsdp slopes to the West wall with all octants and axes.

    Expected: octants (1,4,5,8) and (2,3,6,7) should have same slope ranges.
    """
    cells = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]

    o1aX = [dsdp_slopes_west(dp, ds, Octant.O1, Axis.X) for (dp, ds, _) in cells]
    o1aY = [dsdp_slopes_west(dp, ds, Octant.O1, Axis.Y) for (dp, ds, _) in cells]
    o4aX = [dsdp_slopes_west(dp, ds, Octant.O4, Axis.X) for (dp, ds, _) in cells]
    o4aY = [dsdp_slopes_west(dp, ds, Octant.O4, Axis.Y) for (dp, ds, _) in cells]
    o5aX = [dsdp_slopes_west(dp, ds, Octant.O5, Axis.X) for (dp, ds, _) in cells]
    o5aY = [dsdp_slopes_west(dp, ds, Octant.O5, Axis.Y) for (dp, ds, _) in cells]
    o8aX = [dsdp_slopes_west(dp, ds, Octant.O8, Axis.X) for (dp, ds, _) in cells]
    o8aY = [dsdp_slopes_west(dp, ds, Octant.O8, Axis.Y) for (dp, ds, _) in cells]

    o2aX = [dsdp_slopes_west(dp, ds, Octant.O2, Axis.X) for (dp, ds, _) in cells]
    o2aY = [dsdp_slopes_west(dp, ds, Octant.O2, Axis.Y) for (dp, ds, _) in cells]
    o3aX = [dsdp_slopes_west(dp, ds, Octant.O3, Axis.X) for (dp, ds, _) in cells]
    o3aY = [dsdp_slopes_west(dp, ds, Octant.O3, Axis.Y) for (dp, ds, _) in cells]
    o6aX = [dsdp_slopes_west(dp, ds, Octant.O6, Axis.X) for (dp, ds, _) in cells]
    o6aY = [dsdp_slopes_west(dp, ds, Octant.O6, Axis.Y) for (dp, ds, _) in cells]
    o7aX = [dsdp_slopes_west(dp, ds, Octant.O7, Axis.X) for (dp, ds, _) in cells]
    o7aY = [dsdp_slopes_west(dp, ds, Octant.O7, Axis.Y) for (dp, ds, _) in cells]

    expected_o1aX = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 1.0), (1.0, 1.0)]
    expected_o1aY = [(0.0, 0.0), (0.0, 0.0), (1 / 3, 1.0), (0.0, 0.0), (1 / 3, 1.0)]
    expected_o4aX = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 1.0), (1.0, 1.0)]
    expected_o4aY = [(0.0, 0.0), (0.0, 0.0), (1 / 3, 1.0), (0.0, 0.0), (1 / 3, 1.0)]
    expected_o5aX = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 1.0), (1.0, 1.0)]
    expected_o5aY = [(0.0, 0.0), (0.0, 0.0), (1 / 3, 1.0), (0.0, 0.0), (1 / 3, 1.0)]
    expected_o8aX = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 1.0), (1.0, 1.0)]
    expected_o8aY = [(0.0, 0.0), (0.0, 0.0), (1 / 3, 1.0), (0.0, 0.0), (1 / 3, 1.0)]

    expected_o2aX = [(0.0, 1.0), (0.0, 1 / 3), (1 / 3, 1.0), (0.0, 1 / 3), (1 / 3, 1.0)]
    expected_o2aY = [(0.0, 1.0), (1 / 3, 1.0), (1.0, 1.0), (1 / 3, 1.0), (1.0, 1.0)]
    expected_o3aX = [(0.0, 1.0), (0.0, 1 / 3), (1 / 3, 1.0), (0.0, 1 / 3), (1 / 3, 1.0)]
    expected_o3aY = [(0.0, 1.0), (1 / 3, 1.0), (1.0, 1.0), (1 / 3, 1.0), (1.0, 1.0)]
    expected_o6aX = [(0.0, 1.0), (0.0, 1 / 3), (1 / 3, 1.0), (0.0, 1 / 3), (1 / 3, 1.0)]
    expected_o6aY = [(0.0, 1.0), (1 / 3, 1.0), (1.0, 1.0), (1 / 3, 1.0), (1.0, 1.0)]
    expected_o7aX = [(0.0, 1.0), (0.0, 1 / 3), (1 / 3, 1.0), (0.0, 1 / 3), (1 / 3, 1.0)]
    expected_o7aY = [(0.0, 1.0), (1 / 3, 1.0), (1.0, 1.0), (1 / 3, 1.0), (1.0, 1.0)]

    assert o1aX == expected_o1aX
    assert o1aY == expected_o1aY
    assert o2aX == expected_o2aX
    assert o2aY == expected_o2aY
    assert o3aX == expected_o3aX
    assert o3aY == expected_o3aY
    assert o4aX == expected_o4aX
    assert o4aY == expected_o4aY

    assert o5aX == expected_o5aX
    assert o5aY == expected_o5aY
    assert o6aX == expected_o6aX
    assert o6aY == expected_o6aY
    assert o7aX == expected_o7aX
    assert o7aY == expected_o7aY
    assert o8aX == expected_o8aX
    assert o8aY == expected_o8aY


def test_dzdp_slopes_cell():
    """Test dzdp slopes to the entire Cell with all octants and axes.

    Expected: octants (1,2,3,4) and (5,6,7,8) should have same slope ranges.
    """
    cells = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]

    upper = [Octant.O1, Octant.O2, Octant.O3, Octant.O4]
    lower = [Octant.O5, Octant.O6, Octant.O7, Octant.O8]

    expected_upper = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_lower = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1 / 3, 1.0), (1 / 3, 1.0)]

    for octant in upper:
        for axis in (Axis.X, Axis.Y):
            actual = [
                dzdp_slopes_cell(dp, dz, 1.0, 0.5, octant, axis)
                for (dp, _, dz) in cells
            ]
            assert actual == expected_upper

    for octant in lower:
        for axis in (Axis.X, Axis.Y):
            actual = [
                dzdp_slopes_cell(dp, dz, 1.0, 0.5, octant, axis)
                for (dp, _, dz) in cells
            ]
            assert actual == expected_lower


def test_dzdp_slopes_floor():
    """Test dzdp slopes to the entire Cell with all octants and axes.

    Expected: octants (1,2,3,4) and (5,6,7,8) should have same slope ranges.
    """
    cells = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]

    upper = [Octant.O1, Octant.O2, Octant.O3, Octant.O4]
    lower = [Octant.O5, Octant.O6, Octant.O7, Octant.O8]

    expected_upper = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1 / 3, 1.0), (1 / 3, 1.0)]
    expected_lower = [(1.0, 1.0), (1 / 3, 1.0), (1 / 3, 1.0), (1.0, 1.0), (1.0, 1.0)]

    for octant in upper:
        for axis in (Axis.X, Axis.Y):
            actual = [
                dzdp_slopes_floor(dp, dz, 0.5, octant, axis) for (dp, _, dz) in cells
            ]
            assert actual == expected_upper

    for octant in lower:
        for axis in (Axis.X, Axis.Y):
            actual = [
                dzdp_slopes_floor(dp, dz, 0.5, octant, axis) for (dp, _, dz) in cells
            ]
            assert actual == expected_lower


def test_dzdp_slopes_north():
    """Test dzdp slopes to the North wall with all octants and axes.

    Expected: octants (1, 2, 3X, 4X), (5, 6, 7X, 8X), (3Y, 4Y) and (7Y, 8Y) should
    have same slope ranges.
    """
    test_func = dzdp_slopes_north
    cells = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]

    o1aX = [test_func(dp, dz, 1.0, 0.5, Octant.O1, Axis.X) for (dp, _, dz) in cells]
    o1aY = [test_func(dp, dz, 1.0, 0.5, Octant.O1, Axis.Y) for (dp, _, dz) in cells]
    o2aX = [test_func(dp, dz, 1.0, 0.5, Octant.O2, Axis.X) for (dp, _, dz) in cells]
    o2aY = [test_func(dp, dz, 1.0, 0.5, Octant.O2, Axis.Y) for (dp, _, dz) in cells]
    o3aX = [test_func(dp, dz, 1.0, 0.5, Octant.O3, Axis.X) for (dp, _, dz) in cells]
    o3aY = [test_func(dp, dz, 1.0, 0.5, Octant.O3, Axis.Y) for (dp, _, dz) in cells]
    o4aX = [test_func(dp, dz, 1.0, 0.5, Octant.O4, Axis.X) for (dp, _, dz) in cells]
    o4aY = [test_func(dp, dz, 1.0, 0.5, Octant.O4, Axis.Y) for (dp, _, dz) in cells]

    o5aX = [test_func(dp, dz, 1.0, 0.5, Octant.O5, Axis.X) for (dp, _, dz) in cells]
    o5aY = [test_func(dp, dz, 1.0, 0.5, Octant.O5, Axis.Y) for (dp, _, dz) in cells]
    o6aX = [test_func(dp, dz, 1.0, 0.5, Octant.O6, Axis.X) for (dp, _, dz) in cells]
    o6aY = [test_func(dp, dz, 1.0, 0.5, Octant.O6, Axis.Y) for (dp, _, dz) in cells]
    o7aX = [test_func(dp, dz, 1.0, 0.5, Octant.O7, Axis.X) for (dp, _, dz) in cells]
    o7aY = [test_func(dp, dz, 1.0, 0.5, Octant.O7, Axis.Y) for (dp, _, dz) in cells]
    o8aX = [test_func(dp, dz, 1.0, 0.5, Octant.O8, Axis.X) for (dp, _, dz) in cells]
    o8aY = [test_func(dp, dz, 1.0, 0.5, Octant.O8, Axis.Y) for (dp, _, dz) in cells]

    expected_o1aX = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o1aY = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o2aX = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o2aY = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]

    expected_o3aX = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o3aY = [(0.0, 1.0), (0.0, 1 / 3), (0.0, 1 / 3), (1 / 3, 1.0), (1 / 3, 1.0)]
    expected_o4aX = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o4aY = [(0.0, 1.0), (0.0, 1 / 3), (0.0, 1 / 3), (1 / 3, 1.0), (1 / 3, 1.0)]

    expected_o5aX = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o5aY = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o6aX = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o6aY = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]

    expected_o7aX = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o7aY = [(0.0, 1.0), (0.0, 1 / 3), (0.0, 1 / 3), (1 / 3, 1.0), (1 / 3, 1.0)]
    expected_o8aX = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o8aY = [(0.0, 1.0), (0.0, 1 / 3), (0.0, 1 / 3), (1 / 3, 1.0), (1 / 3, 1.0)]

    assert o1aX == expected_o1aX
    assert o1aY == expected_o1aY
    assert o2aX == expected_o2aX
    assert o2aY == expected_o2aY
    assert o3aX == expected_o3aX
    assert o3aY == expected_o3aY
    assert o4aX == expected_o4aX
    assert o4aY == expected_o4aY

    assert o5aX == expected_o5aX
    assert o5aY == expected_o5aY
    assert o6aX == expected_o6aX
    assert o6aY == expected_o6aY
    assert o7aX == expected_o7aX
    assert o7aY == expected_o7aY
    assert o8aX == expected_o8aX
    assert o8aY == expected_o8aY


def test_dzdp_slopes_west():
    """Test dzdp slopes to the West wall with all octants and axes.

    Expected: octants (1, 2Y, 3Y, 4), (5, 6Y, 7Y, 8), (2X, 3X) and (6X, 7X) should
    have same slope ranges."""
    test_func = dzdp_slopes_west
    cells = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]

    o1aX = [test_func(dp, dz, 1.0, 0.5, Octant.O1, Axis.X) for (dp, _, dz) in cells]
    o1aY = [test_func(dp, dz, 1.0, 0.5, Octant.O1, Axis.Y) for (dp, _, dz) in cells]
    o2aX = [test_func(dp, dz, 1.0, 0.5, Octant.O2, Axis.X) for (dp, _, dz) in cells]
    o2aY = [test_func(dp, dz, 1.0, 0.5, Octant.O2, Axis.Y) for (dp, _, dz) in cells]
    o3aX = [test_func(dp, dz, 1.0, 0.5, Octant.O3, Axis.X) for (dp, _, dz) in cells]
    o3aY = [test_func(dp, dz, 1.0, 0.5, Octant.O3, Axis.Y) for (dp, _, dz) in cells]
    o4aX = [test_func(dp, dz, 1.0, 0.5, Octant.O4, Axis.X) for (dp, _, dz) in cells]
    o4aY = [test_func(dp, dz, 1.0, 0.5, Octant.O4, Axis.Y) for (dp, _, dz) in cells]

    o5aX = [test_func(dp, dz, 1.0, 0.5, Octant.O5, Axis.X) for (dp, _, dz) in cells]
    o5aY = [test_func(dp, dz, 1.0, 0.5, Octant.O5, Axis.Y) for (dp, _, dz) in cells]
    o6aX = [test_func(dp, dz, 1.0, 0.5, Octant.O6, Axis.X) for (dp, _, dz) in cells]
    o6aY = [test_func(dp, dz, 1.0, 0.5, Octant.O6, Axis.Y) for (dp, _, dz) in cells]
    o7aX = [test_func(dp, dz, 1.0, 0.5, Octant.O7, Axis.X) for (dp, _, dz) in cells]
    o7aY = [test_func(dp, dz, 1.0, 0.5, Octant.O7, Axis.Y) for (dp, _, dz) in cells]
    o8aX = [test_func(dp, dz, 1.0, 0.5, Octant.O8, Axis.X) for (dp, _, dz) in cells]
    o8aY = [test_func(dp, dz, 1.0, 0.5, Octant.O8, Axis.Y) for (dp, _, dz) in cells]

    expected_o1aX = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o1aY = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o2aX = [(0.0, 1.0), (0.0, 1 / 3), (0.0, 1 / 3), (1 / 3, 1.0), (1 / 3, 1.0)]
    expected_o2aY = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]

    expected_o3aX = [(0.0, 1.0), (0.0, 1 / 3), (0.0, 1 / 3), (1 / 3, 1.0), (1 / 3, 1.0)]
    expected_o3aY = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o4aX = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o4aY = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]

    expected_o5aX = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o5aY = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o6aX = [(0.0, 1.0), (0.0, 1 / 3), (0.0, 1 / 3), (1 / 3, 1.0), (1 / 3, 1.0)]
    expected_o6aY = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]

    expected_o7aX = [(0.0, 1.0), (0.0, 1 / 3), (0.0, 1 / 3), (1 / 3, 1.0), (1 / 3, 1.0)]
    expected_o7aY = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o8aX = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    expected_o8aY = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]

    assert o1aX == expected_o1aX
    assert o1aY == expected_o1aY
    assert o2aX == expected_o2aX
    assert o2aY == expected_o2aY
    assert o3aX == expected_o3aX
    assert o3aY == expected_o3aY
    assert o4aX == expected_o4aX
    assert o4aY == expected_o4aY

    assert o5aX == expected_o5aX
    assert o5aY == expected_o5aY
    assert o6aX == expected_o6aX
    assert o6aY == expected_o6aY
    assert o7aX == expected_o7aX
    assert o7aY == expected_o7aY
    assert o8aX == expected_o8aX
    assert o8aY == expected_o8aY


def test_octant_to_relative():
    octants = [
        Octant.O1,
        Octant.O2,
        Octant.O3,
        Octant.O4,
        Octant.O5,
        Octant.O6,
        Octant.O7,
        Octant.O8,
    ]
    axes = [Axis.X, Axis.Y, Axis.Z]
    actual = [
        octant_to_relative(1, 2, 3, o, a).as_tuple() for o in octants for a in axes
    ]
    expected = [
        (1, 2, 3),
        (2, 1, 3),
        (1, 2, 3),
        (-1, 2, 3),
        (-2, 1, 3),
        (-1, 2, 3),
        (-1, -2, 3),
        (-2, -1, 3),
        (-1, -2, 3),
        (1, -2, 3),
        (2, -1, 3),
        (1, -2, 3),
        (1, 2, -3),
        (2, 1, -3),
        (1, 2, -3),
        (-1, 2, -3),
        (-2, 1, -3),
        (-1, 2, -3),
        (-1, -2, -3),
        (-2, -1, -3),
        (-1, -2, -3),
        (1, -2, -3),
        (2, -1, -3),
        (1, -2, -3),
    ]
    assert actual == expected


if __name__ == "__main__":
    print("\n----- Map Functions 3D -----")

    # for octant in [Octant.O1, Octant.O2, Octant.O3, Octant.O4, Octant.O5, Octant.O6, Octant.O7, Octant.O8]:
    # for axis in [Axis.X, Axis.Y, Axis.Z]:
    for octant in [
        Octant.O1,
        # Octant.O2,
        # Octant.O3,
        # Octant.O4,
        # Octant.O5,
        # Octant.O6,
        # Octant.O7,
        # Octant.O8,
    ]:
        print(octant)
        for axis in [Axis.X, Axis.Y, Axis.Z]:
            for dpri, dsec, dz in [(0, 0, 0), (1, 0, 0), (1, 1, 0)]:
                rel_coords = octant_to_relative(dpri, dsec, dz, octant, axis)
                print(f"[{axis}] {dpri, dsec, dz} - relative: {rel_coords}")

    p1 = Point(0.0, 0.0, 0.0)
    p2 = Point(3.0, 4.0, 10.0)
    print(f"distance: {p1.distance(p2)}")
