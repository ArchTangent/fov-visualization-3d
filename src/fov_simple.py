"""30 JUL 2023 - 3D FOV - Simple.

Key Ideas:
- Max in-game FOV is 127 and number of FOV bits is 256 (128x2)
- Uses angle ranges to determine which FOV rays are visible/blocked
- All structures are cuboids of full cell width and full cell height.
- All walls are rectangles of full cell width and full cell height.
- Peformance is improved by pre-baking FovMaps by (a) radius and (b) eye_ht.

NOTE: all FovOctants include FovCell 0 (origin). This is by design, as the 
Rect-based fov_calc() doesn't have the same issue as the slope-range-based
version. 
"""
import math
import pygame, pygame.freetype
from enum import Enum
from pygame import Color, Surface, Vector2
from pygame.freetype import Font
from typing import Dict, List, Self, Tuple
from helpers import (
    Axis,
    Coords,
    FovRect,
    Octant,
    Point,
    QBits,
    Vector,
    octant_to_relative,
    ray_rect_intersection,
    to_cell_id,
    xyz_boundary_distance,
)

# Isometric shifts for cell dimensions
XS = 16.0
YS = 8.0
ZS = 32.0


class Blockers:
    """FOV blocking data for CellMap construction."""

    def __init__(
        self,
        floor: bool = False,
        structure: int = 0,
        wall_n: int = 0,
        wall_w: int = 0,
    ) -> None:
        self.floor = floor
        self.structure = structure
        self.wall_n = wall_n
        self.wall_w = wall_w


class Settings:
    """Settings for Pygame."""

    def __init__(
        self,
        width: int,
        height: int,
        map_dims: Coords,
        font: Font,
        font_color: Color,
        radius: int = 63,
        line_width: int = 1,
        floor_color="steelblue2",
        floor_trim_color="steelblue4",
        wall_color="seagreen3",
        wall_trim_color="seagreen4",
        structure_color="seagreen3",
        structure_trim_color="seagreen4",
    ) -> None:
        if map_dims.x < 1 or map_dims.y < 1 or map_dims.z < 1:
            raise ValueError("all map dimensions must be > 0!")

        self.width = width
        self.height = height
        self.line_width = line_width
        self.font = font
        self.font_color = font_color
        self.radius = radius
        self.max_radius = 127
        self.floor_color = Color(floor_color)
        self.floor_trim_color = Color(floor_trim_color)
        self.wall_color = Color(wall_color)
        self.wall_trim_color = Color(wall_trim_color)
        self.structure_color = Color(structure_color)
        self.structure_trim_color = Color(structure_trim_color)
        self.xdims, self.ydims, self.zdims = map_dims.x, map_dims.y, map_dims.z


class FovLine:
    """Data for a line in `FovLines` with origin `r0` and vector `rv`."""

    __slots__ = "array_ix", "bit_ix", "r0", "rv"

    def __init__(self, array_ix: int, bit_ix: int, r0: Point, rv: Vector) -> None:
        self.array_ix = array_ix
        self.bit_ix = bit_ix
        self.r0 = r0
        self.rv = rv

    def __repr__(self) -> str:
        r0, rv = self.r0, self.rv
        aix, bix = self.array_ix, self.bit_ix
        return f"FovLine {r0} + {rv}, index[{aix}] bit {bix}"


class FovLines:
    """List of (x1, y1, z1, x2, y2, z2) start/end points for FOV lines in range [0, radius].

    There is one FOV bit / FOV index for each FOV line, for each z-level.

    (rx, ry, rz) are cells relative to the origin; (vx, vy, vz) are vectors from the origin
    r0 to 'center cell' of the relative cells (rx + 0.5, ry + 0.5, rz + 0.5).

    FOV lines are fired from the xy center of the tile at eye height: (0.5, 0.5, eye_ht).

    NOTE: each FovCell is for a particular Octant-Axis pairing, and gets exactly one group of
    associated FovLines, matching that pairing (i.e. Octant 1X, Octant 7Z).
    """

    __slots__ = "lines"

    def __init__(
        self, eye_ht: float, radius: int, z_levels: int, octant: Octant, axis: Axis
    ) -> None:
        self.lines: List[FovLine] = []
        r0x, r0y, r0z = 0.5, 0.5, eye_ht
        r0 = Point(r0x, r0y, r0z)

        match axis:
            # --- X-Axis --- #
            # Hold primary X/Y fixed as radius, Z is tertiary
            case Axis.X | Axis.Y:
                for ter in range(z_levels):
                    for sec in range(radius + 1):
                        rx, ry, rz = octant_to_relative(radius, sec, ter, octant, axis)
                        vx, vy, vz = rx + 0.5 - r0x, ry + 0.5 - r0y, rz + 0.5 - r0z
                        line = FovLine(ter, sec, r0, Vector(vx, vy, vz))
                        self.lines.append(line)
            # --- Z-Axis --- #
            # Hold primary Z fixed as z-levels - 1, X is secondary, Y is tertiary
            case Axis.Z:
                for sec in range(radius + 1):
                    for ter in range(radius + 1):
                        rx, ry, rz = octant_to_relative(
                            z_levels - 1, sec, ter, octant, axis
                        )
                        vx, vy, vz = rx + 0.5 - r0x, ry + 0.5 - r0y, rz + 0.5 - r0z
                        line = FovLine(ter, sec, r0, Vector(vx, vy, vz))
                        self.lines.append(line)

    def __iter__(self):
        return iter(self.lines)


class FovCell:
    """3D FOV Cell used in an `FovMap` at a given octant-axis pairing.

    An FOV cell is visible if at least one of its `visible_bits` is not blocked
    by `blocked` bits in the FOV calculation.

    ### Big Picture

    1.) Get coordinates relative to origin/observer: (rx, ry, rz)
    2.) Get rectangles representing North wall, West wall, and structure
    3.) For X/Y/Z-primary: bits needed for cell to be visible (visible_bits)
    4.) For X/Y/Z-primary: bits blocked by N/W wall (wall_bits_n/w)
    5.) For X/Y/Z-primary: bits blocked by a structure (structure_bits)

    NOTE: for fov_simple, all walls and structures are full-height.

    ### Parameters

    `cix`: int
        Cell index within the Octant and list of FOV cells.
    `dpri, dsec, dter`: int
        Relative (pri,sec,ter) coordinates of the FOV cell compared to FOV origin.
    `indices`: int
        Number of indices in which bitflags are stored. This is z_levels for
        XY-primary FovCells, and qbits for Z-primary FovCells.

    ### Fields

    `dsec, dz`: int
        Distance along secondary and Z axes.  Used for filtering out-of-bounds cells.
    `visible_bits`: List[int]
        Array of Bitflags spanning the visible slope ranges for the cell. The Δsec/Δpri
        range is stored in the bits; the Δz range is stored by the list index.
    """

    def __init__(
        self,
        cix: int,
        dpri: int,
        dsec: int,
        dter: int,
        indices: int,
        octant: Octant,
        axis: Axis,
        fov_lines: FovLines,
    ):
        # --- General Cell Information --- #
        # Octant-adjusted relative x/y/z used to select cell in CellMap
        rx, ry, rz = octant_to_relative(dpri, dsec, dter, octant, axis)
        self.rx, self.ry, self.rz = rx, ry, rz
        self.dsec, self.dter = dsec, dter
        self.cix = cix

        # --- Blocking and Visible Bits --- #
        wall_n = self.wall_n_rect(rx, ry, rz)
        wall_w = self.wall_w_rect(rx, ry, rz)
        floor = self.floor_rect(rx, ry, rz)
        structure = self.structure_rects(rx, ry, rz, octant)

        self.wall_n = wall_n
        self.wall_w = wall_w
        self.floor = floor
        self.structure = structure
        self.wall_n_bits: List[int] = [0 for z in range(indices)]
        self.wall_w_bits: List[int] = [0 for z in range(indices)]
        self.floor_bits: List[int] = [0 for z in range(indices)]
        self.structure_bits: List[int] = [0 for z in range(indices)]
        self.visible_bits: List[int] = [0 for z in range(indices)]

        # Set blocking and visible bits from walls, floor and structure
        for fov_line in fov_lines.lines:
            array_ix = fov_line.array_ix
            bit_ix = fov_line.bit_ix
            r0 = fov_line.r0
            rv = fov_line.rv

            if ray_rect_intersection(r0, rv, wall_n):
                self.wall_n_bits[array_ix] |= bit_ix

            if ray_rect_intersection(r0, rv, wall_w):
                self.wall_w_bits[array_ix] |= bit_ix

            if ray_rect_intersection(r0, rv, floor):
                self.floor_bits[array_ix] |= bit_ix

            if ray_rect_intersection(r0, rv, structure[0]):
                self.structure_bits[array_ix] |= bit_ix
                self.visible_bits[array_ix] |= bit_ix

            if ray_rect_intersection(r0, rv, structure[1]):
                self.structure_bits[array_ix] |= bit_ix
                self.visible_bits[array_ix] |= bit_ix

    def __repr__(self) -> str:
        return f"FovCell {self.cix}, Rel {self.rx ,self.ry, self.rz}"

    def ceiling_rect(self, rx: int, ry: int, rz: int) -> FovRect:
        """Returns FovRect representing a ceiling in the cell."""
        r0 = Point(rx, ry, rz + 1.0)
        s1 = Vector(1.0, 0.0, 0.0)
        s2 = Vector(0.0, 1.0, 0.0)
        normal = Vector(0.0, 0.0, 1.0)
        rect = FovRect(r0, s1, s2, 1.0, 1.0, normal)

        return rect

    def floor_rect(self, rx: int, ry: int, rz: int) -> FovRect:
        """Returns FovRect representing a floor in the cell."""
        r0 = Point(rx, ry, rz)
        s1 = Vector(1.0, 0.0, 0.0)
        s2 = Vector(0.0, 1.0, 0.0)
        normal = Vector(0.0, 0.0, 1.0)
        rect = FovRect(r0, s1, s2, 1.0, 1.0, normal)

        return rect

    def structure_rects(
        self, rx: int, ry: int, rz: int, octant: Octant
    ) -> Tuple[FovRect, FovRect, FovRect]:
        """Returns three FovRects representing a structure in the cell.

        Rectangles consist of:
        - two walls AND
        - one floor (upper octants) OR one ceiling (lower octants)
        """
        match octant:
            case Octant.O1:
                wall1 = self.wall_n_rect(rx, ry, rz)
                wall2 = self.wall_w_rect(rx, ry, rz)
                plane = self.floor_rect(rx, ry, rz)
            case Octant.O2:
                wall1 = self.wall_n_rect(rx, ry, rz)
                wall2 = self.wall_e_rect(rx, ry, rz)
                plane = self.floor_rect(rx, ry, rz)
            case Octant.O3:
                wall1 = self.wall_s_rect(rx, ry, rz)
                wall2 = self.wall_e_rect(rx, ry, rz)
                plane = self.floor_rect(rx, ry, rz)
            case Octant.O4:
                wall1 = self.wall_s_rect(rx, ry, rz)
                wall2 = self.wall_w_rect(rx, ry, rz)
                plane = self.floor_rect(rx, ry, rz)
            case Octant.O5:
                wall1 = self.wall_n_rect(rx, ry, rz)
                wall2 = self.wall_w_rect(rx, ry, rz)
                plane = self.ceiling_rect(rx, ry, rz)
            case Octant.O6:
                wall1 = self.wall_n_rect(rx, ry, rz)
                wall2 = self.wall_e_rect(rx, ry, rz)
                plane = self.ceiling_rect(rx, ry, rz)
            case Octant.O7:
                wall1 = self.wall_s_rect(rx, ry, rz)
                wall2 = self.wall_e_rect(rx, ry, rz)
                plane = self.ceiling_rect(rx, ry, rz)
            case Octant.O8:
                wall1 = self.wall_s_rect(rx, ry, rz)
                wall2 = self.wall_w_rect(rx, ry, rz)
                plane = self.ceiling_rect(rx, ry, rz)

        return wall1, wall2, plane

    def wall_e_rect(self, rx: int, ry: int, rz: int) -> FovRect:
        """Returns FovRect representing East wall (for structures)."""
        r0 = Point(rx + 1.0, ry, rz)
        s1 = Vector(0.0, 1.0, 0.0)
        s2 = Vector(0.0, 0.0, 1.0)
        normal = Vector(1.0, 0.0, 0.0)
        rect = FovRect(r0, s1, s2, 1.0, 1.0, normal)

        return rect

    def wall_n_rect(self, rx: int, ry: int, rz: int) -> FovRect:
        """Returns FovRect representing North wall."""
        r0 = Point(rx, ry, rz)
        s1 = Vector(1.0, 0.0, 0.0)
        s2 = Vector(0.0, 0.0, 1.0)
        normal = Vector(0.0, -1.0, 0.0)
        rect = FovRect(r0, s1, s2, 1.0, 1.0, normal)

        return rect

    def wall_s_rect(self, rx: int, ry: int, rz: int) -> FovRect:
        """Returns FovRect representing South wall (for structures)."""
        r0 = Point(rx, ry + 1.0, rz)
        s1 = Vector(1.0, 0.0, 0.0)
        s2 = Vector(0.0, 0.0, 1.0)
        normal = Vector(0.0, 1.0, 0.0)
        rect = FovRect(r0, s1, s2, 1.0, 1.0, normal)

        return rect

    def wall_w_rect(self, rx: int, ry: int, rz: int) -> FovRect:
        """Returns FovRect representing West wall."""
        r0 = Point(rx, ry, rz)
        s1 = Vector(0.0, 1.0, 0.0)
        s2 = Vector(0.0, 0.0, 1.0)
        normal = Vector(-1.0, 0.0, 0.0)
        rect = FovRect(r0, s1, s2, 1.0, 1.0, normal)

        return rect


class FovOctant:
    """One of 8 octants within a 3D FovMap, with data for FOV calculations.

    ### X and Y Primary

    Slope targets and thresholds:  z-slope is taken from origin (at eye height)
    to center cell at the highest z-level (or lowest, for lower octants).
    Z-thresholds are calculated for each primary increment, determining how high
    (or low) a z-slice will go at that primary value.

    For Octants 1-4:
    ```
    z_slope = (z_levels - 0.5 - eye_ht) / radius

    z_threshold = int(eye_ht + z_slope * (pri + 0.5)) + 1
    ```

    For Octants 5-8:
    ```
    z_slope = (z_levels - 0.5 - (1.0 - eye_ht)) / radius

    z_threshold = int(eye_ht + z_slope * (pri + 0.5)) + 1
    ```

    ### Z Primary

    The calculation for xy_threshold varies slightly between +Z/-Z octants.

    For Octants 1-4:
    ```
    xy_slope = radius / (z_levels - 0.5 - eye_ht)

    xy_threshold = min(int(0.5 + xy_slope * (pri + 1.0 - eye_ht)) + 1, radius + 1)
    ```

    For Octants 5-8:
    ```
    xy_slope = radius / (z_levels - 0.5 - (1.0 - eye_ht))

    xy_threshold = min(int(0.5 + xy_slope * (pri + eye_ht)) + 1, radius + 1)
    ```

    Other notes:
    - `xy_threshold` is the farthest XY value allowed for an FovTile at a given
      Z slice to be allowed in a Z-primary FovOctant.
    - `m` is the radius appromixation margin. This only applies to the X
       and Y axes, making the FOV cylindrical in shape.

    ### Parameters

    radius: int
        Maximum in-game FOV radius.  Used to set highest X/Y pri/sec value.
    octant: Octant
        One of 24 Octant-Axis pairings represented by this instance.

    ### Fields

    `x`, `y`, `z`: List[Fovcell]
        FovCells for each primary axis.
    `max_fov_ix_x`, `max_fov_ix_y`, `max_fov_ix_z`: List[int]
        Maximum FovCell index of x,y, or z for a given radius. For example,
        max_x[32] gives the index of the farthest FovCell in FovOctant.x
        for a radius of 32.
    """

    def __init__(self, cells: List[FovCell], max_fov_ix: List[int]):
        self.cells = cells
        self.max_fov_ix = max_fov_ix

    @staticmethod
    def new(eye_ht: float, radius: int, z_levels: int, octant: Octant, axis: Axis):
        cells: List[FovCell] = []
        max_fov_ix: List[int] = []
        limit = radius * radius
        fov_ix = 0
        cix = 0
        m = 0.5

        fov_lines = FovLines(eye_ht, radius, z_levels, octant, axis)

        # --- X/Y Primary --- #
        if axis == Axis.X or axis == Axis.Y:
            z_thresholds = FovOctant.z_thresholds(eye_ht, radius, z_levels, octant)
            sec_threshold = 2

            for dpri in range(radius + 1):
                for dsec in range(sec_threshold):
                    # Radius filter
                    if dpri == 0:
                        r = (dpri - m) * (dpri - m) + (dsec * dsec)
                    else:
                        r = (dpri - m) * (dpri - m) + (dsec - m) * (dsec - m)

                    if r > limit:
                        continue

                    z_threshold = z_thresholds[dpri]

                    for dter in range(z_threshold):
                        # Update both X and Y in their respective lists
                        cell = FovCell(
                            cix, dpri, dsec, dter, z_levels, octant, axis, fov_lines
                        )
                        cells.append(cell)
                        fov_ix += 1
                        cix += 1

                max_fov_ix.append(fov_ix)
                sec_threshold += 1

        # --- Z Primary --- #
        else:
            print(f"=== Z PRIMARY ===")
            xy_thresholds = FovOctant.xy_thresholds(eye_ht, radius, z_levels, octant)

            for dpri in range(z_levels):
                xy_threshold = xy_thresholds[dpri]
                print(f"  XY Threshold: {xy_threshold}")

                for dsec in range(xy_threshold):
                    # Radius filter
                    if dpri == 0:
                        r = (dpri - m) * (dpri - m) + (dsec * dsec)
                    else:
                        r = (dpri - m) * (dpri - m) + (dsec - m) * (dsec - m)

                    if r > limit:
                        continue

                    for dter in range(xy_threshold):
                        cell = FovCell(
                            cix, dpri, dsec, dter, z_levels, octant, axis, fov_lines
                        )
                        cells.append(cell)
                        fov_ix += 1
                        cix += 1

                max_fov_ix.append(fov_ix)

        return FovOctant(cells, max_fov_ix)

    @staticmethod
    def xy_thresholds(
        eye_ht: float, radius: int, z_levels: int, octant: Octant
    ) -> List[int]:
        """Return list of XY thresholds by Z-level."""
        result = []
        z0 = eye_ht

        match octant:
            case Octant.O1 | Octant.O2 | Octant.O3 | Octant.O4:
                zf = z_levels - 0.5

                for z in range(z_levels):
                    if z == z_levels - 1:
                        result.append(radius)
                    else:
                        zt = z + 1.0
                        xyt = int(0.5 + ((zt - z0) / (zf - z0)) * radius)
                        result.append(xyt)

            case Octant.O5 | Octant.O6 | Octant.O7 | Octant.O8:
                zf = -z_levels + 1.5

                for z in range(z_levels):
                    if z == z_levels - 1:
                        result.append(radius)
                    else:
                        zt = -z
                        xyt = int(0.5 + ((zt - z0) / (zf - z0)) * radius)
                        result.append(xyt)

        return result

    @staticmethod
    def z_thresholds(eye_ht: float, radius: int, z_levels: int, octant: Octant):
        """Returns list of Z thresholds by dpri value."""
        result = []
        z0 = eye_ht

        match octant:
            case Octant.O1 | Octant.O2 | Octant.O3 | Octant.O4:
                zf = z_levels - 0.5
            case Octant.O5 | Octant.O6 | Octant.O7 | Octant.O8:
                zf = -z_levels + 1.5

        for dpri in range(radius + 1):
            dzdp = (zf - z0) / radius
            zt = abs(math.floor(z0 + dzdp * (dpri + 0.5)))
            result.append(zt)

        return result


class FovMap:
    """3D FOV map of FovCells used with CellMap to determine visible cells.

    There are eight 3D octants with three primary axes, for 24 sets of FovCells.

    NOTE: `0.0 <= eye_ht < 1.0`.

    Parameters
    ---
    `eye_ht`: float
        Height, proportional to cell height, of the observer's FOV within the cell.
        Must be in range `[0.0, 1.0)`.
    `z_levels`: int
        Number of z-levels in the game.  Typically 4 to 8.
    `qbits`: QBits
        Enum representing number of bits to use along the X/Y axes.
    """

    def __init__(self, eye_ht: float, z_levels: int, qbits: QBits) -> None:
        self.eye_ht = eye_ht
        self.z_levels = z_levels
        self.qbits = qbits.value

        radius = qbits.value - 1

        self.octant_1x = FovOctant.new(eye_ht, radius, z_levels, Octant.O1, Axis.X)
        self.octant_1y = FovOctant.new(eye_ht, radius, z_levels, Octant.O1, Axis.Y)
        self.octant_1z = FovOctant.new(eye_ht, radius, z_levels, Octant.O1, Axis.Z)
        # self.octant_2x = FovOctant(radius, Octant.O2, Axis.X)
        # self.octant_2y = FovOctant(radius, Octant.O2, Axis.Y)
        # self.octant_2z = FovOctant(radius, Octant.O2, Axis.Z)
        # self.octant_3x = FovOctant(radius, Octant.O3, Axis.X)
        # self.octant_3y = FovOctant(radius, Octant.O3, Axis.Y)
        # self.octant_3z = FovOctant(radius, Octant.O3, Axis.Z)
        # self.octant_4x = FovOctant(radius, Octant.O4, Axis.X)
        # self.octant_4y = FovOctant(radius, Octant.O4, Axis.Y)
        # self.octant_4z = FovOctant(radius, Octant.O4, Axis.Z)
        # self.octant_5x = FovOctant(radius, Octant.O5, Axis.X)
        # self.octant_5y = FovOctant(radius, Octant.O5, Axis.Y)
        # self.octant_5z = FovOctant(radius, Octant.O5, Axis.Z)
        # self.octant_6x = FovOctant(radius, Octant.O6, Axis.X)
        # self.octant_6y = FovOctant(radius, Octant.O6, Axis.Y)
        # self.octant_6z = FovOctant(radius, Octant.O6, Axis.Z)
        # self.octant_7x = FovOctant(radius, Octant.O7, Axis.X)
        # self.octant_7y = FovOctant(radius, Octant.O7, Axis.Y)
        # self.octant_7z = FovOctant(radius, Octant.O7, Axis.Z)
        # self.octant_8x = FovOctant(radius, Octant.O8, Axis.X)
        # self.octant_8y = FovOctant(radius, Octant.O8, Axis.Y)
        # self.octant_8z = FovOctant(radius, Octant.O8, Axis.Z)


#   ########    ####    ##    ##
#   ##        ##    ##  ##    ##
#   ######    ##    ##  ##    ##
#   ##        ##    ##   ##  ##
#   ##         ######      ##


#   #######   #######      ##     ##    ##
#   ##    ##  ##    ##   ##  ##   ##    ##
#   ##    ##  #######   ##    ##  ## ## ##
#   ##    ##  ##   ##   ########  ###  ###
#   #######   ##    ##  ##    ##   ##  ##


#   ########  ########   ######   ########
#      ##     ##        ##           ##
#      ##     ######     ######      ##
#      ##     ##              ##     ##
#      ##     ########  #######      ##


def check_fov_cells(z_levels: int, qbits: QBits, octant: Octant, axis: Axis):
    """ "Calls check_fov_cell for each cell in suite."""
    eye_ht = 0.7
    radius = qbits.value - 1
    fov_lines = FovLines(eye_ht, radius, z_levels, octant, axis)
    relative_points = [(5, 0, 0)]

    for dpri, dsec, dter in relative_points:
        check_fov_cell(dpri, dsec, dter, fov_lines, z_levels, qbits, octant, axis)


def check_fov_cell(
    dpri: int,
    dsec: int,
    dter: int,
    fov_lines: FovLines,
    z_levels: int,
    qbits: QBits,
    octant: Octant,
    axis: Axis,
):
    """Checks intersections with FovCells."""
    match axis:
        case Axis.X | Axis.Y:
            indices = z_levels
        case Axis.Z:
            indices = qbits.value

    print(f"--- Check FOV Cell ---")
    fc = FovCell(-1, dpri, dsec, dter, indices, octant, axis, fov_lines)
    print(f"{fc} in {octant}, {axis}")
    print(f"# of FOV lines: {len(fov_lines.lines)}")

    for fov_line in fov_lines:
        r0 = fov_line.r0
        rv = fov_line.rv
        summary = []

        wall_n = ray_rect_intersection(r0, rv, fc.wall_n)
        if wall_n:
            isect = wall_n.rounded()
            dist = round(r0.distance(isect), 3)
            summary.append(f"  intersects wall_n at    {isect} at distance {dist}")

        wall_w = ray_rect_intersection(r0, rv, fc.wall_w)
        if wall_w:
            isect = wall_w.rounded()
            dist = round(r0.distance(isect), 3)
            summary.append(f"  intersects wall_w at    {isect} at distance {dist}")

        floor = ray_rect_intersection(r0, rv, fc.floor)
        if floor:
            isect = floor.rounded()
            dist = round(r0.distance(isect), 3)
            summary.append(f"  intersects floor at     {isect} at distance {dist}")

        structure0 = ray_rect_intersection(r0, rv, fc.structure[0])
        structure1 = ray_rect_intersection(r0, rv, fc.structure[1])
        structure2 = ray_rect_intersection(r0, rv, fc.structure[2])
        if structure0:
            isect = structure0.rounded()
            dist = round(r0.distance(isect), 3)
            summary.append(
                f"  intersects structure (wall) at {isect} at distance {dist}"
            )
        if structure1:
            isect = structure1.rounded()
            dist = round(r0.distance(isect), 3)
            summary.append(
                f"  intersects structure (wall) at {isect} at distance {dist}"
            )
        if structure2:
            isect = structure2.rounded()
            dist = round(r0.distance(isect), 3)
            summary.append(
                f"  intersects structure (plane) at {isect} at distance {dist}"
            )

        if len(summary) > 0:
            print(fov_line)
            for line in summary:
                print(line)


#   ##    ##     ##     ########  ##    ##
#   ###  ###   ##  ##      ##     ####  ##
#   ## ## ##  ##    ##     ##     ## ## ##
#   ##    ##  ########     ##     ##  ####
#   ##    ##  ##    ##  ########  ##    ##

if __name__ == "__main__":
    print(f"\n===== 3D FOV SIMPLE TESTING =====\n")
    pygame.freetype.init()

    # NOTE testing
    z_levels = 8
    radius = 7
    eye_ht = 0.7
    octant = Octant.O5
    axis = Axis.X
    qbits = QBits.Q32
    fovlines = FovLines(eye_ht, radius, z_levels, octant, axis)
    check_fov_cell(3, 1, 0, fovlines, z_levels, qbits, octant, axis)
