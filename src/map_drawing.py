"""Top-down drawing functions for 3D maps."""
import pygame, pygame.freetype
from pygame import Vector2
from pygame.color import Color
from pygame.surface import Surface
from helpers import FovLineType
from lines import bresenham, bresenham_full

# TODO: update for 3D

def draw_floor(screen: Surface, pr: Vector2, ts: int, color: Color):
    """Draws a floor tile with reference point `pr` and tile size `ts`."""
    p1 = Vector2(pr.x, pr.y)
    p2 = Vector2(pr.x + ts, pr.y)
    p3 = Vector2(pr.x + ts, pr.y + ts)
    p4 = Vector2(pr.x, pr.y + ts)

    pygame.draw.lines(screen, color, True, [p1, p2, p3, p4], width=2)


def draw_north_wall(
    screen: Surface,
    pr: Vector2,
    ts: int,
    sts: int,
    width: int,
    color: Color,
    trim: Color,
):
    """Draws a north wall w/reference `pr`, tile size `ts`, and subtile size `sts`."""
    p1 = Vector2(pr.x, pr.y)
    p2 = Vector2(pr.x + ts, pr.y)
    p3 = Vector2(pr.x + ts, pr.y + sts)
    p4 = Vector2(pr.x, pr.y + sts)

    pygame.draw.polygon(screen, color, [p1, p2, p3, p4])
    pygame.draw.lines(screen, trim, True, [p1, p2, p3, p4], width=width)


def draw_west_wall(
    screen: Surface,
    pr: Vector2,
    ts: int,
    sts: int,
    width: int,
    color: Color,
    trim: Color,
):
    """Draws a north wall w/reference `pr`, tile size `ts`, and subtile size `sts`."""
    p1 = Vector2(pr.x, pr.y)
    p2 = Vector2(pr.x + sts, pr.y)
    p3 = Vector2(pr.x + sts, pr.y + ts)
    p4 = Vector2(pr.x, pr.y + ts)

    pygame.draw.polygon(screen, color, [p1, p2, p3, p4])
    pygame.draw.lines(screen, trim, True, [p1, p2, p3, p4], width=width)


def draw_structure(
    screen: Surface, pr: Vector2, ts: int, width: int, color: Color, trim: Color
):
    """Draws a structure in a tile with reference point `pr` and tile size `ts`."""
    p1 = Vector2(pr.x, pr.y)
    p2 = Vector2(pr.x + ts, pr.y)
    p3 = Vector2(pr.x + ts, pr.y + ts)
    p4 = Vector2(pr.x, pr.y + ts)

    pygame.draw.polygon(screen, color, [p1, p2, p3, p4])
    pygame.draw.lines(screen, trim, True, [p1, p2, p3, p4], width=width)


def draw_tile(screen: Surface, tile, settings):
    """Renders a visible Tile on the map."""
    p1 = tile.p1
    p1x, p1y = tile.p1
    s = settings
    w = s.line_width
    ts = s.tile_size
    sts = s.subtile_size
    trim_color = settings.floor_trim_color

    # Draw grid if no structure present
    if not tile.blocks_sight:
        for dx in range(1, settings.subtiles_xy):
            x1 = p1x + dx * sts
            y1 = p1y
            x2 = p1x + dx * sts
            y2 = p1y + ts
            pygame.draw.line(screen, trim_color, (x1, y1), (x2, y2))

        for dy in range(1, settings.subtiles_xy):
            x1 = p1x
            y1 = p1y + dy * sts
            x2 = p1x + ts
            y2 = p1y + dy * sts
            pygame.draw.line(screen, trim_color, (x1, y1), (x2, y2))

        draw_floor(screen, p1, ts, s.floor_color)
    else:
        draw_structure(screen, p1, ts, w, s.structure_color, s.structure_trim_color)


def draw_map(
    tilemap,
    visible_tiles,
    screen: Surface,
    settings,
):
    """Renders the Tilemap, accounting for FOV."""
    # Row is y; col is x
    for ty, row_data in enumerate(tilemap.tiles):
        for tx, tile in enumerate(row_data):
            if (tx, ty) in visible_tiles:
                draw_tile(screen, tile, settings)


def draw_player(screen: Surface, px: int, py: int, tile_size: int):
    """Renders the player (always visible) on the Tilemap."""
    ctrx = px * tile_size + 0.51 * tile_size
    ctry = py * tile_size + 0.51 * tile_size
    r = 0.33 * tile_size
    gold = (242, 215, 16)
    pygame.draw.circle(screen, gold, (ctrx, ctry), r)


def draw_enemy(screen: Surface, px: int, py: int, tile_size: int):
    """Renders an enemy unit (subject to LOS) on the Tilemap."""
    ctrx = px * tile_size + 0.51 * tile_size
    ctry = py * tile_size + 0.51 * tile_size
    r = 0.33 * tile_size
    red = (215, 30, 120)
    pygame.draw.circle(screen, red, (ctrx, ctry), r)


def draw_line_to_cursor(screen: Surface, px: int, py: int, mx: int, my: int, settings):
    """Draws a line from player to mouse cursor.

    `px, py`: int
        player tile position.
    `mx, my`: int
        mouse cursor position.
    """
    ts = settings.tile_size
    mid = settings.tile_size * 0.5
    line_width = settings.line_width

    pygame.draw.line(
        screen,
        Color("red"),
        (px * ts + mid, py * ts + mid),
        (mx, my),
        line_width,
    )


def draw_tile_at_cursor(screen: Surface, tx: int, ty: int, settings, line=True):
    """Draws border around Tile at cursor.  Also draws line if `line=True`."""
    ts = settings.tile_size
    tile_mid = ts * 0.5
    w = settings.line_width
    rx, ry = tx * ts, ty * ts

    if line:
        pygame.draw.line(
            screen,
            Color("red"),
            (tile_mid, tile_mid),
            (rx + tile_mid, ry + tile_mid),
            w,
        )

    pygame.draw.lines(
        screen,
        Color("yellow"),
        True,
        [(rx, ry), (rx + ts, ry), (rx + ts, ry + ts), (rx, ry + ts)],
    )


def draw_fov_line(screen: Surface, sx: int, sy: int, tx: int, ty: int, settings):
    """Draws a 2D FOV line from source (sx, sy) to tile at mouse cursor (tx, ty)."""
    tiles = bresenham_full(sx, sy, tx, ty)

    ts = settings.tile_size
    color = Color(settings.fov_line_color)
    for fx, fy in tiles:
        draw_fov_tile(screen, Vector2(fx * ts, fy * ts), ts, color)


def draw_line(
    screen: Surface, x1: float, y1: float, x2: float, y2: float, color: Color
):
    """Draws a 2D LOS line from source tile (x1, y1) to target (x2, y2)."""
    pygame.draw.line(screen, color, (x1, y1), (x2, y2))


def draw_fov_line_subpixel(
    screen: Surface, sx: int, sy: int, tx: int, ty: int, settings
):
    """Draws a subpixel 2D FOV line from (0,0) to tile at mouse cursor (tx, ty)."""
    sp = settings.subtiles_xy
    half_sp = sp // 2
    sts = settings.subtile_size
    # Origin / target pixels vary based on FOV height and octant
    ox = sx * sp + half_sp
    oy = sy * sp + half_sp
    fx = tx * sp + half_sp
    fy = ty * sp + half_sp

    if settings.fov_line_type == FovLineType.NORMAL:
        tiles = bresenham(ox, oy, fx, fy)
    else:
        tiles = bresenham_full(ox, oy, fx, fy)

    color = Color(settings.fov_line_color)

    for fx, fy in tiles:
        draw_fov_tile(screen, Vector2(fx * sts, fy * sts), sts, color)


def draw_fov_tile(screen: Surface, pr: Vector2, ts: int, color: Color):
    """Draws an FOV tile with reference point `pr` and tile size `ts`."""

    left = pr.x
    top = pr.y
    w = ts * 1.0
    h = ts * 1.0

    s = pygame.Surface((w, h))
    s.set_alpha(128)
    s.fill(color)
    screen.blit(s, (left, top))
