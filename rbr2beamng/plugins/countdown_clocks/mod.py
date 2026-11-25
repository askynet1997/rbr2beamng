from __future__ import annotations

import json
import string
from pathlib import Path

from ... import __version__
from ...conversion_common import install_companion_mod
from ...filesystem import FileSandbox


COUNTDOWN_CLOCKS_MOD_FILENAME = "rbr_countdown_clocks.zip"
START_CLOCKS_FILENAME = "rbr_start_clocks.json"
WARTHOG = "warthog"
TAG_HEUER = "tagHeuer"
ALGE = "alge"
ALGE_BANNER = "algeBanner"
CERTINA = "certina"
# Halved for the board backgrounds, which stay a multiple of 4 pixels: the game
# shears block compressed textures of other widths.
START_CLOCK_TEXTURE_SIZES = {
    WARTHOG: (1024, 128),
    TAG_HEUER: (1040, 704),
    ALGE: (1040, 648),
    ALGE_BANNER: (1040, 680),
    CERTINA: (1040, 840),
}
# Warthog display layout, in pixels: eight HH:MM:SS glyph cells, then the lamps.
START_CLOCK_DIGITS_RECT = (0, 0, 640, 128)
START_CLOCK_RED_RECT = (640, 0, 128, 128)
START_CLOCK_GREEN_RECT = (768, 0, 128, 128)
# Boards whose display is a region of their source texture: the reference texture
# size, and the region x0, y0, x1, y1 in its pixels, top-down.
START_CLOCK_REGIONS = {
    TAG_HEUER: ((256, 256), (0, 0, 210, 142)),
    ALGE: ((1024, 1024), (108, 7, 458, 226)),
    ALGE_BANNER: ((512, 256), (380, 82, 484, 150)),
    CERTINA: ((2048, 2048), (850, 5, 2048, 968)),
}
# The center of the Certina board lamp, in reference pixels.
CERTINA_LAMP_CENTER = (1946, 875)
_HELPER_MODULE = "lua/ge/extensions/gameplay/rally/rbrStartClocks"
_CLOCK_TYPES_DIR = "lua/ge/extensions/gameplay/rally/startClocks"


def start_clock_texture_tag(design: str, key: str) -> str:
    return f"@rbr{design[0].upper()}{design[1:]}Clock{key.capitalize()}"


def _mod_info() -> str:
    return json.dumps(
        {
            "title": "RBR Countdown Clocks",
            "author": "RBR2BeamNG",
            "tagid": "RBRCOUNTDOWNCLOCKS",
            "version_string": __version__,
            "category_title": "Gameplay",
            "tag_line": "Live start clocks for compatible RBR2BeamNG stages.",
            "message": (
                "Shows the rally wall clock, countdown and start lights on the RBR, "
                "TAG Heuer, ALGE-TIMING and Certina start clocks of compatible RBR2BeamNG stages."
            ),
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n"


_HELPER = string.Template("""-- Shared by the RBR2BeamNG clock types in startClocks/

local R = {}

local descriptorName = '$descriptor'
-- the stock clock view framing: camera distance per metre of board width
local viewDistancePerWidth = 2.04

R.digitSegments = {
  ['0'] = 'abcdef', ['1'] = 'bc', ['2'] = 'abdeg', ['3'] = 'abcdg', ['4'] = 'bcfg',
  ['5'] = 'acdfg', ['6'] = 'acdefg', ['7'] = 'abc', ['8'] = 'abcdefg', ['9'] = 'abcdfg',
}

-- display textures are sampled as linear, colors picked from RBR textures are sRGB
function R.srgb(r, g, b)
  local function linear(value)
    value = value / 255
    return 255 * (value <= 0.04045 and value / 12.92 or ((value + 0.055) / 1.055) ^ 2.4)
  end
  return color(linear(r), linear(g), linear(b), 255)
end

-- the clocks of one design in the descriptor the converter writes into the level, and the textures of that design
function R.findClocks(design)
  local levelId = getCurrentLevelIdentifier()
  local path = levelId and '/levels/' .. levelId .. '/' .. descriptorName
  if not path or not FS:fileExists(path) then return end
  local descriptor = jsonReadFile(path)
  local designData = descriptor and descriptor.designs and descriptor.designs[design]
  if not designData then return end
  local clocks = {}
  for _, clock in ipairs(descriptor.clocks or {}) do
    if clock.design == design then
      local transform = MatrixF(true)
      transform:setPosition(vec3(clock.position))
      table.insert(clocks, {shapeName = clock.shapeName, transform = transform, viewDirection = vec3(clock.normal), width = clock.width})
    end
  end
  return clocks, designData.textures
end

-- convex polygon {x1, y1, x2, y2, ...} moved by (dx, dy), as a fan of SDF triangles drawn with roundness round
function R.drawPolygon(target, dx, dy, points, fillColor, round)
  local ax, ay = dx + points[1], dy + points[2]
  for i = 3, #points - 3, 2 do
    target:triangle(ax, ay, dx + points[i], dy + points[i + 1], dx + points[i + 2], dy + points[i + 3], round, 0, fillColor, fillColor, fillColor, fillColor)
  end
end

-- unit direction of an analog face angle in turns, clockwise from 12 o'clock, y down
function R.faceDirection(turns)
  local angle = turns * 2 * math.pi
  return math.sin(angle), -math.cos(angle)
end

-- radial line of an analog face centered at (x, y), from radius r0 to r1
function R.faceLine(target, x, y, turns, r0, r1, halfWidthA, halfWidthB, round, strokeWidth, fillColor, strokeColor)
  local dx, dy = R.faceDirection(turns)
  target:line(x + dx * r0, y + dy * r0, x + dx * r1, y + dy * r1, halfWidthA, halfWidthB, round, strokeWidth,
    fillColor, fillColor, strokeColor, strokeColor)
end

-- seven-segment polygons a to g around the segment centerlines box (x0, y0)-(x1, y1): outer corners are cut by one
-- 45 degree diagonal gap, middle junctions are pointed. They are laid out round smaller than drawn, so the
-- triangles of drawPolygon overlap along their shared edges and leave no seams
function R.segmentPolygons(x0, y0, x1, y1, width, gap, round)
  local ym = (y0 + y1) / 2
  local h = width / 2 - round
  local s = gap / math.sqrt(2) + round * math.sqrt(2)
  return {
    a = {x0 - h + s, y0 - h, x1 + h - s, y0 - h, x1 - h - s, y0 + h, x0 + h + s, y0 + h},
    b = {x1 - h, y0 + h + s, x1 + h, y0 - h + s, x1 + h, ym - h - s, x1, ym - s, x1 - h, ym - h - s},
    c = {x1, ym + s, x1 + h, ym + h + s, x1 + h, y1 + h - s, x1 - h, y1 - h - s, x1 - h, ym + h + s},
    d = {x0 + h + s, y1 - h, x1 - h - s, y1 - h, x1 + h - s, y1 + h, x0 - h + s, y1 + h},
    e = {x0, ym + s, x0 + h, ym + h + s, x0 + h, y1 - h - s, x0 - h, y1 + h - s, x0 - h, ym + h + s},
    f = {x0 - h, y0 - h + s, x0 + h, y0 + h + s, x0 + h, ym - h - s, x0, ym - s, x0 - h, ym - h - s},
    g = {x0 + s, ym, x0 + h + s, ym - h, x1 - h - s, ym - h, x1 - s, ym, x1 - h - s, ym + h, x0 + h + s, ym + h},
  }
end

-- like the stock clock view: diagonal between the board face and the road, with the clock off-center so the
-- road beyond it, and the launching car, shows in the background; far enough to fit the whole board
function R.getView(clock, startLine)
  local distance = (clock.width or 1) * viewDistancePerWidth
  local normal = vec3(clock.viewDirection)
  normal.z = 0
  normal = normal:normalized()
  if not startLine.rot then return normal, distance end

  local along = quat(startLine.rot) * vec3(0, -1, 0)
  along.z = 0
  along = along:normalized()
  if along:dot(normal) < 0 then along = along * -1 end
  local direction = (normal + along):normalized()
  local viewRight = direction:cross(vec3(0, 0, -1))
  return direction, distance, viewRight:dot(along) > 0 and 1 / 3 or -1 / 3
end

return R
""")


_WARTHOG = string.Template("""-- Richard Burns Rally START_CLOCK boards of stages converted by RBR2BeamNG: HH:MM:SS digits, red and green lamps

local R = require('/$helper')
local C = {}

local digitsRect = {$digits_rect}
local lampRects = {red = {$red_rect}, green = {$green_rect}}
local cellWidth = digitsRect[3] / 8
-- RBR's digit and board sRGB colors, converted to linear
local litColor = color(120, 121, 1, 255)
local unlitColor = color(10, 10, 6, 255)
local boardColor = color(4, 4, 4, 255)
local segmentWidth = cellWidth * 0.19
local edgeRound = 1
local colonSize = segmentWidth * 1.1
local y0, ym, y1 = digitsRect[2] + digitsRect[4] * 0.1, digitsRect[2] + digitsRect[4] * 0.5, digitsRect[2] + digitsRect[4] * 0.9
local segments = R.segmentPolygons(cellWidth * 0.2, y0, cellWidth * 0.8, y1, segmentWidth, cellWidth * 0.05, edgeRound)
local colonDots = {}
for _, y in ipairs({ym - (ym - y0) * 0.45, ym + (y1 - ym) * 0.45}) do
  local cx, half = cellWidth / 2, colonSize / 2 - edgeRound
  table.insert(colonDots, {cx - half, y - half, cx + half, y - half, cx + half, y + half, cx - half, y + half})
end
-- RBR's lamp textures: a lit disc on a dark square, the lamp quads show their middle
local lampBackColor = R.srgb(37, 34, 37)
local lampColors = {red = R.srgb(247, 8, 0), green = R.srgb(74, 203, 45)}
local lampRadius = 0.38

-- color is the lit base look of the whole display, glow is emissive: the lit segments and the lit lamp only
C.textures = {
  color = {tag = '$color_tag', width = $width, height = $height, clearColor = boardColor},
  glow = {tag = '$glow_tag', width = $width, height = $height, clearColor = color(0, 0, 0, 255)},
}

function C.findClocks()
  return (R.findClocks('$design'))
end

function C.draw(textures, display)
  local litLamp = display.semaphore == 'G' and 'green' or 'red'
  for key, target in pairs(textures) do
    local glow = key == 'glow'
    for index = 1, #display.clock do
      local character = display.clock:sub(index, index)
      local cellX = digitsRect[1] + (index - 1) * cellWidth
      if character == ':' then
        for _, dot in ipairs(colonDots) do R.drawPolygon(target, cellX, 0, dot, litColor, edgeRound) end
      else
        local litSegments = R.digitSegments[character]
        for name, segment in pairs(segments) do
          local lit = litSegments:find(name, 1, true)
          if lit or not glow then R.drawPolygon(target, cellX, 0, segment, lit and litColor or unlitColor, edgeRound) end
        end
      end
    end
    for lamp, rect in pairs(lampRects) do
      local x, y, w, h = rect[1], rect[2], rect[3], rect[4]
      if not glow then
        R.drawPolygon(target, 0, 0, {x, y, x + w, y, x + w, y + h, x, y + h}, lampBackColor, 0)
      end
      if not glow or lamp == litLamp then
        local fill = lampColors[lamp]
        target:circle(x + w / 2, y + h / 2, w * lampRadius, 0, fill, fill, fill, fill)
      end
    end
  end
end

C.getView = R.getView

return C
""")


_TAG_HEUER = string.Template("""-- TAG Heuer start clock boards of RBR stages converted by RBR2BeamNG: analog wall clock, two digit countdown, red lamp

local R = require('/$helper')
local C = {}

local width, height = $width, $height
-- layout in pixels of the 256x256 source board texture, the display textures show its region
local region = {$region}
local scale = width / (region[3] - region[1])
local function px(x, y) return (x - region[1]) * scale, (y - region[2]) * height / (region[4] - region[2]) end

local black = color(0, 0, 0, 255)
local yellow = R.srgb(247, 255, 16)
local edgeColor = R.srgb(40, 40, 32)
local digitColor = R.srgb(214, 215, 74)
local ghostColor = R.srgb(48, 47, 44)
local lampColor = R.srgb(255, 112, 57)
local lampCoreColor = R.srgb(255, 190, 140)
local lampOffColor = R.srgb(70, 22, 12)
local faceX, faceY = px(61.75, 73)
-- covers the hexagonal lamp lens set into the board
local lampX, lampY = px(164.5, 60.4)
local lampRadius = 15 * scale
local edgeRound = 1
local digitSegments = {}
for index, cell in ipairs({{130, 83, 158, 126}, {168, 83, 196, 126}}) do
  local half = 2.5
  local x0, y0 = px(cell[1] + half, cell[2] + half)
  local x1, y1 = px(cell[3] - half, cell[4] - half)
  digitSegments[index] = R.segmentPolygons(x0, y0, x1, y1, half * 2 * scale, 0.9 * scale, edgeRound)
end
local background

-- color is the lit base look of the whole board, glow is emissive: the lit segments and the lit lamp only
C.textures = {
  color = {tag = '$color_tag', width = width, height = height, clearColor = black},
  glow = {tag = '$glow_tag', width = width, height = height, clearColor = black},
}

function C.findClocks()
  local clocks, textures = R.findClocks('$design')
  background = textures and textures.background
  return clocks
end

-- radial line from radius r0 to r1 of the face, in source pixels
local function faceLine(target, turns, r0, r1, halfWidthA, halfWidthB, fillColor, strokeWidth)
  R.faceLine(target, faceX, faceY, turns, r0 * scale, r1 * scale, halfWidthA * scale, halfWidthB * scale, 0.2 * scale,
    (strokeWidth or 0) * scale, fillColor, edgeColor)
end

local function drawFace(target)
  for minute = 0, 59 do
    local halfWidth = minute % 5 == 0 and 1.1 or 0.55
    faceLine(target, minute / 60, 51, 55, halfWidth, halfWidth, yellow)
  end
  for hour = 0, 11 do
    if hour % 3 == 0 then
      local dx, dy = R.faceDirection(hour / 12)
      local baseX, baseY = faceX + dx * 37 * scale, faceY + dy * 37 * scale
      local sideX, sideY = -dy * 3.6 * scale, dx * 3.6 * scale
      target:triangle(faceX + dx * 46 * scale, faceY + dy * 46 * scale, baseX + sideX, baseY + sideY, baseX - sideX, baseY - sideY,
        0.3 * scale, 0, yellow, yellow, yellow, yellow)
    else
      faceLine(target, hour / 12, 37.5, 45.5, 1.6, 1.6, yellow)
    end
  end
end

function C.draw(textures, display)
  local seconds = display.clockSeconds
  local countdown = display.countdownSeconds or (60 - seconds % 60) % 60
  local digits = string.format('%02d', math.min(99, countdown))
  local lampLit = display.semaphore ~= 'G'
  for key, target in pairs(textures) do
    local glow = key == 'glow'
    if not glow then
      if background then
        target:queueBlitTextureFile(background.path, 0, 0, background.width, background.height, 0, 0, width, height, true)
      end
      drawFace(target)
      faceLine(target, seconds / 43200 % 1, -6, 28, 2.4, 1.3, yellow, 0.6)
      faceLine(target, seconds / 3600 % 1, -8, 46, 1.9, 1, yellow, 0.6)
      faceLine(target, seconds % 60 / 60, -12, 52, 0.6, 0.45, yellow, 0.5)
      target:circle(faceX, faceY, 3.2 * scale, 0.6 * scale, yellow, yellow, edgeColor, edgeColor)
    end
    for index, segments in ipairs(digitSegments) do
      local litSegments = R.digitSegments[digits:sub(index, index)]
      for name, segment in pairs(segments) do
        local lit = litSegments:find(name, 1, true)
        if lit or not glow then R.drawPolygon(target, 0, 0, segment, lit and digitColor or ghostColor, edgeRound) end
      end
    end
    if lampLit then
      target:circle(lampX, lampY, lampRadius, 0, lampCoreColor, lampColor, lampColor, lampColor)
    elseif not glow then
      target:circle(lampX, lampY, lampRadius, 0, lampOffColor, lampOffColor, lampOffColor, lampOffColor)
    end
  end
end

C.getView = R.getView

return C
""")


_ALGE = string.Template("""-- ALGE-TIMING start clock boards of RBR stages converted by RBR2BeamNG: LED wall clock, M:SS countdown and a
-- red, amber and green start light

local R = require('/$helper')
local C = {}

local width, height = $width, $height
-- layout in pixels of the 1024x1024 source atlas, the display textures show its region
local region = {$region}
local scaleX, scaleY = width / (region[3] - region[1]), height / (region[4] - region[2])
local function px(x, y) return (x - region[1]) * scaleX, (y - region[2]) * scaleY end

local black = color(0, 0, 0, 255)
local clockColor, clockGhost = R.srgb(150, 245, 185), R.srgb(62, 98, 82)
local countdownColor, countdownGhost = R.srgb(255, 120, 110), R.srgb(105, 58, 58)
local clockDotRadius, countdownDotRadius, lampDotRadius = 2 * scaleX, 2.3 * scaleX, 2.5 * scaleX

-- dot centers of a dot-matrix seven-segment digit, the segments leave the corners out: x0 and x1 are the left and
-- right columns of the bottom row, the top row leans slant pixels right
local function digitDots(x0, x1, y0, y1, horizontalDots, verticalDots, slant)
  local columns, rows = horizontalDots + 1, 2 * verticalDots + 2
  local dots = {a = {}, b = {}, c = {}, d = {}, e = {}, f = {}, g = {}}
  local function add(segment, column, row)
    table.insert(dots[segment], {px(x0 + (x1 - x0) * column / columns + slant * (1 - row / rows), y0 + (y1 - y0) * row / rows)})
  end
  for column = 1, horizontalDots do
    add('a', column, 0)
    add('g', column, verticalDots + 1)
    add('d', column, rows)
  end
  for row = 1, verticalDots do
    add('f', 0, row)
    add('b', columns, row)
    add('e', 0, verticalDots + 1 + row)
    add('c', columns, verticalDots + 1 + row)
  end
  return dots
end

local function points(list)
  local result = {}
  for _, point in ipairs(list) do table.insert(result, {px(point[1], point[2])}) end
  return result
end

local clockDigits, countdownDigits = {}, {}
for _, x in ipairs({163, 200, 252, 289, 341, 378}) do table.insert(clockDigits, digitDots(x, x + 25, 44, 83, 4, 3, 4)) end
for _, x in ipairs({256, 321, 371}) do table.insert(countdownDigits, digitDots(x, x + 32, 115, 167, 5, 4, 2.5)) end
local clockColons = points({{240, 52}, {238, 74}, {329, 52}, {327, 74}})
local countdownColon = points({{305, 130}, {302.5, 154}})
-- start light clusters: a ring of eight dots around a center dot
local lamps = {}
for code, lamp in pairs({
  R = {199, 110.5, R.srgb(255, 45, 35), R.srgb(110, 40, 38)},
  A = {199.5, 140.5, R.srgb(255, 160, 40), R.srgb(120, 86, 52)},
  G = {199.5, 171.5, R.srgb(50, 255, 100), R.srgb(34, 92, 52)},
}) do
  local cluster = {{lamp[1], lamp[2]}}
  for index = 0, 7 do
    local angle = index / 8 * 2 * math.pi
    table.insert(cluster, {lamp[1] + math.sin(angle) * 9, lamp[2] - math.cos(angle) * 9})
  end
  lamps[code] = {dots = points(cluster), lit = lamp[3], ghost = lamp[4]}
end
local background

-- color is the lit base look of the whole board, glow is emissive: the lit LEDs only
C.textures = {
  color = {tag = '$color_tag', width = width, height = height, clearColor = black},
  glow = {tag = '$glow_tag', width = width, height = height, clearColor = black},
}

function C.findClocks()
  local clocks, textures = R.findClocks('$design')
  background = textures and textures.background
  return clocks
end

local function drawDots(target, dots, radius, fillColor)
  for _, dot in ipairs(dots) do target:circle(dot[1], dot[2], radius, 0, fillColor, fillColor, fillColor, fillColor) end
end

local function drawDigit(target, dots, character, radius, litColor, ghostColor, glow)
  local litSegments = R.digitSegments[character] or ''
  for name, segmentDots in pairs(dots) do
    local lit = litSegments:find(name, 1, true)
    if lit or not glow then drawDots(target, segmentDots, radius, lit and litColor or ghostColor) end
  end
end

function C.draw(textures, display)
  local clockText = display.clock:gsub(':', '')
  local countdown = display.countdownSeconds and math.min(599, display.countdownSeconds)
  local countdownText = countdown and string.format('%d%02d', math.floor(countdown / 60), countdown % 60) or ''
  for key, target in pairs(textures) do
    local glow = key == 'glow'
    if not glow and background then
      target:queueBlitTextureFile(background.path, 0, 0, background.width, background.height, 0, 0, width, height, true)
    end
    for index, dots in ipairs(clockDigits) do
      drawDigit(target, dots, clockText:sub(index, index), clockDotRadius, clockColor, clockGhost, glow)
    end
    drawDots(target, clockColons, clockDotRadius, clockColor)
    for index, dots in ipairs(countdownDigits) do
      drawDigit(target, dots, countdownText:sub(index, index), countdownDotRadius, countdownColor, countdownGhost, glow)
    end
    if countdown or not glow then
      drawDots(target, countdownColon, countdownDotRadius, countdown and countdownColor or countdownGhost)
    end
    for code, lamp in pairs(lamps) do
      local lit = display.semaphore == code
      if lit or not glow then drawDots(target, lamp.dots, lampDotRadius, lit and lamp.lit or lamp.ghost) end
    end
  end
end

C.getView = R.getView

return C
""")


_ALGE_BANNER = string.Template("""-- ALGE-TIMING display of the Karlstad start banners converted by RBR2BeamNG: stage label and countdown seconds
-- above, time of day below

local R = require('/$helper')
local C = {}

local width, height = $width, $height
-- layout in pixels of the 512x256 source banner texture, the display textures show its region
local region = {$region}
local scaleX, scaleY = width / (region[3] - region[1]), height / (region[4] - region[2])
local function px(x, y) return (x - region[1]) * scaleX, (y - region[2]) * scaleY end

local black = color(0, 0, 0, 255)
local litColor = R.srgb(255, 90, 40)
local rows = {
  {top = 90, bottom = 107.5, ghost = R.srgb(90, 98, 110)},
  {top = 123, bottom = 140.5, ghost = R.srgb(40, 32, 28)},
}
local edgeRound = 1
for _, row in ipairs(rows) do
  row.cells = {}
  for _, x in ipairs({388.5, 404.3, 420.1, 435.9, 451.7, 467.5}) do
    local half = 1.2
    local x0, y0 = px(x + half, row.top + half)
    local x1, y1 = px(x + 11.5 - half, row.bottom - half)
    table.insert(row.cells, R.segmentPolygons(x0, y0, x1, y1, half * 2 * scaleX, 0.4 * scaleX, edgeRound))
  end
end
local background

-- color is the lit base look of the whole display, glow is emissive: the lit segments only
C.textures = {
  color = {tag = '$color_tag', width = width, height = height, clearColor = black},
  glow = {tag = '$glow_tag', width = width, height = height, clearColor = black},
}

function C.findClocks()
  local clocks, textures = R.findClocks('$design')
  background = textures and textures.background
  return clocks
end

function C.draw(textures, display)
  -- the stage label in the left four cells, the countdown seconds in the right two
  local label = display.label:sub(1, 4)
  local countdown = display.countdownSeconds and string.format('%02d', math.min(99, display.countdownSeconds)) or ''
  local texts = {
    label .. string.rep(' ', 4 - #label) .. countdown,
    (display.clock:gsub(':', '')),
  }
  for key, target in pairs(textures) do
    local glow = key == 'glow'
    if not glow and background then
      target:queueBlitTextureFile(background.path, 0, 0, background.width, background.height, 0, 0, width, height, true)
    end
    for rowIndex, row in ipairs(rows) do
      for index, segments in ipairs(row.cells) do
        -- a seven-segment S is a 5
        local character = texts[rowIndex]:sub(index, index):gsub('S', '5')
        local litSegments = R.digitSegments[character] or ''
        for name, segment in pairs(segments) do
          local lit = litSegments:find(name, 1, true)
          if lit or not glow then R.drawPolygon(target, 0, 0, segment, lit and litColor or row.ghost, edgeRound) end
        end
      end
    end
  end
end

C.getView = R.getView

return C
""")


_CERTINA = string.Template("""-- Certina start clock boards of RBR stages converted by RBR2BeamNG: analog wall clock with a ring of second LEDs,
-- red or green LED start lamp

local R = require('/$helper')
local C = {}

local width, height = $width, $height
-- layout in pixels of the 2048x2048 source atlas, the display textures show its region
local region = {$region}
local scale = width / (region[3] - region[1])
local function px(x, y) return (x - region[1]) * scale, (y - region[2]) * height / (region[4] - region[2]) end

local black = color(0, 0, 0, 255)
local markColor = R.srgb(112, 112, 110)
local handColor = R.srgb(130, 130, 132)
local edgeColor = R.srgb(45, 45, 45)
local dotColor = R.srgb(105, 104, 105)
local litDotColor = R.srgb(40, 255, 110)
local litDotCoreColor = R.srgb(215, 255, 230)
local faceX, faceY = px(1193, 394)
local lampX, lampY = px($lamp_center)
-- the lamp LEDs: rings of 6, 12, 18 and 24 around a dark center, on the haze of their light
local lampLeds = {}
for ring = 1, 4 do
  for index = 0, ring * 6 - 1 do
    local x, y = R.faceDirection((index + ring % 2 * 0.5) / (ring * 6))
    table.insert(lampLeds, {lampX + x * ring * 14.3 * scale, lampY + y * ring * 14.3 * scale})
  end
end
local lampColors = {
  red = {haze = R.srgb(90, 6, 6), led = R.srgb(255, 40, 40), core = R.srgb(255, 165, 160)},
  green = {haze = R.srgb(6, 80, 24), led = litDotColor, core = litDotCoreColor},
}
local background

-- color is the lit base look of the whole board, glow is emissive: the lit second LEDs and the lamp only
C.textures = {
  color = {tag = '$color_tag', width = width, height = height, clearColor = black},
  glow = {tag = '$glow_tag', width = width, height = height, clearColor = black},
}

function C.findClocks()
  local clocks, textures = R.findClocks('$design')
  background = textures and textures.background
  return clocks
end

-- radial line from radius r0 to r1 of the face, in source pixels
local function faceLine(target, turns, r0, r1, halfWidthA, halfWidthB, fillColor, strokeWidth)
  R.faceLine(target, faceX, faceY, turns, r0 * scale, r1 * scale, halfWidthA * scale, halfWidthB * scale, 0.5 * scale,
    (strokeWidth or 0) * scale, fillColor, edgeColor)
end

function C.draw(textures, display)
  local seconds = display.clockSeconds
  local lamp = lampColors[display.semaphore == 'G' and 'green' or 'red']
  for key, target in pairs(textures) do
    local glow = key == 'glow'
    if not glow then
      if background then
        target:queueBlitTextureFile(background.path, 0, 0, background.width, background.height, 0, 0, width, height, true)
      end
      for hour = 0, 11 do faceLine(target, hour / 12, 222, 289, 6, 10.5, markColor) end
      faceLine(target, seconds / 43200 % 1, -24, 215, 11, 8, handColor, 1)
      faceLine(target, seconds / 3600 % 1, -24, 285, 14, 10, handColor, 1)
      faceLine(target, seconds % 60 / 60, -60, 296, 4, 2.5, handColor, 1)
      target:circle(faceX, faceY, 20 * scale, 1.5 * scale, handColor, handColor, edgeColor, edgeColor)
    end
    -- one LED per second, filling up from 12 o'clock
    for second = 0, 59 do
      local x, y = R.faceDirection(second / 60)
      x, y = faceX + x * 306 * scale, faceY + y * 306 * scale
      if second <= seconds % 60 then
        target:circle(x, y, 7 * scale, 0, litDotCoreColor, litDotColor, litDotColor, litDotColor)
      elseif not glow then
        target:circle(x, y, 3.5 * scale, 0, dotColor, dotColor, dotColor, dotColor)
      end
    end
    target:circle(lampX, lampY, 64 * scale, 0, lamp.haze, lamp.haze, lamp.haze, lamp.haze)
    for _, led in ipairs(lampLeds) do
      target:circle(led[1], led[2], 6.3 * scale, 0, lamp.core, lamp.led, lamp.led, lamp.led)
    end
  end
end

C.getView = R.getView

return C
""")


def _numbers(values) -> str:
    return ", ".join(f"{value:g}" for value in values)


def _clock_type(design: str, template: string.Template, **values) -> str:
    width, height = START_CLOCK_TEXTURE_SIZES[design]
    return template.substitute(
        helper=_HELPER_MODULE,
        design=design,
        color_tag=start_clock_texture_tag(design, "color"),
        glow_tag=start_clock_texture_tag(design, "glow"),
        width=width,
        height=height,
        **values,
    )


def _mod_files() -> dict[str, str]:
    return {
        "mod_info/RBRCOUNTDOWNCLOCKS/info.json": _mod_info(),
        f"{_HELPER_MODULE}.lua": _HELPER.substitute(descriptor=START_CLOCKS_FILENAME),
        f"{_CLOCK_TYPES_DIR}/rbrWarthogClock.lua": _clock_type(
            WARTHOG,
            _WARTHOG,
            digits_rect=_numbers(START_CLOCK_DIGITS_RECT),
            red_rect=_numbers(START_CLOCK_RED_RECT),
            green_rect=_numbers(START_CLOCK_GREEN_RECT),
        ),
        f"{_CLOCK_TYPES_DIR}/rbrTagHeuerClock.lua": _clock_type(
            TAG_HEUER,
            _TAG_HEUER,
            region=_numbers(START_CLOCK_REGIONS[TAG_HEUER][1]),
        ),
        f"{_CLOCK_TYPES_DIR}/rbrAlgeClock.lua": _clock_type(
            ALGE,
            _ALGE,
            region=_numbers(START_CLOCK_REGIONS[ALGE][1]),
        ),
        f"{_CLOCK_TYPES_DIR}/rbrAlgeBannerClock.lua": _clock_type(
            ALGE_BANNER,
            _ALGE_BANNER,
            region=_numbers(START_CLOCK_REGIONS[ALGE_BANNER][1]),
        ),
        f"{_CLOCK_TYPES_DIR}/rbrCertinaClock.lua": _clock_type(
            CERTINA,
            _CERTINA,
            region=_numbers(START_CLOCK_REGIONS[CERTINA][1]),
            lamp_center=_numbers(CERTINA_LAMP_CENTER),
        ),
    }


def install_countdown_clocks_mod(destination: Path, filesystem: FileSandbox) -> None:
    """Build and atomically replace the countdown clocks mod archive."""
    install_companion_mod(destination, _mod_files(), filesystem)
