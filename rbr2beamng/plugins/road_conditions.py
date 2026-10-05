"""The RBR Road Conditions mod, which switches Original stages between RBR's
road conditions in BeamNG."""

from __future__ import annotations

import json
from pathlib import Path

from .. import __version__
from ..conversion_common import install_companion_mod
from ..filesystem import FileSandbox
from . import CompanionMod, PluginLevel


CONDITION_SELECTOR_MOD_FILENAME = "rbr_road_conditions_selector.zip"
_APP_NAME = "RbrRoadConditions"
_APP_DIRECTIVE = "rbrRoadConditions"
_APP_DOM_ELEMENT = "<rbr-mat-conditions></rbr-mat-conditions>"
_APP_CSS = {
    "width": "312px",
    "height": "80px",
    "bottom": "10px",
    "left": "10px",
}


def _mod_info() -> str:
    return json.dumps(
        {
            "title": "RBR Road Condition Selector",
            "author": "RBR2BeamNG",
            "tagid": "RBRROADCONDITIONS",
            "version_string": __version__,
            "category_title": "UI Apps",
            "tag_line": "Select road conditions for compatible RBR2BeamNG stages.",
            "message": "Select road conditions for compatible RBR2BeamNG stages.",
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n"


_APP_INFO = json.dumps(
    {
        "name": "RBR Road Condition",
        "author": "RBR2BeamNG",
        "version": __version__,
        "description": "Select the converted RBR MAT road condition.",
        "directive": _APP_DIRECTIVE,
        "domElement": _APP_DOM_ELEMENT,
        "css": _APP_CSS,
        "preserveAspectRatio": False,
        "vue": True,
        "interactive": "required",
    },
    ensure_ascii=False,
    indent=2,
) + "\n"


_APP_VUE = """<template>
  <div class="rbr-mat-conditions" :style="{ opacity }">
    <span class="row-label">Surface:</span>
    <button v-for="surface in surfaces" :key="surface" :class="{ selected: selected(surface, selectedWear) }" :disabled="!available(surface, selectedWear) || !state.canChange" @click="select(surface, selectedWear)">
      {{ label(surface) }}
    </button>
    <span class="row-label">Wear:</span>
    <button v-for="wear in wears" :key="wear" :class="{ selected: selected(selectedSurface, wear) }" :disabled="!available(selectedSurface, wear) || !state.canChange" @click="select(selectedSurface, wear)">
      {{ label(wear) }}
    </button>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from "vue"
import { useBridge } from "@/bridge"
import { useEvents } from "@/services/events"

const { api } = useBridge()
const events = useEvents()
const surfaces = ["dry", "damp", "wet"]
const wears = ["new", "normal", "worn"]
const state = ref({})
const label = value => value.charAt(0).toUpperCase() + value.slice(1)
const available = (surface, wear) => !!state.value.conditions?.[surface + "/" + wear]
const selected = (surface, wear) => state.value.selected?.surface === surface && state.value.selected?.wear === wear
const selectedSurface = computed(() => state.value.selected?.surface || "dry")
const selectedWear = computed(() => state.value.selected?.wear || "new")
const opacity = computed(() => {
  const speed = Number(state.value.speedKph) || 0
  return speed <= 1 ? 1 : Math.max(0, 1 - (speed - 1) / 4)
})
const select = (surface, wear) => {
  if (available(surface, wear) && state.value.canChange) {
    api.engineLua(`rbrRoadConditions.selectCondition(${JSON.stringify(surface)}, ${JSON.stringify(wear)})`)
  }
}

events.on("rbrRoadConditionsUpdate", value => {
  state.value = value || {}
})
onMounted(() => {
  api.engineLua("rbrRoadConditions.getConditionState()")
})
</script>

<style scoped>
.rbr-mat-conditions {
  display: grid;
  grid-template-columns: 4.5rem repeat(3, 4.5rem);
  grid-template-rows: repeat(2, 2rem);
  align-items: stretch;
  place-content: center;
  gap: .25rem;
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  padding: .375rem;
  color: #d8dade;
  background: #2d3440;
  border: .0625rem solid #424a56;
  border-radius: .125rem;
  box-shadow: 0 .125rem .35rem rgba(0, 0, 0, .35);
  transition: opacity .2s linear;
}
.row-label {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: .25rem;
  text-align: right;
}
button {
  box-sizing: border-box;
  height: 2rem;
  padding: .25rem .5rem;
  border: .125rem solid transparent;
  border-radius: .25rem;
  color: #d8dade;
  font: inherit;
  background: #3d4652;
  box-shadow: inset 0 .0625rem #535d69;
  cursor: pointer;
}
button:hover:not(:disabled) { background: #4a5562; }
button.selected {
  border-color: #ff7a00;
  color: #ff7a00;
}
button:disabled { opacity: .4; cursor: default; }
</style>
"""


_EXTENSION = """local M = {}
M.dependencies = {"ui_appLayouts"}

local conditionData
local selectedCondition
local stateUpdateTimer = 0
local selectorAdded = false
local selectorSpawnRequested = false
local activeLevelId
local lastDiagnostic
local wasLoading = false
local loadingFinishedTimer
local selectorAppName = "rbrRoadConditions"

local function diagnostic(message)
  if message == lastDiagnostic then return end
  lastDiagnostic = message
  log("I", "rbrRoadConditions", message)
end

local function playerSpeedKph()
  local vehicle = be:getPlayerVehicle(0)
  if not vehicle then return 0 end
  local velocity = vehicle:getVelocity()
  return velocity:length() * 3.6
end

local function sendConditionState()
  guihooks.trigger("rbrRoadConditionsUpdate", {
    title = "Road Condition",
    conditions = conditionData and conditionData.conditions or {},
    selected = selectedCondition,
    speedKph = playerSpeedKph(),
    canChange = conditionData ~= nil and playerSpeedKph() <= 1,
  })
end

local function removeSelectorApp()
  local layouts = extensions.ui_appLayouts
  local layout = layouts and layouts.getCurrentLayout and layouts.getCurrentLayout()
  local removed = false
  if layout and layout.apps then
    for index = #layout.apps, 1, -1 do
      if layout.apps[index].appName == selectorAppName then
        if layout.filename and layouts.removeApp then
          layouts.removeApp(layout.filename, index - 1)
        else
          table.remove(layout.apps, index)
        end
        removed = true
      end
    end
    if removed then
      if layout.filename and layouts.removeApp then
        extensions.ui_apps.notifyLayoutsChanged()
      else
        layouts.setCurrentLayout(layout)
      end
    end
  end
  selectorAdded = false
  selectorSpawnRequested = false
end

local function selectorInLayout()
  local layouts = extensions.ui_appLayouts
  local layout = layouts and layouts.getCurrentLayout and layouts.getCurrentLayout()
  if not layout or not layout.apps then return false end
  for _, app in ipairs(layout.apps) do
    if app.appName == selectorAppName then
      selectorAdded = true
      selectorSpawnRequested = false
      return true
    end
  end
  return false
end

local function addSelectorApp()
  if selectorInLayout() then
    diagnostic("selector already in layout")
    return
  end
  if selectorSpawnRequested then return end
  selectorSpawnRequested = true
  guihooks.trigger("appContainer:spawn", {appName = selectorAppName})
  diagnostic("selector spawn requested")
end

local function loadConditionData()
  local levelId = getCurrentLevelIdentifier()
  if type(levelId) ~= "string" or levelId == "" then return nil end
  local data = jsonReadFile("/levels/" .. levelId .. "/rbr_road_conditions.json")
  if type(data) ~= "table" or type(data.conditions) ~= "table" then return nil end
  return data
end

local function getConditionState()
  sendConditionState()
end

local function selectCondition(surface, wear)
  if type(surface) ~= "string" or type(wear) ~= "string" or playerSpeedKph() > 1 then
    return
  end
  local profiles = conditionData and conditionData.conditions and conditionData.conditions[surface .. "/" .. wear]
  if type(profiles) ~= "table" then return end
  for profileName, groundType in pairs(profiles) do
    local material = scenetree.findObject(profileName)
    if material and type(groundType) == "string" then
      material.groundType = groundType
    end
  end
  for _, waterName in ipairs(conditionData.wetOnlyWater or {}) do
    local water = scenetree.findObject(waterName)
    if water then water:setHidden(surface ~= "wet") end
  end
  be:reloadCollision()
  selectedCondition = {surface = surface, wear = wear}
  sendConditionState()
end

local function onClientStartMission()
  removeSelectorApp()
  activeLevelId = getCurrentLevelIdentifier()
  conditionData = loadConditionData()
  diagnostic("mission check level=" .. tostring(activeLevelId) .. " metadata=" .. tostring(conditionData ~= nil))
  local defaultKey = conditionData and type(conditionData.default) == "string" and conditionData.default or "dry/new"
  local defaultSurface, defaultWear = string.match(defaultKey, "^([^/]+)/([^/]+)$")
  selectedCondition = (
    conditionData and defaultSurface and type(conditionData.conditions[defaultKey]) == "table"
    and {surface = defaultSurface, wear = defaultWear}
    or nil
  )
  stateUpdateTimer = 0
  sendConditionState()
end

local function onClientEndMission()
  removeSelectorApp()
  conditionData = nil
  selectedCondition = nil
  activeLevelId = nil
end

local function onExtensionLoaded()
  diagnostic("extension loaded")
  onClientStartMission()
  if conditionData then addSelectorApp() end
end

local function onLoadingFinished()
  diagnostic("loading finished")
  selectorSpawnRequested = false
  if not conditionData then
    activeLevelId = getCurrentLevelIdentifier()
    conditionData = loadConditionData()
  end
  if conditionData then
    addSelectorApp()
  else
    removeSelectorApp()
  end
end

local function onUiReady()
  selectorSpawnRequested = false
  if conditionData then
    addSelectorApp()
  else
    removeSelectorApp()
  end
end

local function onUpdate(dtReal)
  local currentLevelId = getCurrentLevelIdentifier()
  local hasConditionData = type(currentLevelId) == "string"
    and FS:fileExists("/levels/" .. currentLevelId .. "/rbr_road_conditions.json")
  if currentLevelId ~= activeLevelId
    or (hasConditionData and not conditionData) then
    onClientStartMission()
  end
  -- BeamNG 0.39 has no onLoadingScreenInactive hook; the loading screen fades out for 2 s
  local loading = core_gamestate.loading()
  if loading then
    loadingFinishedTimer = nil
  elseif wasLoading then
    loadingFinishedTimer = 2
  end
  wasLoading = loading
  if loadingFinishedTimer then
    loadingFinishedTimer = loadingFinishedTimer - dtReal
    if loadingFinishedTimer <= 0 then
      loadingFinishedTimer = nil
      onLoadingFinished()
    end
  end
  if not conditionData then return end
  stateUpdateTimer = stateUpdateTimer + dtReal
  if stateUpdateTimer >= 0.1 then
    stateUpdateTimer = 0
    sendConditionState()
  end
end

M.getConditionState = getConditionState
M.selectCondition = selectCondition
M.onClientStartMission = onClientStartMission
M.onClientPostStartMission = onClientStartMission
M.onClientEndMission = onClientEndMission
M.onExtensionLoaded = onExtensionLoaded
M.onUiReady = onUiReady
M.onUpdate = onUpdate
M.onExtensionUnloaded = onClientEndMission

return M
"""


_MOD_SCRIPT = """setExtensionUnloadMode("rbrRoadConditions", "manual")
extensions.load("rbrRoadConditions")
"""


def install_condition_selector_mod(destination: Path, filesystem: FileSandbox) -> None:
    """Build and atomically replace the companion selector mod archive."""
    install_companion_mod(
        destination,
        {
            "mod_info/RBRROADCONDITIONS/info.json": _mod_info(),
            f"ui/modules/apps/{_APP_NAME}/app.json": _APP_INFO,
            f"ui/modules/apps/{_APP_NAME}/app.vue": _APP_VUE,
            "lua/ge/extensions/rbrRoadConditions.lua": _EXTENSION,
            "scripts/rbrRoadConditions/modScript.lua": _MOD_SCRIPT,
        },
        filesystem,
    )


COMPANION_MOD = CompanionMod(
    "Road Conditions",
    CONDITION_SELECTOR_MOD_FILENAME,
    install_condition_selector_mod,
)


def write_level(level: PluginLevel) -> dict[str, int]:
    prepared = level.prepared
    if prepared is None or not prepared.condition_ground_types:
        return {}
    from ..beamng import write_json

    wet_only_water = [water.name for water in level.waters if water.wet_only]
    write_json(
        level.level_dir / "rbr_road_conditions.json",
        {
            "default": "/".join(prepared.road_condition),
            "conditions": {
                f"{surface}/{wear}": profiles
                for (surface, wear), profiles in prepared.condition_ground_types.items()
            },
            **({"wetOnlyWater": wet_only_water} if wet_only_water else {}),
        },
    )
    return {}
