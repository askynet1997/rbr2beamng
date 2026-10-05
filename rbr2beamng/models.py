from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .surface_profiles import SurfaceProfile


DEFAULT_FOLIAGE_NAME_MATCHES = (
    "tree",
    "trees",
    "leaves",
    "vegetation",
    "bush",
    "grass",
    "plant",
    "palm",
    "flower",
    "objecttextureplatetransparent",
)


DEFAULT_FOLIAGE_GROUND_TYPES = (
    "GRASS",
    "LEAVES_THIN",
    "LEAVES_STRONG",
    "BRANCHES_STRONG",
)


DEFAULT_SNOWBANK_NAME_MATCHES = ("snovall", "snowbottom")


DEFAULT_SNOWBANK_NAME_MESH_PATTERNS = (r"wall\d+",)


DEFAULT_WATER_NAME_MATCHES = (
    "water",
    "voda",
    "woda",
    "!plant",
    "!vodafone",
    "!truck",
)


@dataclass(frozen=True)
class StageMetadata:
    folder_name: str
    name: str
    author: str
    physics: str
    version: str
    date: str
    comment: str
    length_km: float | None
    splashscreen: Path | None
    surface_composition: tuple[tuple[str, float], ...] = ()
    author_website: str = ""
    author_note: str = ""

    @property
    def surface_composition_text(self) -> str:
        if not self.surface_composition:
            return self.physics.title()
        nonzero = [
            (name, amount)
            for name, amount in self.surface_composition
            if amount > 0
        ]
        if not nonzero:
            return "Unknown"
        if len(nonzero) == 1 and nonzero[0][1] == 100:
            return nonzero[0][0].title()
        return ", ".join(
            f"{name.title()} {amount:g}%"
            for name, amount in nonzero
        )


@dataclass(frozen=True)
class StageInspection:
    root: Path
    metadata: StageMetadata
    valid: bool
    issues: tuple[str, ...] = ()
    location: StageLocation | None = None
    source_format: str = "rx"
    source_key: str = ""
    variants: tuple[str, ...] = ()
    documents: tuple[StageDocument, ...] = ()


@dataclass(frozen=True)
class StageObject:
    index: int
    mesh_name: str
    mesh_path: Path
    transform_path: Path
    clone_count: int
    lod_in: float
    lod_out: float
    visible: bool
    draw_instanced: bool
    moveable: bool
    shadow_caster: bool | None
    collision_model: int
    collision_box: tuple[float, ...] = ()


@dataclass(frozen=True)
class RbrMaterial:
    index: int
    name: str
    effect: str
    technique: str
    diffuse_texture: Path | None
    second_diffuse_texture: Path | None
    normal_texture: Path | None
    specular_texture: Path | None
    properties: dict[str, str]
    double_sided: bool = False
    multiplier_texture: Path | None = None
    additive_texture: Path | None = None
    # Source renderer alpha passes. ``cutout_alpha_ref`` alpha-tests the
    # depth-writing pass (alpha >= ref); ``blend_alpha_ref`` adds a blended pass
    # without depth writes (0 = no alpha test). ``None`` omits the test/pass.
    cutout_alpha_ref: int | None = None
    blend_alpha_ref: int | None = None

    @property
    def uses_alpha(self) -> bool:
        return self.cutout_alpha_ref is not None or self.blend_alpha_ref is not None


OpacityKind = Literal["binary", "smooth"]


@dataclass(frozen=True)
class OpacityTexture:
    name: str
    kind: OpacityKind
    transparent_coverage: float


@dataclass(frozen=True)
class WaterAppearance:
    baseColor: list[int] | None = None
    underwaterColor: list[int] | None = None
    clarity: float | None = None
    overallFoamOpacity: float | None = None
    overallRippleMagnitude: float | None = None
    overallWaveMagnitude: float | None = None
    reflectivity: float | None = None


@dataclass(frozen=True)
class PbrMaterialOverride:
    base_color_texture: Path
    base_color_uv: int = 0
    base_vertex_color: bool = False
    normal_texture: Path | None = None
    normal_uv: int = 0
    roughness_texture: Path | None = None
    roughness_uv: int = 0
    clear_coat_texture: Path | None = None
    clear_coat_uv: int = 0
    clear_coat_roughness: float | None = None
    layer_color_texture: Path | None = None
    layer_color_uv: int = 1
    layer_opacity_texture: Path | None = None
    layer_opacity_uv: int = 1
    layer_vertex_color: bool = False
    material_version: float = 1.5

    @classmethod
    def vertex_color_lerp(
        cls,
        base_color_texture: Path,
        layer_color_texture: Path,
    ) -> PbrMaterialOverride:
        return cls(
            base_color_texture=base_color_texture,
            base_color_uv=0,
            base_vertex_color=True,
            layer_color_texture=layer_color_texture,
            layer_color_uv=1,
            layer_vertex_color=True,
            material_version=1.0,
        )


@dataclass(frozen=True)
class MaterialVariant:
    material: RbrMaterial | None
    ground_type: str
    hard: bool
    water: bool
    bendable: bool
    source_surface_ids: tuple[int, ...]
    ground_depth: float = 0.0
    snowbank: bool = False
    pbr_override: PbrMaterialOverride | None = None
    base_vertex_color: bool = False
    classification_fallback: str | None = None
    opacity_depth_prepass: bool = False


@dataclass(frozen=True)
class SurfaceMap:
    texture: Path
    cells: tuple[tuple[int, ...], ...]

    def sample(self, u: float, v: float) -> int:
        x = min(15, max(0, int((u % 1.0) * 16.0)))
        y = min(15, max(0, int((1.0 - (v % 1.0)) * 16.0)))
        return self.cells[y][x]


@dataclass(frozen=True)
class RbrSurface:
    surface_id: int
    name: str
    ground_type: str
    wetness: str
    depth_class: str
    hard: bool
    bendable: bool
    water_factor: float
    soil_thickness: float
    coefficients: dict[str, float] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    profile: SurfaceProfile | None = None
    profile_status: str = "unmapped"
    physics_fingerprint: str = ""
    catalog_version: int = 0

    def __post_init__(self) -> None:
        if self.profile is not None or not self.ground_type:
            return
        from .surface_profiles import SurfaceProfile

        object.__setattr__(
            self,
            "profile",
            SurfaceProfile(
                "explicit",
                self.ground_type,
                self.hard,
                self.bendable,
                self.ground_type == "WATER",
                self.ground_type not in {"VOID", "WATER"},
            ),
        )
        if self.profile_status == "unmapped":
            object.__setattr__(self, "profile_status", "mapped")


@dataclass(frozen=True)
class StageLocation:
    country_code: str
    country: str
    region: str
    latitude: float
    longitude: float
    utc_offset: str
    dst_rule: str
    precision: str
    altitude_meters: float = 0.0
    summer_temperature_night: float | None = None
    summer_temperature_day: float | None = None
    autumn_temperature_night: float | None = None
    autumn_temperature_day: float | None = None
    winter_temperature_night: float | None = None
    winter_temperature_day: float | None = None


@dataclass(frozen=True)
class StageDocument:
    path: Path
    title: str


@dataclass(frozen=True)
class Spawn:
    matrix: tuple[float, ...]
    angles: tuple[float, float, float]

    @property
    def position(self) -> tuple[float, float, float]:
        return self.matrix[12], self.matrix[13], self.matrix[14]


@dataclass(frozen=True)
class DrivelinePoint:
    position: tuple[float, float, float]
    direction: tuple[float, float, float]
    distance: float
    flags: int


@dataclass(frozen=True)
class Pacenote:
    note_type: int
    distance: float
    flag: int


@dataclass
class RbrStage:
    root: Path
    metadata: StageMetadata
    objects: list[StageObject]
    materials: list[RbrMaterial]
    surface_maps: dict[str, SurfaceMap]
    surface_types: dict[int, str]
    surfaces: dict[int, RbrSurface]
    spawn: Spawn
    driveline: list[DrivelinePoint]
    pacenotes: list[Pacenote]
    location: StageLocation | None = None
    documents: tuple[StageDocument, ...] = ()
    sun_direction: tuple[float, float, float] | None = None
    warnings: list[str] = field(default_factory=list)
    source_format: str = "rx"
    source_variant: str = ""
    source_provenance: dict[str, str] = field(default_factory=dict)
    unresolved_surface_ids: tuple[int, ...] = ()
    used_surface_ids: tuple[int, ...] = ()
    # RSF's RX plugin gives collision whose texture has no mat.ini map the
    # stage's first map, MAP0.
    unmatched_surface_map: SurfaceMap | None = None

    @property
    def has_finish(self) -> bool:
        return any(note.note_type == 22 for note in self.pacenotes)


@dataclass
class MeshPart:
    name: str
    vertices: Any
    faces: Any
    normals: Any
    texcoords: Any
    colors: Any
    material_index: int
    material_name: str
    collision_eligible: bool = True
    water: bool = False
    texcoord_sets: tuple[Any, ...] = ()
    semantic_uv_indices: dict[str, int] = field(default_factory=dict)
    specular_strengths: Any | None = None
    blend_weights: Any | None = None
    force_color_stream: bool = False
    lod_group: str | None = None
    lod_kind: str = "any"
    snowbank_floor: bool = False


@dataclass
class MeshAsset:
    source: Path
    parts: list[MeshPart]
    material_names: list[str]


@dataclass
class WaterRegion:
    material_name: str
    vertices: Any
    faces: Any
    tiled: bool = False
    source_ids: tuple[str, ...] = ()
    appearance: WaterAppearance | None = None


@dataclass(frozen=True)
class ConversionResult:
    output_path: Path
    level_id: str
    warnings: tuple[str, ...]
    stats: dict[str, bool | int | float | str]
