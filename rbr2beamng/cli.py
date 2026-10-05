from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
import warnings
from datetime import date
from pathlib import Path

from . import __version__
from .beamng import render_stage_pacenote_visualizer
from .blend_audit import build_blend_audit
from .conversion_common import ConversionOptions
from .converter import convert
from .core import (
    ConversionCancelled,
    ConversionError,
    ProgressReporter,
    create_runtime_filesystem,
    find_beamng_mods_dir,
    find_rbr_install,
    format_duration,
    format_file_size,
    settings_path,
    slugify,
    stage_zip_name,
    temporary_root,
)
from .filesystem import use_filesystem
from .geometry import AssimpMeshLoader, source_position_to_beamng
from .models import (
    DEFAULT_FOLIAGE_GROUND_TYPES,
    DEFAULT_FOLIAGE_NAME_MATCHES,
    DEFAULT_SNOWBANK_NAME_MATCHES,
    DEFAULT_SNOWBANK_NAME_MESH_PATTERNS,
    DEFAULT_WATER_NAME_MATCHES,
)
from .original.lbs import parse_lbs
from .surface_profiles import GROUND_TYPES, load_surface_rules, use_surface_rules
from .original.fnc_inventory import build_fnc_inventory
from .original.source import (
    build_original_pacenote_stage,
    original_level_id,
    original_stage_id,
    resolve_original_pacenote_visualizer_variants,
    resolve_original_variants,
    select_original_variants,
)
from .pacenote_audit import build_pacenote_audit
from .pacenote_inventory import build_pacenote_inventory
from .profiling import (
    ConversionProfiler,
    profile_conversion,
    profile_directory,
)
from .rbr import find_stage, load_pacenote_visualizer_stage, load_stage
from .stage_sources import discover_stages
from .water_appearance import (
    DEFAULT_WATER_PROFILE_MANIFEST,
    water_profile_manifest_from_data,
)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _environment_date(value: str) -> str:
    text = value.strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must use YYYY-MM-DD format"
        ) from exc
    if text != parsed.isoformat():
        raise argparse.ArgumentTypeError("must use YYYY-MM-DD format")
    return text


def _map_altitude(value: str) -> float:
    try:
        altitude = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number of meters") from exc
    if not math.isfinite(altitude):
        raise argparse.ArgumentTypeError("must be a finite number of meters")
    return altitude


def _name_pattern(value: str) -> str:
    try:
        re.compile(value)
    except re.error as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid name pattern: {exc}"
        ) from exc
    return value


def _rbr_root(value: Path | None) -> Path:
    if value:
        return value.resolve()
    detected = find_rbr_install()
    if detected is None:
        raise ConversionError("RBR installation not found; pass --rbr-dir")
    return detected


def _beamng_mods_dir(value: Path | None, output: Path | None) -> Path | None:
    if value:
        return value.resolve()
    if output:
        return None
    detected = find_beamng_mods_dir()
    if detected is None:
        raise ConversionError("BeamNG mods folder not found; pass --beamng-mods-dir or --output")
    return detected


def _add_rbr_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rbr-dir", type=_path, help="RBR installation folder")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rbr2beamng-cli",
        description="Convert RBR RX and Original RBR stages into BeamNG.drive map mods.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List supported RBR stages")
    _add_rbr_argument(list_parser)
    list_parser.add_argument("--json", action="store_true", help="Print JSON")

    inspect_parser = subparsers.add_parser("inspect", help="Validate and describe one stage")
    _add_rbr_argument(inspect_parser)
    inspect_parser.add_argument("--stage", required=True, help="Stage folder or displayed name")
    inspect_parser.add_argument("--json", action="store_true", help="Print JSON")

    audit_parser = subparsers.add_parser("audit", help="Check every discovered RBR stage")
    _add_rbr_argument(audit_parser)
    audit_parser.add_argument(
        "--meshes",
        choices=("none", "sample", "all"),
        default="none",
        help="Also load no meshes, one mesh per stage, or every unique mesh through Assimp",
    )
    audit_parser.add_argument("--json", action="store_true", help="Print JSON")

    inventory_parser = subparsers.add_parser(
        "pacenote-inventory",
        help="Inventory raw pacenote records without converting stages",
    )
    _add_rbr_argument(inventory_parser)
    inventory_parser.add_argument(
        "--output",
        type=_path,
        required=True,
        help="New directory for the inventory report bundle",
    )
    inventory_parser.add_argument(
        "--hash-inputs",
        action="store_true",
        help="Add SHA-256 hashes to source fingerprints",
    )
    inventory_parser.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit one machine-readable JSON progress event per line",
    )

    pacenote_audit_parser = subparsers.add_parser(
        "pacenote-audit",
        help="Run a pacenote conversion sweep and write a self-contained HTML report",
    )
    _add_rbr_argument(pacenote_audit_parser)
    pacenote_audit_parser.add_argument(
        "--output",
        type=_path,
        required=True,
        help="New HTML file for the pacenote conversion audit",
    )
    pacenote_audit_parser.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit one machine-readable JSON progress event per line",
    )

    blend_audit_parser = subparsers.add_parser(
        "blend-audit",
        help="Measure vertex-blend material capability across RX stages",
    )
    _add_rbr_argument(blend_audit_parser)
    blend_audit_parser.add_argument(
        "--output",
        type=_path,
        required=True,
        help="New HTML file for the blend capability audit",
    )
    blend_audit_parser.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit one machine-readable JSON progress event per line",
    )

    fnc_inventory_parser = subparsers.add_parser(
        "fnc-inventory",
        help="Inventory Original RBR fence selector usage",
    )
    _add_rbr_argument(fnc_inventory_parser)
    fnc_inventory_parser.add_argument(
        "--output",
        type=_path,
        required=True,
        help="New JSON file for the FNC selector inventory",
    )
    fnc_inventory_parser.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit one machine-readable JSON progress event per line",
    )

    visualizer_parser = subparsers.add_parser(
        "pacenote-visualizer",
        help="Generate pacenote visualizer HTML without converting a stage",
    )
    _add_rbr_argument(visualizer_parser)
    visualizer_parser.add_argument(
        "--stage",
        required=True,
        help="Stage folder, displayed name, or Original RBR stage selector",
    )
    visualizer_parser.add_argument(
        "--output",
        type=_path,
        required=True,
        help="Directory for generated visualizer HTML file(s)",
    )
    visualizer_parser.add_argument(
        "--original-variant",
        action="append",
        choices=("M", "N", "O", "E", "S"),
        default=[],
        help=(
            "Generate this Original RBR variant (default: only the primary "
            "variant); repeat to select several"
        ),
    )

    convert_parser = subparsers.add_parser("convert", help="Convert and install one stage")
    _add_rbr_argument(convert_parser)
    convert_parser.add_argument("--stage", required=True, help="Stage folder or displayed name")
    convert_parser.add_argument("--beamng-mods-dir", type=_path, help="BeamNG mods folder")
    convert_parser.add_argument("--output", type=_path, help="Explicit ZIP output path")
    convert_parser.add_argument("--overwrite", action="store_true", help="Replace converter output")
    convert_parser.add_argument(
        "--remove-source-skybox",
        action="store_true",
        help="Remove source sky domes and cloud layers: mesh parts at least 100 m above a tenth of the route",
    )
    convert_parser.add_argument(
        "--no-water-name-fallback",
        dest="use_water_name_fallback",
        action="store_false",
        help="Do not use material and texture names to detect procedural water",
    )
    convert_parser.add_argument(
        "--water-name-matches",
        nargs="*",
        metavar="NAME",
        default=None,
        help="Replace the water name word starts; prefix an entry with ! to reject names containing it",
    )
    convert_parser.add_argument(
        "--water-profile-manifest",
        type=_path,
        metavar="PATH",
        help="JSON manifest mapping exact source material IDs to water profiles",
    )
    convert_parser.add_argument(
        "--no-snowwall-collision-override",
        dest="use_snowwall_collision_override",
        action="store_false",
        help="Keep snow on snowbank textures (Snowwall 62/88) without collision when the stage marks it as non-colliding",
    )
    convert_parser.add_argument(
        "--no-snowbank-name-fallback",
        dest="use_snowbank_name_fallback",
        action="store_false",
        help="Do not repair snow walls without surface data by guessing from material and texture names",
    )
    convert_parser.add_argument(
        "--snowbank-name-matches",
        nargs="*",
        metavar="NAME",
        default=None,
        help="Replace the snow wall names that get snowbank physics",
    )
    convert_parser.add_argument(
        "--snowbank-name-mesh-patterns",
        type=_name_pattern,
        nargs="*",
        metavar="REGEX",
        default=None,
        help="Replace the mesh file name patterns (matched at the start) whose snowbank-named materials get collision",
    )
    convert_parser.add_argument(
        "--no-thin-wall-inflation",
        dest="inflate_thin_walls",
        action="store_false",
        help="Keep Original RBR thin paired collision walls instead of inflating them outward",
    )
    convert_parser.add_argument(
        "--no-visual-lods",
        dest="use_visual_lods",
        action="store_false",
        help="Keep only the highest-quality source visuals without distance LODs",
    )
    convert_parser.add_argument(
        "--no-map-border-brake-walls",
        dest="use_map_border_brake_walls",
        action="store_false",
        help="Do not convert Original RBR outer map brake-wall boundaries",
    )
    convert_parser.add_argument(
        "--no-foliage-name-fallback",
        dest="use_foliage_name_fallback",
        action="store_false",
        help="Do not use material and texture names to select vegetation shading",
    )
    convert_parser.add_argument(
        "--foliage-name-matches",
        nargs="*",
        metavar="NAME",
        default=None,
        help="Replace the fallback vegetation name matches",
    )
    convert_parser.add_argument(
        "--no-foliage-ground-types",
        dest="use_foliage_ground_types",
        action="store_false",
        help="Do not use surface types to select vegetation shading",
    )
    convert_parser.add_argument(
        "--foliage-ground-types",
        nargs="*",
        choices=sorted(GROUND_TYPES),
        metavar="GROUND_TYPE",
        default=None,
        help="Replace the BeamNG ground types that select vegetation shading",
    )
    convert_parser.add_argument(
        "--temperature-night",
        type=float,
        help="Override the nighttime temperature in Celsius",
    )
    convert_parser.add_argument(
        "--temperature-day",
        type=float,
        help="Override the daytime temperature in Celsius",
    )
    convert_parser.add_argument(
        "--latitude",
        type=float,
        help="Override the stage latitude in decimal degrees",
    )
    convert_parser.add_argument(
        "--longitude",
        type=float,
        help="Override the stage longitude in decimal degrees",
    )
    convert_parser.add_argument(
        "--map-altitude",
        type=_map_altitude,
        metavar="METERS",
        help="Place the source spawn at this altitude above mean sea level",
    )
    convert_parser.add_argument(
        "--environment-date",
        type=_environment_date,
        metavar="YYYY-MM-DD",
        help="Override the generated TimeOfDay calendar date",
    )
    convert_parser.add_argument(
        "--preview-radius",
        type=float,
        metavar="METERS",
        help="Temporary: only convert geometry within METERS of the rally start",
    )
    convert_parser.add_argument(
        "--original-variant",
        action="append",
        choices=("M", "N", "O", "E", "S"),
        default=[],
        help=(
            "Convert this Original RBR variant (default: only the primary "
            "variant, Overcast when available); repeat to select several"
        ),
    )
    convert_parser.add_argument(
        "--environment",
        choices=("D", "M", "N", "O", "E"),
        default="N",
        help=(
            "Set the starting environment of RX and primary Original levels "
            "(default: N/noon)"
        ),
    )
    convert_parser.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit one machine-readable JSON progress event per line",
    )
    return parser


def _list(args: argparse.Namespace) -> int:
    root = _rbr_root(args.rbr_dir)
    filesystem = create_runtime_filesystem(
        rbr_root=root,
        include_runtime_write_roots=False,
    )
    with use_filesystem(filesystem):
        stages = discover_stages(root)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "folder": stage.metadata.folder_name,
                        "name": stage.metadata.name,
                        "author": stage.metadata.author,
                        "physics": stage.metadata.physics,
                        "sourceFormat": stage.source_format,
                        "variants": stage.variants,
                        "lengthKm": stage.metadata.length_km,
                        "valid": stage.valid,
                        "issues": stage.issues,
                    }
                    for stage in stages
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(f"{'Selector':32} {'Name':34} {'Format':8} {'Surface':9} {'km':>7} Status")
    for stage in stages:
        length = f"{stage.metadata.length_km:.2f}" if stage.metadata.length_km is not None else "-"
        status = "ready" if stage.valid else "; ".join(stage.issues)
        print(
            f"{stage.metadata.folder_name[:32]:32} "
            f"{stage.metadata.name[:34]:34} "
            f"{stage.source_format[:8]:8} "
            f"{stage.metadata.physics[:9]:9} "
            f"{length:>7} {status}"
        )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    root = _rbr_root(args.rbr_dir)
    filesystem = create_runtime_filesystem(
        rbr_root=root,
        include_runtime_write_roots=False,
    )
    filesystem.add_read_write(temporary_root())
    with use_filesystem(filesystem):
        original_id = original_stage_id(args.stage)
        if original_id is not None:
            inspection, variants = resolve_original_variants(root, args.stage)
            data = {
                "selector": inspection.metadata.folder_name,
                "sourceFormat": "original",
                "stageId": original_id,
                "name": inspection.metadata.name,
                "author": inspection.metadata.author,
                "physics": inspection.metadata.physics,
                "lengthKm": inspection.metadata.length_km,
                "variants": [variant.tint.value for variant in variants],
                "variantFiles": {
                    variant.tint.value: {
                        name: str(path)
                        for name, path in variant.files.items()
                    }
                    for variant in variants
                },
                "country": (
                    inspection.location.country
                    if inspection.location
                    else None
                ),
                "documents": [
                    document.title for document in inspection.documents
                ],
                "warnings": list(inspection.issues),
            }
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                for key, value in data.items():
                    print(f"{key}: {value}")
            return 0
        stage = load_stage(find_stage(root, args.stage))
        data = {
            "folder": stage.metadata.folder_name,
            "name": stage.metadata.name,
            "author": stage.metadata.author,
            "physics": stage.metadata.physics,
            "lengthKm": stage.metadata.length_km,
            "objects": len(stage.objects),
            "instances": sum(stage_object.clone_count for stage_object in stage.objects),
            "uniqueMeshes": len({stage_object.mesh_path for stage_object in stage.objects}),
            "materials": len(stage.materials),
            "surfaceMaps": len(stage.surface_maps),
            "surfaceTypes": len(stage.surfaces),
            "drivelinePoints": len(stage.driveline),
            "pacenotes": len(stage.pacenotes),
            "country": stage.location.country if stage.location else None,
            "geolocationPrecision": stage.location.precision if stage.location else None,
            "documents": [document.title for document in stage.documents],
            "warnings": stage.warnings,
        }
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        for key, value in data.items():
            print(f"{key}: {value}")
    return 0


def _audit(args: argparse.Namespace) -> int:
    root = _rbr_root(args.rbr_dir)
    filesystem = create_runtime_filesystem(
        rbr_root=root,
        include_runtime_write_roots=False,
    )
    filesystem.add_read_write(temporary_root())
    with use_filesystem(filesystem):
        inspections = discover_stages(root)
        loader = AssimpMeshLoader() if args.meshes != "none" else None
        results: list[dict[str, object]] = []
        for index, inspection in enumerate(inspections, 1):
            print(f"Auditing {index}/{len(inspections)}: {inspection.metadata.folder_name}", file=sys.stderr)
            errors = list(inspection.issues)
            mesh_count = 0
            if not errors:
                try:
                    if inspection.source_format == "original":
                        if args.meshes != "none":
                            _, original_variants = resolve_original_variants(
                                root,
                                inspection.metadata.folder_name,
                            )
                            if args.meshes == "sample":
                                original_variants = original_variants[:1]
                            for variant in original_variants:
                                parse_lbs(variant.files["track"])
                                mesh_count += 1
                    else:
                        stage = load_stage(inspection.root)
                        mesh_paths = sorted(
                            {
                                stage_object.mesh_path
                                for stage_object in stage.objects
                            }
                        )
                        if args.meshes == "sample":
                            mesh_paths = mesh_paths[:1]
                        elif args.meshes == "none":
                            mesh_paths = []
                        for mesh_path in mesh_paths:
                            assert loader is not None
                            loader.load(mesh_path)
                            mesh_count += 1
                except (ConversionError, OSError, ValueError) as exc:
                    errors.append(str(exc))
            results.append(
                {
                    "folder": inspection.metadata.folder_name,
                    "name": inspection.metadata.name,
                    "ok": not errors,
                    "meshesChecked": mesh_count,
                    "errors": errors,
                }
            )
    summary = {
        "rbrDir": str(root),
        "stageCount": len(results),
        "passed": sum(1 for result in results if result["ok"]),
        "failed": sum(1 for result in results if not result["ok"]),
        "meshMode": args.meshes,
        "results": results,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Stages: {summary['stageCount']}; passed: {summary['passed']}; failed: {summary['failed']}")
        for result in results:
            if not result["ok"]:
                print(f"{result['folder']}: {'; '.join(result['errors'])}")
    return 0 if summary["failed"] == 0 else 1


def _pacenote_inventory(args: argparse.Namespace) -> int:
    root = _rbr_root(args.rbr_dir)
    output = args.output.resolve()
    filesystem = create_runtime_filesystem(
        rbr_root=root,
        output=output,
        include_runtime_write_roots=False,
    )
    reporter = ProgressReporter(json_lines=args.progress_json)
    with use_filesystem(filesystem):
        result = build_pacenote_inventory(
            root,
            output,
            reporter=reporter,
            hash_inputs=args.hash_inputs,
        )
        reporter.emit(
            "complete",
            "Pacenote inventory complete",
            detail=str(result.output_path),
        )
    if not args.progress_json:
        print(f"Inventory: {result.output_path}")
        print(
            f"Sources: {result.source_count}; records: {result.record_count}; "
            f"source errors: {result.error_count}"
        )
    return 1 if result.error_count else 0


def _pacenote_audit(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    rbr_root = _rbr_root(args.rbr_dir)
    filesystem = create_runtime_filesystem(
        rbr_root=rbr_root,
        output=output,
        include_runtime_write_roots=False,
    )
    reporter = ProgressReporter(json_lines=args.progress_json)
    with use_filesystem(filesystem):
        result = build_pacenote_audit(
            rbr_root,
            output,
            reporter=reporter,
        )
        reporter.emit(
            "complete",
            "Pacenote audit complete",
            detail=str(result.output_path),
        )
    if not args.progress_json:
        print(f"Pacenote audit: {result.output_path}")
        print(
            f"Stage runs: {result.completed_stage_count}/{result.stage_count}; "
            f"problems: {result.problem_count}; source errors: {result.error_count}"
        )
        print(f"Elapsed total: {format_duration(reporter.elapsed_seconds())}")
    return 1 if result.error_count else 0


def _blend_audit(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    rbr_root = _rbr_root(args.rbr_dir)
    filesystem = create_runtime_filesystem(
        rbr_root=rbr_root,
        output=output,
        include_runtime_write_roots=False,
    )
    filesystem.add_read_write(temporary_root())
    reporter = ProgressReporter(json_lines=args.progress_json)
    with use_filesystem(filesystem):
        result = build_blend_audit(
            rbr_root,
            output,
            reporter=reporter,
        )
        reporter.emit(
            "complete",
            "Blend capability audit complete",
            detail=str(result.output_path),
        )
    if not args.progress_json:
        print(f"Blend audit: {result.output_path}")
        print(
            f"Stages: {result.completed_stage_count}/{result.stage_count}; "
            f"needs custom material: {result.affected_stage_count}; "
            f"source errors: {result.error_count}"
        )
        print(f"Elapsed total: {format_duration(reporter.elapsed_seconds())}")
    return 1 if result.error_count else 0


def _fnc_inventory(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    rbr_root = _rbr_root(args.rbr_dir)
    filesystem = create_runtime_filesystem(
        rbr_root=rbr_root,
        output=output,
        include_runtime_write_roots=False,
    )
    reporter = ProgressReporter(json_lines=args.progress_json)
    with use_filesystem(filesystem):
        result = build_fnc_inventory(
            rbr_root,
            output,
            reporter=reporter,
        )
        reporter.emit(
            "complete",
            "FNC inventory complete",
            detail=str(result.output_path),
        )
    if not args.progress_json:
        print(f"FNC inventory: {result.output_path}")
        print(
            f"Source files: {result.source_count}; "
            f"selector pairs: {result.selector_count}; "
            f"source errors: {result.error_count}"
        )
        print(f"Elapsed total: {format_duration(reporter.elapsed_seconds())}")
    return 1 if result.error_count else 0


def _pacenote_visualizer(args: argparse.Namespace) -> int:
    output_dir = args.output.resolve()
    rbr_root = _rbr_root(args.rbr_dir)
    filesystem = create_runtime_filesystem(
        rbr_root=rbr_root,
        output=output_dir,
        include_runtime_write_roots=False,
    )
    outputs: list[Path] = []
    with use_filesystem(filesystem):
        filesystem.mkdir(output_dir, parents=True, exist_ok=True)
        if original_stage_id(args.stage) is not None:
            inspection, variants = resolve_original_pacenote_visualizer_variants(
                rbr_root,
                args.stage,
                tuple(args.original_variant),
            )
            variants, primary_tint = select_original_variants(
                variants,
                tuple(args.original_variant),
            )
            package_id = Path(
                stage_zip_name(
                    inspection.metadata.folder_name,
                    display_name=inspection.metadata.name,
                    source_format="original",
                )
            ).stem
            stages = [
                build_original_pacenote_stage(rbr_root, inspection, files)
                for files in variants
            ]
            map_yaws = [0.0] * len(stages)
            output_stems = [
                original_level_id(package_id, files.tint.value, primary_tint)
                for files in variants
            ]
        else:
            stage = load_pacenote_visualizer_stage(
                find_stage(rbr_root, args.stage)
            )
            stages = [stage]
            map_yaws = [0.0]
            output_stems = [f"rbr_{slugify(stage.metadata.folder_name)}"]
        for stage, map_yaw, output_stem in zip(stages, map_yaws, output_stems):
            output = output_dir / f"{output_stem}.html"
            html = render_stage_pacenote_visualizer(
                stage,
                source_position_to_beamng(stage.spawn.position),
                map_yaw,
            )
            filesystem.write_text(output, html, encoding="utf-8")
            outputs.append(output)
    for output in outputs:
        print(f"Visualizer: {output}")
    return 0


def _convert(args: argparse.Namespace) -> int:
    output = args.output.resolve() if args.output else None
    rbr_root = _rbr_root(args.rbr_dir)
    beamng_mods_dir = _beamng_mods_dir(args.beamng_mods_dir, output)
    filesystem = create_runtime_filesystem(
        rbr_root=rbr_root,
        beamng_mods_dir=beamng_mods_dir,
        output=output,
    )
    filesystem.add_read_write(profile_directory(settings_path().parent))
    water_profile_manifest_path = (
        args.water_profile_manifest.resolve()
        if args.water_profile_manifest is not None
        else None
    )
    water_profiles = DEFAULT_WATER_PROFILE_MANIFEST
    if water_profile_manifest_path is not None:
        filesystem.add_read_only(water_profile_manifest_path.parent)
        try:
            water_profiles = water_profile_manifest_from_data(
                json.loads(
                    filesystem.read_text(water_profile_manifest_path, encoding="utf-8")
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ConversionError(str(exc)) from exc
    options = ConversionOptions(
        rbr_root=rbr_root,
        stage=args.stage,
        filesystem=filesystem,
        beamng_mods_dir=beamng_mods_dir,
        output=output,
        overwrite=args.overwrite,
        rights_confirmed=os.environ.get("RBR2BEAMNG_GUI_RIGHTS_CONFIRMED") == "1",
        remove_source_skybox=args.remove_source_skybox,
        use_water_name_fallback=args.use_water_name_fallback,
        water_name_matches=(
            tuple(args.water_name_matches)
            if args.water_name_matches is not None
            else DEFAULT_WATER_NAME_MATCHES
        ),
        water_profiles=water_profiles,
        use_snowwall_collision_override=args.use_snowwall_collision_override,
        use_snowbank_name_fallback=args.use_snowbank_name_fallback,
        snowbank_name_matches=(
            tuple(args.snowbank_name_matches)
            if args.snowbank_name_matches is not None
            else DEFAULT_SNOWBANK_NAME_MATCHES
        ),
        snowbank_name_mesh_patterns=(
            tuple(args.snowbank_name_mesh_patterns)
            if args.snowbank_name_mesh_patterns is not None
            else DEFAULT_SNOWBANK_NAME_MESH_PATTERNS
        ),
        inflate_thin_walls=args.inflate_thin_walls,
        use_visual_lods=args.use_visual_lods,
        use_map_border_brake_walls=args.use_map_border_brake_walls,
        temperature_night=args.temperature_night,
        temperature_day=args.temperature_day,
        latitude=args.latitude,
        longitude=args.longitude,
        map_altitude_meters=args.map_altitude,
        environment_date=args.environment_date,
        preview_radius_m=args.preview_radius,
        original_variants=tuple(args.original_variant),
        environment=args.environment,
        use_foliage_name_fallback=args.use_foliage_name_fallback,
        foliage_name_matches=(
            tuple(args.foliage_name_matches)
            if args.foliage_name_matches is not None
            else DEFAULT_FOLIAGE_NAME_MATCHES
        ),
        use_foliage_ground_types=args.use_foliage_ground_types,
        foliage_ground_types=(
            tuple(args.foliage_ground_types)
            if args.foliage_ground_types is not None
            else DEFAULT_FOLIAGE_GROUND_TYPES
        ),
    )
    reporter = ProgressReporter(json_lines=args.progress_json)
    with use_filesystem(filesystem), use_surface_rules(load_surface_rules()):
        profile = ConversionProfiler(
            profile_directory(settings_path().parent),
            stage=args.stage,
            source_format=(
                "original"
                if original_stage_id(args.stage) is not None
                else "rx"
            ),
            converter_version=__version__,
            options=options,
        )
        with profile_conversion(profile), warnings.catch_warnings():
            warnings.showwarning = lambda message, *_args, **_kwargs: reporter.warning(
                "warning",
                str(message),
            )
            result = convert(options, reporter)
            zip_size = int(result.stats["zipSizeBytes"])
            profile.set_result(
                output_path=result.output_path,
                warning_count=len(result.warnings),
                levelId=result.level_id,
                zipSizeBytes=zip_size,
                stats=result.stats,
            )
    if not args.progress_json:
        print(f"Installed: {result.output_path}")
        print(
            f"Stage: {result.stats['routeLengthMeters'] / 1000:.2f} km · "
            f"{result.stats['sectorCount']} sectors"
        )
        print(f"Elapsed total: {format_duration(reporter.elapsed_seconds())}")
        print(f"ZIP size: {format_file_size(zip_size)}")
        if profile.profile_path:
            print(f"Profile: {profile.profile_path}")
        if result.warnings:
            print(f"Warnings: {len(result.warnings)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return {
            "list": _list,
            "inspect": _inspect,
            "audit": _audit,
            "pacenote-inventory": _pacenote_inventory,
            "pacenote-audit": _pacenote_audit,
            "blend-audit": _blend_audit,
            "fnc-inventory": _fnc_inventory,
            "pacenote-visualizer": _pacenote_visualizer,
            "convert": _convert,
        }[args.command](args)
    except ConversionCancelled as exc:
        print(str(exc), file=sys.stderr)
        return 130
    except (ConversionError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if getattr(args, "progress_json", False):
            traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
