from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .contracts import Job
from .cutout import bright_foreground_rgba, grabcut_rgba, rembg_rgba
from .errors import ContractError
from .image_io import load_background, load_mask, save_png
from .learned_effects import RegisteredNpzEffectProvider
from .pipeline import StrictCompositor
from .quality import FinalGateError, verify_deliverable
from .validate import audit_background


def _read_json(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def compose(args: argparse.Namespace) -> int:
    file = Path(args.job).resolve()
    job = Job.from_dict(_read_json(str(file)), file.parent)
    output_dir = Path(args.out).resolve()
    registrations: dict[str, Path] = {}
    for item in args.learned_effect or []:
        if "=" not in item:
            raise ContractError(
                "--learned-effect must be PRODUCT_ID=EFFECTS.npz"
            )
        product_id, raw_path = item.split("=", 1)
        product_id = product_id.strip()
        if not product_id or product_id in registrations:
            raise ContractError(
                "--learned-effect product ids must be non-empty and unique"
            )
        registrations[product_id] = Path(raw_path).resolve()
    expected_ids = {product.product_id for product in job.products}
    if registrations and set(registrations) != expected_ids:
        raise ContractError(
            "--learned-effect must register exactly one NPZ for every product; "
            f"expected={sorted(expected_ids)} got={sorted(registrations)}"
        )
    learned_provider = (
        RegisteredNpzEffectProvider(registrations)
        if registrations
        else None
    )
    result = StrictCompositor(
        learned_effect_provider=learned_provider,
    ).compose(job, output_dir)
    # The review stage reads material, lighting and placement semantics from
    # the run-local job manifest.  UI composition already writes that file,
    # but the CLI historically omitted it, silently causing a clear pitcher to
    # be reviewed as opaque and an ambient-only scene as directional.  Keep an
    # exact provenance copy beside every backend result so CLI and UI runs have
    # identical evidence contracts.
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = output_dir / "job.json"
    if file != provenance:
        shutil.copy2(file, provenance)
    audit_payload = _read_json(str(output_dir / "audit.json"))
    print(json.dumps({
        "pixel_verdict": result.audit.verdict,
        "delivery_verdict": "pending_visual_review",
        "batch_unlock": False,
        "learned_effects": audit_payload.get("learned_effects", {}),
    }, ensure_ascii=False))
    return 0


def validate(args: argparse.Namespace) -> int:
    background = load_background(args.background)
    result = load_background(args.result)
    mask = load_mask(args.authorized_mask, (background.shape[1], background.shape[0]))
    audit = audit_background(background, result, mask)
    print(json.dumps(audit.to_dict(), ensure_ascii=False))
    return 0 if audit.verdict == "pass" else 2


def analyze(args: argparse.Namespace) -> int:
    """Validate analysis JSON; image-producing model outputs are out of contract."""
    data = _read_json(args.analysis)
    required = {"support_polygon", "lighting", "products", "route"}
    missing = required - set(data)
    if missing or data.get("route") not in {"strict_composite", "relight_proxy", "reject"}:
        raise ContractError(f"Invalid analysis JSON; missing={sorted(missing)}")
    print(json.dumps({"verdict": "pass", "accepted_fields": sorted(required)}, ensure_ascii=False))
    return 0


def segment(args: argparse.Namespace) -> int:
    if args.method == "grabcut":
        if args.rect is None:
            raise ContractError("--rect is required for --method grabcut")
        grabcut_rgba(args.input, args.output, tuple(args.rect), args.iterations, not args.no_trim, tuple(args.keep_rect) if args.keep_rect else None, args.remove_dark)
    elif args.method == "rembg":
        rembg_rgba(args.input, args.output, args.model, not args.no_trim, tuple(args.keep_rect) if args.keep_rect else None, args.remove_dark)
    else:
        if args.rect is None or args.brightness_threshold is None:
            raise ContractError("--rect and --brightness-threshold are required for --method bright")
        bright_foreground_rgba(args.input, args.output, tuple(args.rect), args.brightness_threshold, not args.no_trim)
    print(json.dumps({"verdict": "pass", "output": args.output, "method": args.method}, ensure_ascii=False))
    return 0


def quality(args: argparse.Namespace) -> int:
    """Revalidate the same immutable final gate used by web delivery.

    A caller-supplied numeric score is not evidence.  The former ``--score``
    interface allowed any batch consumer to turn a real five-slot reject into
    ``batch_unlock=true``.  CLI, web and automation now share the identical
    five-review, hard-gate and live SHA-256 verification path.
    """
    final_path = Path(args.final).resolve()
    gate_path = (
        Path(args.gate).resolve()
        if args.gate
        else final_path.parent / "final_gate.json"
    )
    if not gate_path.is_file():
        raise ContractError("final_gate.json is required; manual scores are not accepted")
    try:
        gate = _read_json(str(gate_path))
        verify_deliverable(final_path, gate)
    except (OSError, ValueError, TypeError, FinalGateError) as exc:
        raise ContractError(f"Delivery gate rejected: {exc}") from exc
    print(json.dumps({
        "visual_score": float(gate["visual_score"]),
        "verdict": "pass",
        "delivery_verdict": "pass",
        "batch_unlock": True,
        "final_sha256": gate["inputs"]["final_sha256"],
    }, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict deterministic product compositor")
    commands = parser.add_subparsers(required=True)
    compose_parser = commands.add_parser("compose")
    compose_parser.add_argument("--job", required=True)
    compose_parser.add_argument("--out", required=True)
    compose_parser.add_argument(
        "--learned-effect",
        action="append",
        metavar="PRODUCT_ID=EFFECTS.npz",
        help=(
            "experimental restricted learned-effect tensors; repeat exactly "
            "once for every product"
        ),
    )
    compose_parser.set_defaults(func=compose)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--background", required=True)
    validate_parser.add_argument("--result", required=True)
    validate_parser.add_argument("--authorized-mask", required=True)
    validate_parser.set_defaults(func=validate)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--analysis", required=True)
    analyze_parser.set_defaults(func=analyze)
    segment_parser = commands.add_parser("segment")
    segment_parser.add_argument("--input", required=True)
    segment_parser.add_argument("--output", required=True)
    segment_parser.add_argument("--method", choices=["grabcut", "rembg", "bright"], default="grabcut")
    segment_parser.add_argument("--rect", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    segment_parser.add_argument("--iterations", default=5, type=int)
    segment_parser.add_argument("--model", default="u2net")
    segment_parser.add_argument("--keep-rect", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="remove incidental props outside this reviewed foreground rectangle")
    segment_parser.add_argument("--remove-dark", type=int, metavar="0..254", help="optionally remove dark incidental props after alpha estimation")
    segment_parser.add_argument("--brightness-threshold", type=int, metavar="0..254", help="required by --method bright")
    segment_parser.add_argument("--no-trim", action="store_true", help="keep original transparent canvas")
    segment_parser.set_defaults(func=segment)
    quality_parser = commands.add_parser("quality")
    quality_parser.add_argument("--final", required=True, help="reviewed final PNG")
    quality_parser.add_argument(
        "--gate",
        help="final_gate.json; defaults to the final image directory",
    )
    quality_parser.set_defaults(func=quality)
    args = parser.parse_args()
    try:
        return args.func(args)
    except ContractError as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
