from __future__ import annotations

"""Single semantic-to-render-material routing boundary.

Scene advisers, cached JSON and the browser UI used to each carry a slightly
different material interpretation.  A stale adviser result could therefore
call a colourless open glass vessel ``colored_glass`` merely because the
source room was warm, and the final renderer would preserve that room colour.

This module is intentionally small and deterministic.  It does not inspect a
particular filename or product id.  It accepts semantic evidence and returns a
render contract which every caller must apply immediately before rendering.
The product RGB/Alpha remain authoritative; this only selects the optical
transfer branch.
"""

from dataclasses import dataclass
import re

from .material_profiles import (
    TRANSMISSIVE_MATERIAL_MODES,
    canonical_material_mode,
)


@dataclass(frozen=True)
class MaterialRoute:
    requested_mode: str
    resolved_mode: str
    reason: str
    confidence_floor: float = 0.0
    transmission_floor: float = 0.0


def _semantic_tokens(semantic_category: str, physical_type: str) -> set[str]:
    text = f"{semantic_category}_{physical_type}".lower()
    return set(re.findall(r"[a-z]+", text))


def resolve_material_route(
    material_mode: str,
    semantic_category: str = "",
    physical_type: str = "",
) -> MaterialRoute:
    """Resolve a material once, at the last trusted render boundary.

    The important distinction is between intrinsic product colour and colour
    merely visible *through* a transparent product.  Generic words such as
    ``amber`` or ``tinted`` in one source photograph are not sufficient proof
    for an open vessel; stable hue words (blue/green/red/smoked/black) are.
    Labelled liquid bottles remain mixed optical objects.
    """

    requested = canonical_material_mode(material_mode)
    tokens = _semantic_tokens(semantic_category, physical_type)

    if requested not in TRANSMISSIVE_MATERIAL_MODES:
        if tokens & {"aluminum", "aluminium", "steel", "metal", "tin", "can"}:
            resolved = "metal"
        elif tokens & {"ceramic", "porcelain", "lacquer", "glazed", "glassware"}:
            resolved = "opaque_glossy"
        elif tokens & {"wood", "paper", "cardboard", "fabric", "textile", "book"}:
            resolved = "wood_paper_fabric"
        elif tokens & {"matte", "plaster", "stone"}:
            resolved = "opaque_matte"
        else:
            resolved = requested
        return MaterialRoute(
            requested, resolved,
            "semantic_refinement" if resolved != requested else "requested_material",
        )

    explicit_intrinsic_tint = bool(tokens & {
        "smoked", "blue", "green", "red", "purple", "violet", "black",
    })
    packaged_liquid = bool(tokens & {
        "whiskey", "whisky", "liquor", "spirit", "wine", "beer",
        "perfume", "sauce", "oil",
    }) and "bottle" in tokens
    if packaged_liquid:
        return MaterialRoute(
            requested,
            "mixed_transparent",
            "labelled_liquid_package",
            confidence_floor=0.78,
            transmission_floor=0.42,
        )

    # These nouns describe open vessels whose broad interior is normally the
    # photographed environment, not product pigment.  ``votive`` and
    # ``planter`` are included because catalogue advisers frequently use those
    # uncertain nouns for a plain ribbed drinking glass.  A stable intrinsic
    # hue still wins through the explicit-tint gate above.
    neutral_open_vessel = bool(tokens & {
        "pitcher", "jug", "carafe", "decanter", "tumbler", "goblet",
        "votive", "planter", "vase", "cup", "drinking",
    })
    ribbed_open_glass = "ribbed" in tokens and "glass" in tokens
    if (
        requested == "colored_glass"
        and not explicit_intrinsic_tint
        and (neutral_open_vessel or ribbed_open_glass)
    ):
        return MaterialRoute(
            requested,
            "clear_glass",
            "neutral_open_vessel_source_environment_rejection",
            confidence_floor=0.82,
            transmission_floor=0.92,
        )

    return MaterialRoute(requested, requested, "requested_material")
