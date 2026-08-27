"""Duplicate-safe discovery of Kulsary restored GRD SAFE products."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from infer_safe import (
    InferenceError,
    SafeProduct,
    resolve_safe_product,
    validate_safe_product,
)
from utils.kulsary_temporal import ROLE_DATES


_COG_MARKER = "_COG.SAFE"


@dataclass(frozen=True)
class _SafeCandidate:
    path: Path
    from_products: bool


@dataclass(frozen=True)
class _ParsedCandidate:
    product: SafeProduct
    from_products: bool


def _is_safe_directory(path: Path) -> bool:
    return path.is_dir() and path.name.upper().endswith(".SAFE")


def _is_cog_identifier(identifier: str) -> bool:
    return _COG_MARKER in identifier.upper()


def _cog_rejection_error(identifiers: list[str]) -> InferenceError:
    found = ", ".join(sorted(set(identifiers)))
    return InferenceError(
        "full conversion requires restored standard GRD SAFE products; "
        f"COG SAFE inputs were found: {found}. "
        "Pass --safe-root pointing to the restored_grd directory."
    )


def _iter_safe_candidates(safe_root: Path) -> list[_SafeCandidate]:
    """Non-recursive scan of safe_root and safe_root/products .SAFE children."""
    candidates: list[_SafeCandidate] = []
    search_roots = (
        (safe_root, False),
        (safe_root / "products", True),
    )
    for search_root, from_products in search_roots:
        if not search_root.is_dir():
            continue
        for path in sorted(search_root.iterdir(), key=lambda item: item.name):
            if _is_safe_directory(path):
                candidates.append(
                    _SafeCandidate(path=path, from_products=from_products)
                )
    return candidates


def _select_duplicate(candidates: list[_ParsedCandidate]) -> _ParsedCandidate:
    preferred = [item for item in candidates if item.from_products]
    pool = preferred or candidates
    return min(pool, key=lambda item: str(item.product.root))


def validate_kulsary_product_geometry(
    products: dict[str, SafeProduct],
    required_polarizations=("VV",),
) -> None:
    """Validate Kulsary geometry and the requested polarization contract."""
    for product in products.values():
        validate_safe_product(product)

    checks = (
        ("platform", {product.platform for product in products.values()}, {"S1A"}),
        (
            "product type",
            {product.product_type for product in products.values()},
            {"GRD"},
        ),
        (
            "acquisition mode",
            {product.acquisition_mode for product in products.values()},
            {"IW"},
        ),
        (
            "orbit direction",
            {product.orbit_direction for product in products.values()},
            {"ASCENDING"},
        ),
        (
            "relative orbit",
            {product.relative_orbit for product in products.values()},
            {159},
        ),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise InferenceError(
                f"Kulsary products have unexpected {label}: "
                f"{sorted(map(str, actual))}; expected {sorted(map(str, expected))}"
            )
    required = {
        str(value).strip().upper() for value in required_polarizations
    }
    if not required:
        raise ValueError("required_polarizations must not be empty")
    missing = {
        product.identifier: sorted(
            required
            - {value.upper() for value in product.polarizations}
        )
        for product in products.values()
    }
    missing = {
        identifier: values
        for identifier, values in missing.items()
        if values
    }
    if missing:
        details = ", ".join(
            f"{identifier}: {values}"
            for identifier, values in sorted(missing.items())
        )
        raise InferenceError(
            "Kulsary products are missing required polarizations: "
            f"{details}"
        )


def discover_kulsary_grd_products(
    safe_root: Path,
    required_polarizations=("VV",),
) -> dict[str, SafeProduct]:
    """Discover the three Kulsary Orbit 159 GRD SAFE products.

    Direct children of ``safe_root`` and ``safe_root/products`` ending in
    ``.SAFE`` are parsed. Symlinks are resolved, but a ``products/`` candidate
    is preferred over a top-level copy of the same identifier. Nested
    ``*_COG`` wrappers, ``predicted/``, and logs are not scanned. Identifiers
    containing ``_COG.SAFE`` are dropped before grouping.
    """
    safe_root = Path(safe_root).expanduser().resolve()
    if not safe_root.is_dir():
        raise FileNotFoundError(f"SAFE root directory is missing: {safe_root}")

    candidates = _iter_safe_candidates(safe_root)
    if not candidates:
        raise FileNotFoundError(
            "no unpacked SAFE products found directly under "
            f"{safe_root} or in {safe_root / 'products'}"
        )

    grouped: dict[str, list[_ParsedCandidate]] = defaultdict(list)
    filtered_cog: list[str] = []
    parse_errors: list[str] = []
    for candidate in candidates:
        try:
            resolved_name = candidate.path.resolve().name
        except (OSError, RuntimeError):
            resolved_name = candidate.path.name
        if (
            _is_cog_identifier(candidate.path.name)
            or _is_cog_identifier(resolved_name)
        ):
            filtered_cog.append(candidate.path.name)
            continue
        try:
            product = resolve_safe_product(str(candidate.path))
        except (InferenceError, OSError, ValueError) as exc:
            parse_errors.append(f"{candidate.path}: {exc}")
            continue
        if _is_cog_identifier(product.identifier):
            filtered_cog.append(product.identifier)
            continue
        grouped[product.identifier].append(
            _ParsedCandidate(
                product=product,
                from_products=candidate.from_products,
            )
        )

    by_date: dict = defaultdict(list)
    for parsed in (
        _select_duplicate(items) for items in grouped.values() if items
    ):
        if (
            parsed.product.start_time is not None
            and parsed.product.start_time.date() in ROLE_DATES.values()
        ):
            by_date[parsed.product.start_time.date()].append(parsed.product)

    products: dict[str, SafeProduct] = {}
    for role, acquisition_date in ROLE_DATES.items():
        matches = by_date.get(acquisition_date, [])
        if len(matches) != 1:
            if len(matches) == 0 and filtered_cog:
                raise _cog_rejection_error(filtered_cog)
            detail = (
                f"; skipped invalid candidates: {' | '.join(parse_errors[:3])}"
                if len(matches) == 0 and parse_errors
                else ""
            )
            raise InferenceError(
                "expected exactly one SAFE acquired on "
                f"{acquisition_date.isoformat()} for role {role}, "
                f"found {len(matches)}{detail}"
            )
        products[role] = matches[0]

    validate_kulsary_product_geometry(products, required_polarizations)
    return products
