from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from infer_safe import InferenceError
from utils.kulsary_products import discover_kulsary_grd_products


PRODUCTS = {
    "before": {
        "name": (
            "S1A_IW_GRDH_1SDV_20240402T141444_20240402T141510_"
            "053256_06745E_5249.SAFE"
        ),
        "date": "2024-04-02",
        "wrapper": (
            "S1A_IW_GRDH_1SDV_20240402T141444_20240402T141510_"
            "053256_06745E_D77F_COG"
        ),
    },
    "peak": {
        "name": (
            "S1A_IW_GRDH_1SDV_20240414T141444_20240414T141509_"
            "053431_067B51_75FD.SAFE"
        ),
        "date": "2024-04-14",
        "wrapper": (
            "S1A_IW_GRDH_1SDV_20240414T141444_20240414T141509_"
            "053431_067B51_2A05_COG"
        ),
    },
    "after": {
        "name": (
            "S1A_IW_GRDH_1SDV_20240426T141445_20240426T141510_"
            "053606_068232_779A.SAFE"
        ),
        "date": "2024-04-26",
        "wrapper": (
            "S1A_IW_GRDH_1SDV_20240426T141445_20240426T141510_"
            "053606_068232_9DE6_COG"
        ),
    },
}


def write_safe(
    path: Path,
    acquisition_date: str,
    *,
    relative_orbit: int = 159,
    orbit_pass: str = "ASCENDING",
    polarizations=("VV", "VH"),
) -> None:
    path.mkdir(parents=True)
    (path / "measurement").mkdir()
    for polarization in polarizations:
        lower = polarization.lower()
        (path / "measurement" / f"synthetic-{lower}-scene.tiff").write_bytes(
            lower.encode("ascii")
        )
    start = f"{acquisition_date}T14:14:44.000000Z"
    stop = f"{acquisition_date}T14:15:10.000000Z"
    (path / "manifest.safe").write_text(
        (
            "<root>"
            "<productType>GRD</productType>"
            "<mode>IW</mode>"
            f"<startTime>{start}</startTime>"
            f"<stopTime>{stop}</stopTime>"
            f"<pass>{orbit_pass}</pass>"
            f'<relativeOrbitNumber type="start">{relative_orbit}'
            "</relativeOrbitNumber>"
            + "".join(
                "<transmitterReceiverPolarisation>"
                f"{polarization}</transmitterReceiverPolarisation>"
                for polarization in polarizations
            )
            + "</root>"
        ),
        encoding="utf-8",
    )


def write_role_safe(root: Path, role: str, **kwargs) -> Path:
    spec = PRODUCTS[role]
    path = root / spec["name"]
    write_safe(path, spec["date"], **kwargs)
    return path


def link_products(safe_root: Path, role: str, target: Path) -> Path:
    products_dir = safe_root / "products"
    products_dir.mkdir(parents=True, exist_ok=True)
    link = products_dir / PRODUCTS[role]["name"]
    link.symlink_to(
        Path("..") / target.relative_to(safe_root),
        target_is_directory=True,
    )
    return link


class DiscoverKulsaryGrdProductsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.safe_root = Path(self.tmp.name) / "restored_grd"
        self.safe_root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_mixed_tree_prefers_nested_products_targets(self):
        expected = {}
        for role, spec in PRODUCTS.items():
            top_level = write_role_safe(self.safe_root, role)
            nested = self.safe_root / spec["wrapper"] / spec["name"]
            write_safe(nested, spec["date"])
            link_products(self.safe_root, role, nested)
            expected[role] = nested.resolve()
            self.assertNotEqual(top_level.resolve(), expected[role])

        predicted = self.safe_root / "predicted"
        write_safe(
            predicted / PRODUCTS["before"]["name"].replace("_5249.SAFE", "_PRED.SAFE"),
            PRODUCTS["before"]["date"],
        )
        logs = self.safe_root / "logs"
        logs.mkdir()
        (logs / "restore.log").write_text("ignored\n", encoding="utf-8")

        products = discover_kulsary_grd_products(self.safe_root)
        self.assertEqual(set(products), {"before", "peak", "after"})
        self.assertEqual(
            {role: product.root for role, product in products.items()},
            expected,
        )

    def test_top_level_only(self):
        expected = {
            role: write_role_safe(self.safe_root, role).resolve()
            for role in PRODUCTS
        }
        products = discover_kulsary_grd_products(self.safe_root)
        self.assertEqual(set(products), {"before", "peak", "after"})
        self.assertEqual(
            {role: product.root for role, product in products.items()},
            expected,
        )

    def test_products_only(self):
        expected = {}
        for role, spec in PRODUCTS.items():
            nested = self.safe_root / spec["wrapper"] / spec["name"]
            write_safe(nested, spec["date"])
            link_products(self.safe_root, role, nested)
            expected[role] = nested.resolve()

        products = discover_kulsary_grd_products(self.safe_root)
        self.assertEqual(
            {role: product.root for role, product in products.items()},
            expected,
        )

    def test_duplicate_same_identifier_without_products_is_deterministic(self):
        copies = {}
        for role, spec in PRODUCTS.items():
            if role != "before":
                write_role_safe(self.safe_root, role)
                continue
            copy_a = self.safe_root / "copy_a" / spec["name"]
            copy_b = self.safe_root / "copy_b" / spec["name"]
            write_safe(copy_a, spec["date"])
            write_safe(copy_b, spec["date"])
            copies["a"] = copy_a.resolve()
            copies["b"] = copy_b.resolve()
            (self.safe_root / spec["name"]).symlink_to(
                Path("copy_a") / spec["name"],
                target_is_directory=True,
            )
            (self.safe_root / spec["name"].replace("_5249.SAFE", "_AAAA.SAFE")).symlink_to(
                Path("copy_b") / spec["name"],
                target_is_directory=True,
            )

        products = discover_kulsary_grd_products(self.safe_root)
        self.assertEqual(
            products["before"].root,
            min(copies.values(), key=str),
        )

    def test_true_cog_safe_is_filtered(self):
        expected = {
            role: write_role_safe(self.safe_root, role).resolve()
            for role in PRODUCTS
        }
        cog_name = PRODUCTS["before"]["name"].replace(
            "_5249.SAFE",
            "_D77F_COG.SAFE",
        )
        write_safe(self.safe_root / cog_name, PRODUCTS["before"]["date"])

        products = discover_kulsary_grd_products(self.safe_root)
        self.assertEqual(set(products), {"before", "peak", "after"})
        self.assertEqual(products["before"].root, expected["before"])
        self.assertNotIn("_COG.SAFE", products["before"].identifier)

    def test_broken_cog_safe_is_ignored_when_standard_exists(self):
        expected = {
            role: write_role_safe(self.safe_root, role).resolve()
            for role in PRODUCTS
        }
        broken = self.safe_root / PRODUCTS['before']['name'].replace(
            '_5249.SAFE',
            '_D77F_COG.SAFE',
        )
        broken.mkdir()
        products = discover_kulsary_grd_products(self.safe_root)
        self.assertEqual(products['before'].root, expected['before'])

    def test_invalid_products_copy_falls_back_to_valid_top_level(self):
        expected = {
            role: write_role_safe(self.safe_root, role).resolve()
            for role in PRODUCTS
        }
        products_dir = self.safe_root / 'products'
        products_dir.mkdir()
        broken = self.safe_root / 'broken-wrapper' / PRODUCTS['before']['name']
        broken.mkdir(parents=True)
        (products_dir / PRODUCTS['before']['name']).symlink_to(
            Path('..') / 'broken-wrapper' / PRODUCTS['before']['name'],
            target_is_directory=True,
        )
        products = discover_kulsary_grd_products(self.safe_root)
        self.assertEqual(products['before'].root, expected['before'])

    def test_all_cog_products_require_restored_standard_grd(self):
        for role, spec in PRODUCTS.items():
            cog_name = spec["name"].replace(".SAFE", "_COG.SAFE")
            write_safe(self.safe_root / cog_name, spec["date"])
        with self.assertRaises(InferenceError) as context:
            discover_kulsary_grd_products(self.safe_root)
        self.assertIn("restored standard GRD", str(context.exception))
        self.assertIn("restored_grd", str(context.exception))

    def test_two_different_identifiers_on_same_date_are_an_error(self):
        for role in PRODUCTS:
            write_role_safe(self.safe_root, role)
        extra = PRODUCTS["before"]["name"].replace("_5249.SAFE", "_AAAA.SAFE")
        write_safe(self.safe_root / extra, PRODUCTS["before"]["date"])

        with self.assertRaises(InferenceError) as context:
            discover_kulsary_grd_products(self.safe_root)
        self.assertIn("2024-04-02", str(context.exception))
        self.assertIn("found 2", str(context.exception))

    def test_geometry_mismatch_is_an_error(self):
        for role in PRODUCTS:
            if role == "peak":
                write_role_safe(self.safe_root, role, relative_orbit=160)
            else:
                write_role_safe(self.safe_root, role)

        with self.assertRaises(InferenceError) as context:
            discover_kulsary_grd_products(self.safe_root)
        self.assertIn("relative orbit", str(context.exception))


if __name__ == "__main__":
    unittest.main()
