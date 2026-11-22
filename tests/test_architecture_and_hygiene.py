"""Architecture boundary, static hygiene, and profile configuration tests."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

from quadvox.cli import main as cli_main
from quadvox.contracts import load_profile_config

REPO_ROOT = Path(__file__).resolve().parents[1]
_SYS_TEMP = os.path.sep + "tmp"
_USER_HOME_PREFIX = os.path.sep + "var" + os.path.sep + "home" + os.path.sep + "jane"


def test_zero_hardcoded_tmp_and_home_paths() -> None:
    py_files = [
        p
        for folder in ("quadvox", "runners", "tests")
        for p in sorted((REPO_ROOT / folder).rglob("*.py"))
        if "__pycache__" not in p.parts
    ]
    assert py_files
    for p in py_files:
        text = p.read_text(encoding="utf-8")
        assert _USER_HOME_PREFIX not in text, f"Found hardcoded {_USER_HOME_PREFIX} in {p}"
        tree = ast.parse(text, filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.strip()
                assert not (val == _SYS_TEMP or val.startswith(_SYS_TEMP + os.path.sep)), (
                    f"Hardcoded system temp literal {val!r} in {p}:{getattr(node, 'lineno', 0)}"
                )


def test_layer1_and_layer2_isolation_no_layer3_hardcoding() -> None:
    forbidden = ("xiaokui", "hairdeck", "vrm_addon", "mtoon", "xclip", "sailor_collar")
    target_files = [
        p
        for folder in ("quadvox", "configs", "runners")
        for p in sorted((REPO_ROOT / folder).rglob("*"))
        if p.is_file() and p.suffix in (".py", ".yaml", ".yml") and "__pycache__" not in p.parts
    ]
    for p in target_files:
        low = p.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in low, f"Layer-3 domain token {token!r} leaked into {p}"


def test_profiles_layer1_defaults_off_and_layer2_opt_in() -> None:
    obj = load_profile_config("object")
    assert obj.asset_type == "prop"
    assert obj.roi_boxes() == []
    assert obj.coarse_splice().enabled is False
    assert obj.s2["symmetry_x"] is False
    assert obj.s3["roi_sovereignty"] is False

    prop = load_profile_config("hard_surface_prop")
    assert prop.s1["repair"]["cellocut_qef"] is True
    assert prop.s2["retopo_method"] == "lato2"
    assert prop.s2["symmetry_x"] is False
    assert prop.s3["pbr_recipe"] == "hard_surface_metal"

    hum = load_profile_config("symmetric_humanoid")
    assert hum.asset_type == "character"
    assert len(hum.roi_boxes()) == 1
    assert hum.roi_boxes()[0].target_view == "front"
    assert hum.coarse_splice().enabled is True
    assert hum.s2["symmetry_x"] is True
    assert hum.s3["roi_sovereignty"] is True

    hroi = load_profile_config("hard_surface_roi")
    assert hroi.asset_type == "prop"
    assert len(hroi.roi_boxes()) == 1
    assert hroi.s2["symmetry_x"] is True
    assert hroi.s3["roi_sovereignty"] is True


def test_cli_profile_show_smoke(capsys) -> None:
    rc = cli_main(["profile-show", "--profile", "hard_surface_roi"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["name"] == "hard_surface_roi"
    assert out["s1"]["repair"]["cellocut_qef"] is True


def test_no_stale_s1_backend_or_legacy_runner_invocations() -> None:
    for folder in ("quadvox", "runners"):
        for p in sorted((REPO_ROOT / folder).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    val = node.value
                    assert "s1_backend." not in val, f"Stale s1_backend reference in {p}:{getattr(node, 'lineno', 0)}"
                    assert "colab_remote_runner.py" not in val, (
                        f"Stale colab_remote_runner.py reference in {p}:{getattr(node, 'lineno', 0)}"
                    )


def test_git_does_not_ignore_quadvox_tools_package() -> None:
    import subprocess

    for rel in ("quadvox/tools/__init__.py", "quadvox/tools/external_tools.py"):
        target = REPO_ROOT / rel
        assert target.is_file(), f"Missing core module {target}"
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", rel],
            capture_output=True,
        )
        assert proc.returncode != 0, f".gitignore must NOT ignore tracked package file {rel}"
