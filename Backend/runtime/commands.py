"""Build CLI argv for infer subprocess."""

from pathlib import Path


def build_infer_command(profile: str, artifacts: Path) -> list[str]:
    """Return argv for weapon_ai.cli.infer_thermal_objects."""
    return [
        "python",
        "-m",
        "weapon_ai.cli.infer_thermal_objects",
        "--profile",
        profile,
        "--artifacts",
        str(artifacts),
    ]
