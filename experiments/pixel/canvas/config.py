"""Settings, all from environment variables (see README for the table)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = HERE / "assets" / "target.png"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_hex(s: str) -> tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"bad colour {s!r}: want RRGGBB hex")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


@dataclass
class Settings:
    target: Path = DEFAULT_TARGET
    data_dir: Path | None = None
    # Colour that transparent (alpha=0) target pixels become, and the blank canvas colour.
    background: str = "000000"
    blank: str | None = None  # None = same as background
    background_is_work: bool = False  # 1 = background pixels must be painted too
    max_pixels_per_call: int = 10
    max_claim: int | None = None  # None = max_pixels_per_call
    claim_ttl: float = 120.0
    claim_mode: str = "advisory"  # advisory | enforced
    region_enforce: bool = False  # region leases gate paint + pixel claims
    region_ttl: float = 600.0
    region_ttl_max: float = 3600.0
    reveal_target_in_claim: bool = True
    rate_limit: float = 2.0  # write calls per second per agent; 0 = off
    rate_burst: float = 5.0
    token: str | None = None  # mode A shared secret
    admin_token: str | None = None  # mode B: mints per-agent tokens
    read_auth: bool = False
    keyframe_every: int = 1000
    keyframe_cache: int = 32
    snapshot_seconds: float = 30.0
    region_json_max: int = 4096

    def __post_init__(self) -> None:
        if self.max_claim is None:
            self.max_claim = self.max_pixels_per_call
        if self.max_pixels_per_call < 1 or self.max_claim < 1:
            raise ValueError("MAX_PIXELS_PER_CALL and CANVAS_MAX_CLAIM must be >= 1")
        if self.claim_mode not in ("advisory", "enforced"):
            raise ValueError("CANVAS_CLAIM_MODE must be advisory or enforced")

    @classmethod
    def from_env(cls) -> Settings:
        data = os.environ.get("CANVAS_DATA")
        return cls(
            target=Path(_env("CANVAS_TARGET", str(DEFAULT_TARGET))),
            data_dir=Path(data) if data else None,
            background=_env("CANVAS_BACKGROUND", "000000"),
            blank=os.environ.get("CANVAS_BLANK") or None,
            background_is_work=_env("CANVAS_BACKGROUND_IS_WORK", "0") == "1",
            max_pixels_per_call=int(_env("MAX_PIXELS_PER_CALL", "10")),
            max_claim=int(os.environ["CANVAS_MAX_CLAIM"])
            if os.environ.get("CANVAS_MAX_CLAIM")
            else None,
            claim_ttl=float(_env("CANVAS_CLAIM_TTL", "120")),
            claim_mode=_env("CANVAS_CLAIM_MODE", "advisory"),
            region_enforce=_env("REGION_ENFORCE", "0") == "1",
            region_ttl=float(_env("CANVAS_REGION_TTL", "600")),
            region_ttl_max=float(_env("CANVAS_REGION_TTL_MAX", "3600")),
            reveal_target_in_claim=_env("CANVAS_CLAIM_REVEAL", "1") == "1",
            rate_limit=float(_env("CANVAS_RATE_LIMIT", "2")),
            rate_burst=float(_env("CANVAS_RATE_BURST", "5")),
            token=os.environ.get("CANVAS_TOKEN") or None,
            admin_token=os.environ.get("CANVAS_ADMIN_TOKEN") or None,
            read_auth=_env("CANVAS_READ_AUTH", "0") == "1",
            keyframe_every=int(_env("CANVAS_KEYFRAME_EVERY", "1000")),
            keyframe_cache=int(_env("CANVAS_KEYFRAME_CACHE", "32")),
            snapshot_seconds=float(_env("CANVAS_SNAPSHOT_SECONDS", "30")),
        )
