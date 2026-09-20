#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pioc.framework import launch


def main() -> int:
    print("Launching PIOC-TrikaHex builder agents...")
    payload = launch()
    print(f"agents_run={payload['agents_run']}  all_ok={payload['all_ok']}")
    if payload["agents_failed"]:
        print("FAILED:", ", ".join(payload["agents_failed"]))
    for r in payload["reports"]:
        flag = "OK " if r.get("ok") else "ERR"
        extra = r.get("error", "")
        print(f"  [{flag}] {r.get('agent')}" + (f"  {extra}" if extra else ""))
    print("report:", payload["report_path"])
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
