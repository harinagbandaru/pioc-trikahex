"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from .constants import DEFAULT_T
from .dft_pes import DFTConfig
from .pipeline import RunConfig, run_dft, run_empirical, write_outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pioc", description="Al-OH + CO2 fragment calculator.")
    p.add_argument("--engine", choices=("dft", "empirical"), default="dft")
    p.add_argument("--temperature", type=float, default=DEFAULT_T)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="out")
    p.add_argument("--xc", default="PBE0")
    p.add_argument("--basis", default="def2-SVP")
    p.add_argument("--n-scan", type=int, default=12)
    p.add_argument("--no-thermo", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    cfg = RunConfig(seed=args.seed, temperature_k=args.temperature, engine=args.engine, outdir=args.outdir)
    if args.engine == "dft":
        payload = run_dft(
            cfg,
            DFTConfig(xc=args.xc, basis=args.basis, n_scan=args.n_scan,
                      temperature_k=args.temperature, do_thermo=not args.no_thermo),
        )
        print("=" * 68)
        print(f"DFT cluster  {payload['level']}  T={args.temperature:.2f} K")
        print(f"  r_min            {payload['r_min_A']:.3f} A")
        print(f"  dE raw           {payload['dE_raw_eV']:+.4f} eV")
        print(f"  dE counterpoise  {payload['dE_cp_eV']:+.4f} eV  ({payload['dE_cp_kj']:+.2f} kJ/mol)")
        print(f"  BSSE             {payload['bsse_eV']:+.4f} eV")
        print(f"  {payload['disclaimer']}")
    else:
        payload = run_empirical(cfg)
        print("Empirical Coulomb+LJ fragment (not DFT)")
        print(f"  V(A), V(B)   {payload['geometry']['V_A']:.4f}, {payload['geometry']['V_B']:.4f} eV")
        print(f"  dA_WHAM      {payload['pmf']['dA_eV']:+.4f} eV")
        print(f"  AIS ESS      {payload['ais']['ess']:.2f}")
    paths = write_outputs(payload, args.outdir, args.engine)
    print("  wrote", paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
