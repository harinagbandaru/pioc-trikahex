# PIOC-TrikaHex

Fragment calculator for Al-OH / Al(OH)3 + CO2.

Two engines: `dft` (PySCF PBE0/def2-SVP Al(OH)3+CO2 scan + counterpoise) and `empirical` (Coulomb+LJ, Fourier bridges, Path-AIS, WHAM).

```bash
pip install -r requirements.txt
PYTHONPATH=. python3 run_pioc.py --engine dft --n-scan 8 --no-thermo
PYTHONPATH=. python3 run_pioc.py --engine empirical
PYTHONPATH=. python3 -m pytest tests -q
PYTHONPATH=. python3 agents/launch.py
```

See ARCHITECTURE.md and configs/all_parameters.json.
