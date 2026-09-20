# PIOC-TrikaHex architecture

Units: eV, Angstrom, fs, amu. T = 313.15 K. beta = 37.057379 eV^-1.

Two engines:
- dft: PySCF PBE0/def2-SVP Al(OH)3 + CO2 rigid O-down scan + Boys-Bernardi CP
- empirical: 6-site Coulomb+LJ, Fourier sine bridges, Path-AIS, WHAM

Last DFT well: r(Al-O) = 2.729 A, dE_raw = -0.1098 eV, dE_CP = -0.0265 eV.

See configs/all_parameters.json for every numeric parameter.
LIVE vs SPEC vs RETIRED is documented in the conversation writeup.
