# Fluctuating_Environments
Code supplement for "Rapid evolution can select for fitness tradeoffs in fluctuating environments" (doi.org/10.64898/2026.09.01.748641).
See data repository on Zenodo (doi:10.5281/zenodo.22236835) for precomputed simulation results.

SIMULATION CALLER:
sim_run.py

Main code that runs simulated Wright-Fisher evolution in a fluctuating environment.

SIMULATION HELPERS:
simulator.py,
ecosystem.py,
environment.py,
population.py

  Helper classes invoked by sim_run. simulator.py specifies the output files and ecosystem.py specifies most of the evolutionary dynamics.

FIGURE GENERATION CODE:
Figure_Code_Main.ipynb

Jupyter notebook which generates main text figures using precomputed simulation results on Zenodo.
