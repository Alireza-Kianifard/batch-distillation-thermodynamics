# Batch Distillation and VLE Thermodynamic Modeling
A Python-based thermodynamic and dynamic simulation package for multi-component vapor-liquid equilibrium (VLE) calculations and dynamic batch distillation columns.

The project models phase equilibria using the Peng-Robinson Equation of State (PR-EOS) for the vapor phase and the NRTL model for liquid activity coefficients. All core routines have been validated against Aspen HYSYS simulation results.

## Project Overview
The project was developed in three sequential phases:

### Phase 1: VLE Flash Calculation and HYSYS Benchmarking
* Solves the non-linear Rachford-Rice equation coupled with PR-EOS and NRTL activity models.
* Computes vapor phase fraction and equilibrium phase compositions.
* Validated directly against steady-state flash operations in Aspen HYSYS.

### Phase 2: Bubble Point Temperature Calculation
* Implements iterative convergence routines to compute the exact bubble point temperature at specified system pressures.
* Uses the Wilson correlation for initial estimates, followed by nested loops for liquid activity and vapor fugacity corrections.

### Phase 3: Dynamic Batch Distillation Simulation
* Models a transient batch distillation column over time.
* Tracks reboiler composition, dynamic distillate purity profiles, and pot temperature changes throughout the operation.
* Exports simulation time-series data to `distillation_results.csv`.

## Thermodynamic Modeling Notes
* **Phase Equilibria:** Peng-Robinson EOS is utilized for vapor fugacity coefficients, and the NRTL model handles liquid-phase non-idealities.
* **Liquid Activity Correction:** To reconcile differences in binary interaction parameter sets and ensure close alignment with Aspen HYSYS equilibrium data, a calibration factor is applied directly to the liquid activity coefficient of ethanol during calculation.

## Validation Against Aspen HYSYS
Model outputs across flash and bubble point calculations were benchmarked against Aspen HYSYS. 

* The comparison data is available in `HYSYS_comparison.xlsx`.
* The overall relative error across phase fractions and equilibrium temperatures remains below 0.5%.

## Repository Structure
```text
batch-distillation-thermo/
├── .gitignore
├── README.md
├── requirements.txt
├── HYSYS_comparison.xlsx
├── main.py
├── data/
│   └── distillation_results.csv
└── batch_distillation/
├── core/
│   ├── activity_nrtl.py
│   ├── eos_pr.py
│   └── enthalpy.py
├── models/
│   ├── component.py
│   └── mixture.py
├── solvers/
│   ├── rachford_rice.py
│   ├── bubble_point.py
│   └── wilson.py
└── simulation/
└── batch_column.py

## Requirements and Installation

### Prerequisites
* Python 3.9 or higher
* NumPy

### Setup
1. Clone the repository:
bash
git clone https://github.com/Alireza-Kianifard/batch-distillation-thermodynamics.git
cd batch-distillation-thermo

2. Install required packages:
bash
pip install -r requirements.txt

3. Run the simulation:
bash
python main.py
