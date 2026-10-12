# Uncertainty-Aware Visualization for Multi-Criteria Machine Selection

Interactive dashboard combining TOPSIS ranking, Monte Carlo uncertainty
propagation (100,000 runs) and five coordinated visualizations, applied
to the selection of laser powder bed fusion (LPBF) machines.

Companion code for the paper submitted to the 37th CIRP Design Conference
(CIRP Design 2027).

## Installation
pip install -r requirements.txt

## Run
python app.py
Then open http://127.0.0.1:8051 in your browser.

## Data
The decision matrix (criteria c1–c7, 9 alternatives) and AHP weights come
from Tomelleri et al. (2025), Int. J. Adv. Manuf. Technol. 139, 3871–3890.
Criteria c8 (surface roughness) and c9 (porosity) were added for this work
to illustrate aleatoric uncertainty.

## Authors
Hajar El Alami, Laurent Pierre, Nabil Anwer — LURPA, ENS Paris-Saclay
