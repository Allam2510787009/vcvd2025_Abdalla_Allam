# Pacejka Magic Formula – Tyre Model

**Vehicle Components & Vehicle Dynamics — Coding Assignment WS2025**

---

## Author

- **Name:** Abdalla Amr Abdelrahman Allam
- **Student ID:** 2510787009

---

## Overview

This project implements the **Pacejka Magic Formula** tyre model as described in:

> [1] Bakker, Egbert and Nyborg, Lars and Pacejka, Hans B.,
> "Tyre Modelling for Use in Vehicle Dynamics Studies",
> SAE Transactions, 1987

It computes and plots:

- **Fx** — Brake (longitudinal) force vs. longitudinal slip κ (0–100%)
- **Fy** — Side (lateral) force vs. longitudinal slip κ (0–100%)

The slip angle α is used as a curve parameter. The normal load Fz is derived from the vehicle mass assuming equal distribution over 4 wheels.

---

## Repository Structure

```
.
├── 2510787009.py            # Main entry point (rename to your student ID)
├── tyre_coefficients.json  # Tabular coefficients from [1], Table 2 & 3
└── README.md               # This file
```

---

## Requirements

- Python 3.9+
- `numpy`
- `matplotlib`
- `scipy`

---


### Sample Call Statements

```bash
# Reproduce Fig. 19 (α = 2°), 1500 kg vehicle, default friction
python 2510787009.py --slip 2 --weight 1500 --mu 0.8

# Reproduce Fig. 20 (α = 5°)
python 2510787009.py --slip 5 --weight 1500 --mu 0.8

# Heavy vehicle, low friction (wet road)
python 2510787009.py --slip 2 --weight 2000 --mu 0.4

# Light vehicle, high friction (race track)
python 2510787009.py --slip 3 --weight 1200 --mu 1.0
```

The script will display an interactive plot and save a `.png` file named:
```
magic_formula_slip<slip>_w<weight>_mu<mu>.png
```

---

## Formulas Implemented

### Side Force Fy (lateral)

```
Fy = D sin(C arctan(Bφ)) + ΔSv
φ  = (1-E)(α + ΔSh) + (E/B) arctan(B(α + ΔSh))
D  = a1 Fz² + a2 Fz
C  = 1.30
B  = a3 sin(2 arctan(Fz/a4)) · (1 - a5|γ|) / (CD)
E  = a6 Fz² + a7 Fz + a8
ΔSh = a9 γ
ΔSv = (a10 Fz² + a11 Fz) γ
```

### Brake Force Fx (longitudinal)

```
Fx = D sin(C arctan(Bφ))
φ  = (1-E) κ + (E/B) arctan(Bκ)
D  = a1 Fz² + a2 Fz
C  = 1.65
B  = (a3 Fz² + a4 Fz) / (CD e^(a5 Fz))
E  = a6 Fz² + a7 Fz + a8
```

Where Fz is in **kN** and κ is in **fraction** (0–1).

---

## Normal Load Derivation

```
Fz = (mass × g) / 4 wheels
```

Using `scipy.constants.g` for the gravitational acceleration.

---


## Reference

[1] Bakker, E., Nyborg, L., & Pacejka, H. B. (1987).
*Tyre Modelling for Use in Vehicle Dynamics Studies*.
SAE Transactions. https://doi.org/10.4271/870421
