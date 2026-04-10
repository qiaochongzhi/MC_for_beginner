# MC for Beginner

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/Python-58.9%25-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-37.4%25-00599C?logo=c%2B%2B&logoColor=white)

A beginner-friendly **Monte Carlo (MC) simulation** code for molecular systems, implemented in both **Python** and **C++** (with Python bindings via [pybind11](https://github.com/pybind/pybind11)).

---

## ✨ Features

- **NVT (Canonical) Monte Carlo** simulations
- Two sampling algorithms:
  - **Metropolis** algorithm
  - **Geometric Cluster Algorithm (GCA)**
- **Lennard-Jones** potential with virial computation
- **Neighbour list** (link-cell list) for efficient pair searching
- **Python interface** for the high-performance C++ core (via pybind11)
- Pure **Python implementation** for learning and prototyping
- **MBWR Equation of State** module for reference thermodynamic properties
- XYZ trajectory file output

---

## 📁 Project Structure

```
MC_for_beginner/
├── src/
│   ├── MonteCarlo.py       # Pure Python MC implementation
│   ├── MonteCarlo.cpp      # High-performance C++ MC core
│   ├── MonteCarlo.h        # C++ header
│   ├── bind.cpp            # pybind11 bindings
│   ├── linkList.cpp/.h     # Neighbour link-cell list
│   ├── PotentialType.py    # Python potential data class
│   ├── PotentialType.h     # C++ potential data structure
│   ├── Randnumber.h        # Random number utilities
│   ├── MBWR_EOS.py         # Modified Benedict–Webb–Rubin EOS
│   └── CMakeLists.txt      # CMake build configuration
├── examples/
│   └── MC_NVT.ipynb        # Jupyter Notebook demo (NVT simulation)
├── data/                   # Input/output data files
├── tests/                  # Unit tests
├── .gitignore
└── LICENSE                 # GNU GPL v3
```

---

## 🚀 Quick Start

### Option 1 — Pure Python (no compilation needed)

```python
from src.MonteCarlo import MonteCarlo

mc = MonteCarlo(nParticles=256, dim=3, temperature=1.0, rCut=2.5, isNeighbourList=True)
mc.InitPosition()
results = mc.NVTrun(nStep=10000, drMax=0.1, interval=100)
```

### Option 2 — Python + C++ backend (recommended for performance)

#### 1. Build the C++ extension

```bash
# Install pybind11 first
pip install pybind11

cd src
mkdir build && cd build
cmake ..
make
```

> **Note:** You may need to adjust `PYTHON_EXECUTABLE` and `PYTHON_INCLUDE_DIRECTORY` in `src/CMakeLists.txt` to match your Python installation.

#### 2. Run a simulation

```python
import sys
sys.path.append("src/build")
import MonteCarlo as MC

mc = MC.MonteCarlo(nParticles=256, dim=3, temperature=1.0, rCut=2.5, isNeighbourList=True)
mc.InitPosition()
results = mc.NVTrun(nStep=10000, drMax=0.1, interval=100)
```

---

## 📓 Examples

Open the Jupyter Notebook for a step-by-step walkthrough:

```bash
jupyter notebook examples/MC_NVT.ipynb
```

The notebook covers:
- System setup and initialisation
- Running NVT MC with the Metropolis algorithm
- Visualising energy and pressure convergence
- Comparing Python and C++ implementations

---

## 📦 Dependencies

| Dependency | Purpose |
|---|---|
| Python ≥ 3.8 | Core language |
| NumPy | Array operations |
| pybind11 | C++/Python bindings |
| CMake ≥ 3.10 | C++ build system |
| C++17 compiler | C++ compilation (GCC / Clang / MSVC) |
| Jupyter | Running example notebooks |

Install Python dependencies:

```bash
pip install numpy pybind11 jupyter
```

---

## 🧪 Simulation Methods

### Metropolis Algorithm
The standard Metropolis MC scheme: particles are displaced randomly, and the move is accepted or rejected based on the Boltzmann criterion.

### Geometric Cluster Algorithm (GCA)
An advanced cluster-move algorithm that improves sampling efficiency, particularly useful near phase transitions.

---

## 🔬 Physical Model

- **Potential:** Lennard-Jones (LJ) 12-6
- **Truncation:** Spherical cutoff at `r_cut = 2.5σ` (configurable)
- **Overlap detection:** Hard-core overlap check (`pot > 100`)
- **Observables:** Potential energy, virial (pressure), acceptance ratio

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE) for details.

---

## 🙋 Contributing & Feedback

This project is designed for **beginners learning Monte Carlo simulation**. Suggestions, bug reports, and contributions are welcome — feel free to open an [Issue](https://github.com/qiaochongzhi/MC_for_beginner/issues) or submit a Pull Request!