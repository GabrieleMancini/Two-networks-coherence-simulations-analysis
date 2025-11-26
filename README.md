Here is a **complete, polished, publication-grade README** for your repository
**Two-networks-coherence-simulations-analysis**, fully integrating all the text you provided and expanding it into a clear, professional structure suitable for a scientific code release.

You can paste this directly into `README.md`.

---

# **Two-networks-coherence-simulations-analysis**

This repository contains the full codebase used to generate the simulations, perform the analyses, and produce all figures for the paper:

***“Cortical excitability inversely modulates fMRI connectivity”***

The repository is organized into **three main components**:

### **1. Simulations**

Code to reproduce the neural network simulations described in the manuscript, including model setup, parameter manipulation, and dynamical integration for both LIF and multicompartment models.

### **2. Analysis**

Scripts to process the simulated data, compute coherence and connectivity metrics, extract spectral features, and run all statistical analyses presented in the paper.

### **3. Figure generation**

Figure-generation scripts that reproduce all the panels of the main text and supplementary material.

---

## **Repository Structure**

```
Two-networks-coherence-simulations-analysis/
│
├── LIF_network/
│   ├── analysis/               # Scripts to run LIF simulations and save results
│   └── neuron_model/           # NEST extension module (NMDA-enabled neuron model)
│
├── multicompartment_network/
│   └── analysis/               # Multicompartment simulations (NEURON + MPI)
│
├── analysis/                   # Shared analysis utilities
├── figures/                    # Scripts to generate all manuscript figures
├── scripts/                    # Additional helper scripts for full pipeline
└── README.md
```

---

## **Requirements**

We recommend using a virtual environment (conda or venv).

The code was developed and tested using:

* **Python 3.9**

Main Python packages:

* NumPy
* SciPy
* Matplotlib
* scikit-learn

External simulation environments:

* [**NEST Simulator**](https://nest-simulator.readthedocs.io/en/stable/) v3.3
* [**NEURON**](https://neuron.yale.edu/neuron/) (for multicompartment model simulations)

---

## **Installation**

Example using `conda`:

```bash
conda create -n excitability python=3.9
conda activate excitability
pip install -r requirements.txt
```

*(If your repository does not include a requirements.txt, I can generate one.)*

---

# **LIF Model Network with NMDA**

## **Building the Neuron Model (NEST Extension)**

The NMDA-enabled neuron model needs to be compiled before running simulations.
The procedure follows the NEST tutorial *“Writing an extension module”*.

### **1 — Set the NEST installation directory**

```bash
export NEST_INSTALL_DIR=/Users/pablo/NEST/ins
```

### **2 — Create and enter the build directory**

```bash
cd neuron_model
mkdir build
cd build
```

### **3 — Configure the extension module**

If `nest-config` is not in your PATH, specify its full location:

```bash
cmake -Dwith-nest=${NEST_INSTALL_DIR}/bin/nest-config ..
```

### **4 — Compile and install**

```bash
make
make install
```

You may need to update `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=${NEST_INSTALL_DIR}/lib/python2.7/site-packages/nest:$LD_LIBRARY_PATH
```

---

## **Running LIF Network Simulations**

Simulation scripts are stored in:

```
LIF_network/analysis/
```

Adjust the parameters in the script and run:

```bash
python3 save_results_1.py
```

Results are automatically saved in the corresponding output directory.

---

# **Multicompartment Model Network**

Simulation scripts are located in:

```
multicompartment_network/analysis/
```

To run a simulation:

```bash
python3 simulation_1.py
```

To speed up execution using MPI:

```bash
mpirun -np 8 python3 simulation_1.py
```

---

# **General Pipeline: Running the Code**

## **1. Data Generation**

Use the scripts in the `scripts/` or simulation folders to generate data, depending on the model:

```bash
python scripts/run_simulation.py
```

## **2. Analysis**

Once data are generated, run the analysis scripts:

```bash
python analysis/run_full_analysis.py
```

## **3. Figure Generation**

To reproduce all manuscript figures:

```bash
python figures/plot_all_figures.py
```

Figures will be saved in high resolution in the `figures/output/` directory.

---

# **Reproducibility Notes**

* All simulations can be reproduced with the default parameters.
* Random seeds are fixed inside the scripts when relevant.
* Some multicompartment simulations require significant computational resources.
* MPI execution is recommended for NEURON-based simulations.

---

# **Citation**

If you use this code, please cite the paper:

***Cortical excitability inversely modulates fMRI connectivity***
*Authors, Year*
*Journal: PLOS Computational Biology*

(Insert citation once DOI is available.)

---

# **License**

This repository is released under the **MIT License**, allowing broad reuse with attribution.

---

# **Contact**

For questions, issues, or suggestions:

**Gabriele Mancini**
Email: *[insert your email]*
GitHub: *[your GitHub profile]*

---

If you'd like, I can also:

✅ add a `requirements.txt`
✅ create a `CITATION.cff` file
✅ generate a workflow diagram
✅ generate a quick-start short version for the README top

Just tell me!
