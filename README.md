# **Two-networks-coherence-simulations-analysis**

This repository contains the full codebase used to generate the simulations, perform the analyses, and produce all figures for the paper:

***“Cortical excitability inversely modulates fMRI connectivity”***

The repository is organized into **three main components**:

### **1. Simulations**

Code to reproduce the neural network simulations described in the manuscript, including model setup, parameter manipulation, and dynamical integration for the LIF model.

### **2. Analysis**

Scripts to process the simulated data, compute coherence and connectivity metrics, extract spectral features, and run all statistical analyses presented in the paper.

### **3. Figure Generation**

Scripts to generate all figures from the main manuscript and supplementary material.

---

## **Repository Structure**

```
Two-networks-coherence-simulations-analysis/
│
├── LIF_network/
│   ├── analysis/               # Scripts to run LIF simulations and save results
│   └── neuron_model/           # NEST extension module (NMDA-enabled neuron model)
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

The NMDA-enabled neuron model must be compiled before running simulations.
The procedure follows the NEST tutorial *“Writing an extension module”*.

### **1 — Set the NEST installation directory**

```bash
export NEST_INSTALL_DIR=/Users/gabriele/NEST/ins
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

To reproduce the results reported in the paper, adjust the simulation parameters according to the manipulations described in the **Methods** section of the manuscript (e.g., changes in excitability, coupling strength, input levels, or synaptic parameters).

To run a simulation:

```bash
python3 Network_simulation.py
```

Each simulation outputs:

* the **LFP signal** of the two cortical areas (computed exactly as in the Methods section), and
* the **population firing-rate traces** of the two excitatory populations.

All outputs are automatically saved in the corresponding results directory in **CSV format**, allowing for easy loading, inspection, and further analysis.

---

### Coherence and Firing Rate Analysis**

## **Coherence Analysis**
Once you have simulated the network, you can compute the coherence by running:

```matlab
analysis_coherence.m
```

in MATLAB. Make sure to adjust the file paths to point to the correct input CSV files.

The script processes the firing-rate and LFP data and returns a MATLAB `.mat` file containing all the results, ready for plotting or further analysis.

---

## **Firing Rate Analysis**

Once the network has been simulated, you can process the firing-rate traces using:

```matlab
analysis_firing_rate.m
```

in MATLAB.

**Instructions:**

1. Make sure the script points to the correct CSV input files from the simulations and adjust for number of trials:

   * `firing_rate_trace_trial1.csv` … `firing_rate_trace_trial5.csv` for **Control**
   * `firing_rate_trace_trial1.csv` … `firing_rate_trace_trial5.csv` for **Manipulation**

2. Run the script in MATLAB.

3. The script will:

   * Chunk and average the firing-rate traces,
   * Compute baseline-normalized firing rates,
   * Save a MATLAB `.mat` file containing:

     * `data`: raw firing-rate chunks
     * `data_z`: baseline-normalized firing-rate chunks

This `.mat` file is ready for downstream analysis or plotting.


---

# **Reproducibility Notes**

* All simulations can be reproduced using the default parameters.
* Random seeds are fixed inside the scripts when relevant.

---

# **Citation**

If you use this code, please cite:

***“Cortical excitability inversely modulates fMRI connectivity”***
*Authors, Year*
*PLOS Computational Biology*

(Insert citation once DOI is available.)

---

# **License**

This repository is released under the **MIT License**, allowing broad reuse with attribution.

---

# **Contact**

For questions, issues, or suggestions:

**Gabriele Mancini**
Email: **[mancinigabriele814@gmail.com](mailto:mancinigabriele814@gmail.com)**

---

If you want, I can now:

✅ generate a `requirements.txt`
✅ generate a `CITATION.cff`
✅ fix internal folder descriptions or update according to actual repo content
Just tell me!
