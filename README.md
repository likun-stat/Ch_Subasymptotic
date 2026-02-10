# Handbook of Statistics of Extremes: Subasymptotic models for spatial extremes 


This repository mainly contains code to **simulate** and **fit** a HW (Huser-Wadsworth randome scale-mixture [^1]) model for spatio-temporal extremes. This full Bayesian impolementation is based on the works in [^2] and [^3].

Key design choices:
- The fitting of the **max-stable model** is done through calling the `fitmaxstab()` function in the R package ` SpatialExtremes`.
- The fitting of the **max-infinite divisible mdoel** uses the implementation provided by Huser et al. in their paper [^4]. 
- Python orchestrates simulation and MCMC sampling.
- To facilitate computation time, we employ a **C++ shared library** (`RW_inte_cpp.so`) to provide fast numerical integration routines, called from Python via `ctypes`. Further, we utilize **MPI parallelism over time replicates** (each MPI rank handles one time index).

---

## Repository layout 

```
.
├── sampler.py                  # MPI sampler (special-case: constant parameters)
├── simulate_data.py            # Simulation script for RW mixture data
├── RW_inte.py                  # Python wrapper for RW CDF/PDF (loads .so via ctypes)
├── RW_inte_cpp.cpp             # C++ implementation of numerical integration
├── utilities.py                # helper functions (kernels, covariances, transforms, etc.)
├── exe.sh                      # example Slurm script (mpirun + sampler.py)
├── data/                       # PNW Temperature data sets
├── maxID_Code_Raphael/                    # max-ID model impelemted by Huser et al. (2021)
    ├── C_Code/                       
    ├── Data/
    ├── Figures/
    ├── Jobs/
    └── R_Code/
└── README.md
```

---

## 1) Why `mpi4py`?

### MPI in statistician terms
MPI is a **distributed-memory parallel** model. Think:
- You have `P` independent workers (“ranks”).
- Each rank runs the *same program*, but on different pieces of the workload.
- Ranks can communicate (broadcast/scatter/gather), but can also run mostly independently.

In this implementation, the sampler is naturally parallel over time replicates:
- If your data is `Y[s, t]` with `t = 1, …, Nt`,
- then **rank `r` handles time `t = r`** (or a mapping like that).
- This gives clean parallelism with minimal cross-rank communication.

### Why MPI is a better fit than `doParallel` / `doMPI` (R)
For this workflow, MPI + Python is typically easier and more robust because:
1. **Cluster-native**: most HPC systems already support OpenMPI / MPICH and Slurm integration.
2. **Better control of memory and process placement** (important for large spatial matrices).
3. **Plays nicely with compiled code**:
   - Here we call a compiled C++ shared library via **Python `ctypes`**.
   - MPI ranks are OS processes, so each rank can safely call into the `.so` without dealing with R’s serialization model or fork/PSOCK complications.
4. **Cleaner scaling across nodes** if you later move beyond a single node.

---

## 2) Install dependencies (including `mpi4py`)

### Core dependencies
- Python 3.9+ recommended
- `numpy`, `scipy`
- `mpi4py`
- A working MPI implementation (OpenMPI or MPICH)
- GSL (GNU Scientific Library) for the C++ integrator

### Example: conda environment (recommended on clusters)
```bash
conda create -n hwmixture python=3.11 -y
conda activate hwmixture

# MPI stack + mpi4py (conda-forge usually works best)
conda install -c conda-forge numpy scipy mpi4py openmpi gsl -y
```

### Verify `mpi4py`
```bash
python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

If you get compiler / MPI errors during installation, it usually means your environment
does not have `mpicc` available or MPI modules are not loaded. On many clusters you should:
```bash
module load gcc openmpi
```
before installing / running.

---

## 3) Compile the C++ numerical integration library

The file `RW_inte_cpp.cpp` is compiled into a shared library `RW_inte_cpp.so`.
You **must compile this on your own system/cluster**, because compiler + library paths vary.

### Example compile command (macOS/Homebrew-style paths)
```bash
g++ -I/opt/homebrew/include -std=c++11 -Wall -pedantic RW_inte_cpp.cpp \
  -shared -fPIC -L/opt/homebrew/lib -o RW_inte_cpp.so -lgsl -lgslcblas
```

### Typical HPC/Linux compile command (module-based)
On clusters, you often do:
```bash
module load gcc
module load gsl        # if available
# or ensure gsl is installed and on your include/lib path

g++ -I$(gsl-config --prefix)/include  -std=c++11 -Wall -pedantic RW_inte_cpp.cpp -shared -fPIC  -L$(gsl-config --prefix)/lib -o RW_inte_cpp.so -lgsl -lgslcblas
```

### Sanity check
After compiling, you should see:
```bash
ls -lh RW_inte_cpp.so
```

> **Important**: `RW_inte.py` expects `RW_inte_cpp.so` to be discoverable (commonly in the same directory).
> If you reorganize folders (e.g., `src/`), make sure the Python wrapper loads the correct path.

---

## 4) Simulate data from the RW mixture model

Run the simulation script to generate:
- `Y` (typically shape `(Ns, Nt)`)
- `sites_xy` (shape `(Ns, 2)`)
- plus any other cached intermediate arrays you choose to save

Example:
```bash
python simulate_data.py
```

This should produce at minimum:
- `sites_xy.npy`
- a `Y_*.npy` file (your script may encode settings in the filename)

---

## 5) Run the MPI sampler via Slurm

### Conceptual rule: **#MPI ranks = #time replicates**
The sampler is written so that **each MPI rank handles one time replicate**.
Therefore you should set:
- `mpirun -n Nt python sampler.py`

If your dataset has `Nt = 50` time replicates, run with `-n 50`.

### Example Slurm script (`exe.sh`)
A typical Slurm script (single node, 50 MPI tasks) looks like:

```bash
#SBATCH -N 1
#SBATCH --tasks-per-node=50
...
module load gcc
module load openmpi
conda activate hwmixture
...
mpirun -n 50 python -u sampler.py
```

To submit:
```bash
sbatch exe.sh
```

### Notes on your provided `exe.sh`
- You already set:
  - `#SBATCH --tasks-per-node=50`
  - `mpirun -n 50 python -u sampler.py`
- That’s consistent with `Nt = 50`.

---

## Inputs/Outputs expected by `sampler.py`

### Required `.npy` inputs
A fresh run needs:
1. **Data array**: `Y_*.npy` (shape `(Ns, Nt)`)
2. **Coordinates**: `sites_xy.npy` (shape `(Ns, 2)`)

> In the current script, the `Y` filename may be hard-coded. For a public repo, consider making it a variable
> at the top of `sampler.py` or a command-line argument.

### Optional restart `.npy` files
If you resume a chain (`start_iter != 1`), the sampler may load trace files such as:
- `Y_trace.npy`, `loglik_trace.npy`, `R_trace_log.npy`, etc.
- `GEV_trace.npy` (if `UPDATE_GEV=True`)

---


## Troubleshooting

### `ImportError: libmpi.so` or MPI runtime errors
- Ensure the MPI module is loaded at runtime:
  ```bash
  module load openmpi
  ```
- Make sure `mpi4py` was installed against the same MPI you are loading.

### `OSError: RW_inte_cpp.so not found`
- Confirm the `.so` exists and is in the expected directory.
- If needed, set `LD_LIBRARY_PATH` or adjust the path in `RW_inte.py`.

### GSL not found during compile
- Load the cluster’s GSL module (if available), or install GSL via conda-forge.
- Make sure include/lib paths are correct.

---

---
## References
[^1]: Huser, R., & Wadsworth, J. L. (2019). Modeling spatial processes with unknown extremal dependence class. Journal of the American statistical association, 114(525), 434-444.
[^2]: Zhang, L., Shaby, B. A., & Wadsworth, J. L. (2022). Hierarchical transformed scale mixtures for flexible modeling of spatial extremes on datasets with many locations. Journal of the American Statistical Association, 117(539), 1357-1369.
[^3]: Shi, M., Zhang, L., Shaby, B. A. and Risser, M. D. (2026), ‘Spatial Scale-aware Tail Dependence Modeling for High-dimensional Spatial Extremes’. Journal of the American Statistical Association, doi: 10.1080/01621459.2026.2627493.
[^4]: Huser, R., Opitz, T., & Thibaud, E. (2021). Max‐infinitely divisible models and inference for spatial extremes. Scandinavian Journal of Statistics, 48(1), 321-348.
