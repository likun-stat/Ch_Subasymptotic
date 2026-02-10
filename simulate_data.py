"""
simulate_data.py

Simulate a stationary spatio-temporal dataset from the RW mixture model.

Model sketch (one time replicate):
    1) Latent Gaussian field:     Z(s) ~ GP(0, K_theta)  (Matérn/exponential here)
    2) Heavy-tail transform:      W(s) = g(Z(s))         (Normal -> Pareto transform)
    3) Positive random scale:     R(s) = sum_k w_k(s) S_k,   S_k ~ Lévy(alpha=1/2) (via rlevy)
    4) RW mixture (copula scale): X*(s) = R(s)^{phi(s)} * W(s)
    5) Marginal transform:        Y(s) = GEV^{-1}( F_RW( X*(s); phi(s), gamma(s) ) )

We compute F_RW (CDF) and its inverse via utility functions in `utilities.py`.
Outputs are saved as NumPy .npy arrays.

Usage:
    python3 simulate_data.py <seed>

Notes:
- This script is written as a runnable driver (not a library). For GitHub, it is
  typically used with a README that explains dependencies and the saved outputs.
- Requires SciPy/NumPy/Matplotlib/tqdm and the local `utilities.py`.
- `rpy2` is imported but not actively used in this script (kept for historical parity).

Last updated: 2025-07-17 (stationary configuration: k=1 knot; intercept-only marginals)
"""
# %%
import sys
import multiprocessing

import scipy
import numpy                as np
import matplotlib
import matplotlib.pyplot    as plt
from tqdm                   import tqdm
# import gstools              as gs
# Optional dependency: rpy2 was used in earlier versions of this project (e.g.,
# to mirror some R helper code). It is not required for the core simulation below;
# if you do not have rpy2 installed, you can safely remove these imports.
from rpy2.robjects          import r
from rpy2.robjects.numpy2ri import numpy2rpy
from rpy2.robjects.packages import importr

from utilities import *
print('Generating data using', norm_pareto, 'Pareto')

# ----------------------------
# Reproducibility / seeding
# ----------------------------
# The script accepts an optional integer seed (useful for batch runs on HPC).
# If not provided, we fall back to a fixed default for interactive testing.
try:
    data_seed = int(sys.argv[1])
except (IndexError, ValueError):
    data_seed = 2345
print('data_seed:', data_seed)
np.random.seed(data_seed)


# Use a conservative cap locally, but allow larger parallelism on big nodes.
N_CORES = 6 if multiprocessing.cpu_count() < 32 else 32

# %% Simulation Setup

# -----------------------------------------------------------------------------
# 1) Spatial/temporal domain and 'truth' parameters
# -----------------------------------------------------------------------------
# Spatial Domain Setup --------------------------------------------------------------------------------------------

# Numbers - Ns, Nt --------------------------------------------------------

np.random.seed(data_seed)
Nt = 50 # number of time replicates
Ns = 1079 # number of sites/stations
Time = np.linspace(-Nt/2, Nt/2-1, Nt)/np.std(np.linspace(-Nt/2, Nt/2-1, Nt), ddof=1)

# Model Parameter Setup (Truth) -------------------------------------------

N_outer_grid      = 1
radius            = 1000
radius_from_knots = np.repeat(radius, 1) # Wendland kernel influence radius from a knot
effective_range   = radius               # Gaussian kernel effective range: exp(-3) = 0.05
bandwidth         = effective_range**2/6 # range for the gaussian kernel

# Data Model Parameters - X_star = R^phi * g(Z) ---------------------------

range_at_knots = np.array([0.2])
phi_at_knots   = np.array([0.3])

# Marginal Parameters - GEV(mu, sigma, ksi) -------------------------------

mu_matrix    = np.full(shape = (Ns, Nt), fill_value = 40.0)
sigma_matrix = np.full(shape = (Ns, Nt), fill_value = 30.0)
ksi_matrix   = np.full(shape = (Ns, Nt), fill_value = 0.14)

# missing indicator matrix ------------------------------------------------

## random missing
miss_matrix = np.full(shape = (Ns, Nt), fill_value = 0)
for t in range(Nt):
    miss_matrix[:,t] = np.random.choice([0, 1], size=(Ns,), p=[1.0, 0.0])
miss_matrix = miss_matrix.astype(bool) # matrix of True/False indicating missing, True means missing

# Sites are laid out on a regular lon/lat grid (here, 0.125-degree resolution).
# We then drop a coarse sub-grid to create an irregular network of stations.
# Sites - random uniformly (x,y) generate site locations ------------------

x = np.arange(-111.1875, -106.4375 + 0.125, 0.125)   # 39 longitudes
y = np.arange(  34.4375,   38.1875 + 0.125, 0.125)   # 31 latitudes

# R’s expand.grid(x = x, y = y) varies x fastest and y slowest.
# We mimic that ordering with a nested comprehension and keep a NumPy array.
sitexy = np.array([(lon, lat) for lat in y for lon in x])  # shape: (39*31, 2)

# ------------------------------------------------------------------
# 2. Index grids and sub‑indices (identical logic to the R snippet)
# ------------------------------------------------------------------
lon_idx      = np.arange(1, 40)                       # 1 … 39
lat_idx      = np.arange(1, 32)                       # 1 … 31
sub_lon_idx  = np.round(np.linspace(1, 39, 13)).astype(int)  # 13 equally spaced longs
sub_lat_idx  = np.round(np.linspace(1, 31, 10)).astype(int)  # 10 equally spaced lats

# Build the (lon, lat) index pairs in the same order as sitexy
location_idx = np.array([(lx, ly) for ly in lat_idx for lx in lon_idx])  # (1209, 2)

# ------------------------------------------------------------------
# 3. Identify and drop the subset
# ------------------------------------------------------------------
mask = (np.isin(location_idx[:, 0], sub_lon_idx) &
        np.isin(location_idx[:, 1], sub_lat_idx))

sites_xy = sitexy[~mask]
sites_x = sites_xy[:,0]
sites_y = sites_xy[:,1]

# # define the lower and upper limits for x and y
minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))

# Elevation Function ------------------------------------------------------

# elev_surf_generator = gs.SRF(gs.Gaussian(dim=2, var = 1, len_scale = 2), seed=data_seed)
# elevations = elev_surf_generator((sites_x, sites_y))

# Knots define the low-rank spatial parameterization (k knots total).
# Each site receives weights from knots via either a Gaussian smoother or
# compactly-supported Wendland basis (used for the Lévy scale-mixture field).
# Knots locations w/ isometric grid ---------------------------------------

h_dist_between_knots     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid))-1)
v_dist_between_knots     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid))-1)
x_pos                    = np.linspace(minX + h_dist_between_knots/2, maxX + h_dist_between_knots/2,
                                        num = int(2*np.sqrt(N_outer_grid)))
y_pos                    = np.linspace(minY + v_dist_between_knots/2, maxY + v_dist_between_knots/2,
                                        num = int(2*np.sqrt(N_outer_grid)))
x_outer_pos              = x_pos[0::2]
x_inner_pos              = x_pos[1::2]
y_outer_pos              = y_pos[0::2]
y_inner_pos              = y_pos[1::2]
X_outer_pos, Y_outer_pos = np.meshgrid(x_outer_pos, y_outer_pos)
X_inner_pos, Y_inner_pos = np.meshgrid(x_inner_pos, y_inner_pos)
knots_outer_xy           = np.vstack([X_outer_pos.ravel(), Y_outer_pos.ravel()]).T
knots_inner_xy           = np.vstack([X_inner_pos.ravel(), Y_inner_pos.ravel()]).T
knots_xy                 = np.vstack((knots_outer_xy, knots_inner_xy))
knots_id_in_domain       = [row for row in range(len(knots_xy)) if (minX < knots_xy[row,0] < maxX and minY < knots_xy[row,1] < maxY)]
knots_xy                 = knots_xy[knots_id_in_domain]
knots_x                  = knots_xy[:,0]
knots_y                  = knots_xy[:,1]
k                        = len(knots_id_in_domain)
assert                k == len(knots_xy)

# Copula/Data Model Setup - X_star = R^phi * g(Z) -----------------------------------------------------------------

# Splines -----------------------------------------------------------------

# Gaussian weights (non-compact): used to interpolate smooth parameters such as
# the Matérn range and the RW tail index phi from knot values to sites.
# Weight matrix generated using Gaussian Smoothing Kernel
gaussian_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
    gaussian_weight_matrix[site_id, :] = weight_from_knots

# Wendland weights (compact support): used to build the positive stable / Lévy
# scale-mixture field R(s) from knot-level Lévy draws S_k(t).
# Weight matrix generated using wendland basis
wendland_weight_matrix = np.full(shape = (Ns,k), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
    wendland_weight_matrix[site_id, :] = weight_from_knots

# # constant weight matrix
# constant_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
# for site_id in np.arange(Ns):
#     # Compute distance between each pair of the two collections of inputs
#     d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
#                                     XB = knots_xy)
#     # influence coming from each of the knots
#     weight_from_knots = np.repeat(1, k)/k
#     constant_weight_matrix[site_id, :] = weight_from_knots

# -----------------------------------------------------------------------------
# 2) Latent Gaussian field and RW scale-mixture construction
# -----------------------------------------------------------------------------
# Covariance K for Gaussian Field g(Z) ------------------------------------

nu        = 0.5 # exponential kernel for matern with nu = 1/2
sigsq     = 1.0 # sill for Z
sigsq_vec = np.repeat(sigsq, Ns) # hold at 1

# Scale Mixture R^phi -----------------------------------------------------

gamma          = 0.5 # this is the gamma that goes in rlevy, gamma_at_knots
delta          = 0.0 # this is the delta in levy, stays 0
alpha          = 0.5
gamma_at_knots = np.repeat(gamma, k)
gamma_vec      = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha),
                        axis = 1)**(1/alpha) # bar{gamma}, axis = 1 to sum over K knots

# %% Generate Simulation Data

# Step A: Simulate Z(s,t) ~ MVN(0, K) independently over t, then map to Pareto.
# Transformed Gaussian Process - W = g(Z), Z ~ MVN(0, K) ------------------

range_vec = gaussian_weight_matrix @ range_at_knots
K         = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
                    coords = sites_xy, kappa = nu, cov_model = "matern")
Z         = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(Ns,)),cov=K,size=Nt).T
W         = norm_to_Pareto(Z)

# Step B: Simulate the Lévy mixture field R(s,t) and apply the RW power phi(s).
# S_k(t) are i.i.d. Lévy( location=delta, scale=gamma ) across knots and time.
# R(s,t) = sum_k w_k(s) S_k(t) and X*(s,t) = R(s,t)^{phi(s)} * W(s,t).
# Random Scaling Factor - R^phi -------------------------------------------

phi_vec    = gaussian_weight_matrix @ phi_at_knots
S_at_knots = rlevy(n=(k * Nt), m=delta, s=gamma).reshape(k, Nt)
R_at_sites = wendland_weight_matrix @ S_at_knots
R_phi      = R_at_sites ** phi_vec[:, np.newaxis]
X_star     = R_phi * W

# Step C: Convert X* to the data scale.
# pRW(.) returns the RW-mixture CDF F_RW(x; phi, gamma) (vectorized).
# qgev(.) maps uniform U=F_RW(X*) to the desired GEV marginal at each (s,t).
# Marginal Transform to GEV -----------------------------------------------

# Y            = np.full(shape = (Ns, Nt), fill_value = np.nan)

# We parallelize over time t because each column Y[:,t] is conditionally
# independent given X_star and the time-specific parameters.
def compute_column(t):
    return qgev(pRW(X_star[:,t], phi_vec, gamma_vec), mu_matrix[:,t], sigma_matrix[:,t], ksi_matrix[:,t])
with multiprocessing.get_context('fork').Pool(processes = N_CORES) as pool:
    results = list(tqdm(pool.imap(compute_column, np.arange(Nt)), total=Nt, desc='Marginal Transform'))
Y                 = np.column_stack(results) # shape (Ns, Nt)
Y_NA              = Y.copy()
Y_NA[miss_matrix] = np.nan

# %% Save Simulated Dataset ---------------------------------------------------------------------------------------

np.save('sites_xy.npy', sites_xy)
np.save('miss_matrix_bool.npy', miss_matrix)                                  # missing at random indicator matrix
np.save(rf'Y_truth_t{Nt}_s{Ns}_phi{phi_at_knots}_rho{range_at_knots}.npy', Y) # original dataset
np.save(rf'Y_NA_t{Nt}_s{Ns}_phi{phi_at_knots}_rho{range_at_knots}.npy', Y_NA) # dataset with NAs

np.save('mu_matrix.npy',      mu_matrix)
np.save('sigma_matrix.npy',   sigma_matrix)
np.save('ksi_matrix.npy',     ksi_matrix)
np.save('range_at_knots.npy', range_at_knots)
np.save('phi_at_knots.npy',   phi_at_knots)


# %% Checks on Data Generation ------------------------------------------------------------------------------------

# Check stable variables S ------------------------------------------------

# levy.cdf(S_at_knots, loc = 0, scale = gamma) should look uniform

for i in range(k):
    scipy.stats.probplot(scipy.stats.levy.cdf(S_at_knots[i,:], scale=gamma), dist='uniform', fit=False, plot=plt)
    plt.axline((0,0), slope = 1, color = 'black')
    plt.savefig(f'QQPlot_levy_knot_{i}.png')
    plt.show()
    plt.close()

# Check Pareto distribution -----------------------------------------------

# shifted pareto.cdf(W[site_i,:] + 1, b = 1, loc = 0, scale = 1) shoud look uniform

if norm_pareto == 'shifted':
    for site_i in range(Ns):
        if site_i % 10 == 0: # don't print all sites
            scipy.stats.probplot(scipy.stats.pareto.cdf(W[site_i,:]+1, b = 1, loc = 0, scale = 1), dist='uniform', fit=False, plot=plt)
            plt.axline((0,0), slope = 1, color = 'black')
            plt.savefig(f'QQPlot_Pareto_site_{site_i}.png')
            plt.show()
            plt.close()

# standard pareto.cdf(W[site_i,:], b = 1, loc = 0, scale = 1) shoud look uniform
if norm_pareto == 'standard':
    for site_i in range(Ns):
        if site_i % 10 == 0:
            scipy.stats.probplot(scipy.stats.pareto.cdf(W[site_i,:], b = 1, loc = 0, scale = 1), dist='uniform', fit=False, plot=plt)
            plt.axline((0,0), slope = 1, color = 'black')
            plt.savefig(f'QQPlot_Pareto_site_{site_i}.png')
            plt.show()
            plt.close()

# Check model X_star ------------------------------------------------------

# pRW(X_star) should look uniform (at each site with Nt time replicates)
for site_i in range(Ns):
    if site_i % 20 == 0:
        unif = pRW(X_star[site_i,:], phi_vec[site_i], gamma_vec[site_i])
        scipy.stats.probplot(unif, dist="uniform", fit = False, plot=plt)
        plt.axline((0,0), slope=1, color='black')
        plt.savefig(f'QQPlot_Xstar_site_{site_i}.png')
        plt.show()
        plt.close()

# # pRW(X_star) at each time t should deviates from uniform b/c spatial correlation
# for t in range(Nt):
#     if t % 5 == 0:
#         unif = pRW(X_star[:,t], phi_vec, gamma_vec[t])
#         scipy.stats.probplot(unif, dist="uniform", fit = False, plot=plt)
#         plt.axline((0,0), slope=1, color='black')
#         plt.savefig(f'QQPlot_Xstar_time_{t}.png')
#         plt.show()
#         plt.close()

# %% Check Marginal Y ---------------------------------------------------------
# A simple GEV MLE-fit should roughly reflects values around truth

myfits = [scipy.stats.genextreme.fit(Y[site,:][~np.isnan(Y[site,:])]) for site in range(Ns)]

color_loc = 'blue'
color_scale = 'green'
color_shape = 'red'

plt.hist([fit[1] for fit in myfits], bins=15, alpha=0.7, color=color_loc, label='Location')
plt.hist([fit[2] for fit in myfits], bins=15, alpha=0.7, color=color_scale, label='Scale')
plt.hist([fit[0] for fit in myfits], bins=15, alpha=0.7, color=color_shape, label='Shape')
plt.hist([fit[1] for fit in myfits], bins=15, alpha=1.0, histtype='step', edgecolor='black')
plt.hist([fit[2] for fit in myfits], bins=15, alpha=1.0, histtype='step', edgecolor='black')
plt.hist([fit[0] for fit in myfits], bins=15, alpha=1.0, histtype='step', edgecolor='black')

legend_handles = [
    matplotlib.patches.Patch(facecolor=color_loc, edgecolor='black', label='Location'),
    matplotlib.patches.Patch(facecolor=color_scale, edgecolor='black', label='Scale'),
    matplotlib.patches.Patch(facecolor=color_shape, edgecolor='black', label='Shape')
]

plt.legend(handles=legend_handles)
plt.title('MLE-fitted GEV')
plt.savefig('Histogram_MLE_fitted_GEV.png')
plt.show()
plt.close()

# %%