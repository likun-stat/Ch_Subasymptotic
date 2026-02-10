"""
MCMC sampler for the RW mixture model (MPI parallel over time replicates).

This script is written as a *special case* of a more general sampler that can
handle *spatially varying* copula/marginal parameters via knot-based surfaces.
Here we intentionally keep the model **stationary** (constant across space and
time) by:
  - using a single knot (N_outer_grid = N_outer_grid_rho = 1) so that
    phi(s) and range(s) are effectively constant surfaces,
  - using constant GEV marginal parameters (mu, sigma, ksi) across sites/time
    (optionally updated as *global scalars* when UPDATE_GEV=True).

Model (one time replicate t):
  Y(s,t) ~ GEV( mu(s,t), sigma(s,t), ksi(s,t) )
  U(s,t) = F_GEV( Y(s,t) )
  X*(s,t) = F_RW^{-1}( U(s,t); phi(s), gamma_bar(s) )
  X*(s,t) = R_t^{phi(s)} * g(Z_t(s))
    - R_t is a positive Lévy (stable) scale-mixture variable (knot-based, with
      Wendland weights) controlling extremal dependence strength,
    - Z_t is a mean-zero Gaussian field with Matérn covariance (range surface)
      controlling *bulk* dependence, and g() maps Gaussian -> Pareto scale.

Parallelization:
  - Each MPI rank handles a single time index t = rank for the computationally
    expensive pieces (R_t update, likelihood terms involving Cholesky(K), etc.).
  - Rank 0 coordinates global parameter updates (phi knots, range knots, and
    optional global GEV parameters) and stores traces.

Dependencies:
  - utilities.py (RW CDF/PDF/quantile, likelihood, covariance builders, impute)
  - proposal_cov.py (optional: warm-start proposal covariances)
  - mpi4py, numpy, scipy

Notes for the "general" sampler:
  - To allow spatially varying parameters, increase N_outer_grid (and/or
    N_outer_grid_rho), and replace constant marginal matrices with covariate
    models (e.g., spline bases, regression surfaces). The block-update logic is
    already written in terms of knot coefficients.
"""


if __name__ == "__main__":

    # %% imports
    # imports base packages
    import os
    import time
    import pickle
    from   time import strftime, localtime
    os.environ["OMP_NUM_THREADS"]        = "1"  # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"]   = "1"  # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"]        = "1"  # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"]    = "1"  # export NUMEXPR_NUM_THREADS=1

    # imports extension packages
    import scipy
    import matplotlib
    import numpy                  as np
    from   mpi4py                 import MPI
    import matplotlib.pyplot      as plt

    # imports custom packages
    from   utilities import *
    import proposal_cov

    # packages setup
    # mgcv             = importr('mgcv')
    comm             = MPI.COMM_WORLD
    rank             = comm.Get_rank()
    size             = comm.Get_size()
    random_generator = np.random.RandomState((rank+1)*7) # use of this avoids impacting the global np state

    try: # data_seed is defined when python MCMC.py
        data_seed = int(sys.argv[1])
    except: # when running on local machine interactively
        data_seed = 2345
    finally:
        if rank == 0: print('data_seed: ', data_seed)
    np.random.seed(data_seed)

    if rank == 0: print('Using', norm_pareto, 'Pareto in sampler.')

    try:
        with open('iter.pkl','rb') as file:
            start_iter = pickle.load(file) + 1
            if rank == 0: print('start_iter loaded from pickle, set to be:', start_iter)
    except Exception as e:
        if rank == 0:
            print('Exception loading iter.pkl:', e)
            print('Setting start_iter to 1')
        start_iter = 1

    if norm_pareto == 'shifted':  n_iters = 5000
    if norm_pareto == 'standard': n_iters = 5000

    # --- Stationary special case: choose 1 knot so phi(s) is constant ---
    N_outer_grid      = 1
    # --- Stationary special case: choose 1 knot so range(s) is constant ---
    N_outer_grid_rho  = 1
    radius            = 1000                 # radius of Wendland Basis for R
    bandwidth         = radius               # range for the gaussian basis for phi
    bandwidth_rho     = radius               # range for the gaussian basis for rho
    # bandwidth         = radius**2/6
    # bandwidth_rho     = radius**2/6

    UPDATE_GEV        = True

    # %% Load Dataset -----------------------------------------------------------------------------------------------
    # Load Dataset    -----------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------------------
    # data

    Y           = np.load('Y_truth_t50_s1079_phi[0.3]_rho[0.2].npy')
    sites_xy    = np.load('sites_xy.npy')
    miss_matrix = np.isnan(Y)

    # %% Setup (Covariates and Constants) ----------------------------------------------------------------------------
    # Setup (Covariates and Constants)    ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------
    # Ns, Nt

    Nt   = Y.shape[1] # number of time replicates
    Ns   = Y.shape[0] # number of sites/stations
    Time = np.linspace(-Nt/2, Nt/2-1, Nt)/np.std(np.linspace(-Nt/2, Nt/2-1, Nt), ddof=1)

    # ----------------------------------------------------------------------------------------------------------------
    # Sites

    sites_x = sites_xy[:,0]
    sites_y = sites_xy[:,1]

    # define the lower and upper limits for x and y
    minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
    minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))

    # ----------------------------------------------------------------------------------------------------------------
    # Knots

    # isometric knot grid - for R and phi

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

    # isometric knot grid - for rho (de-coupled from phi)

    h_dist_between_knots_rho     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid_rho))-1)
    v_dist_between_knots_rho     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid_rho))-1)
    x_pos_rho                    = np.linspace(minX + h_dist_between_knots_rho/2, maxX + h_dist_between_knots_rho/2,
                                           num = int(2*np.sqrt(N_outer_grid_rho)))
    y_pos_rho                    = np.linspace(minY + v_dist_between_knots_rho/2, maxY + v_dist_between_knots_rho/2,
                                           num = int(2*np.sqrt(N_outer_grid_rho)))
    x_outer_pos_rho              = x_pos_rho[0::2]
    x_inner_pos_rho              = x_pos_rho[1::2]
    y_outer_pos_rho              = y_pos_rho[0::2]
    y_inner_pos_rho              = y_pos_rho[1::2]
    X_outer_pos_rho, Y_outer_pos_rho = np.meshgrid(x_outer_pos_rho, y_outer_pos_rho)
    X_inner_pos_rho, Y_inner_pos_rho = np.meshgrid(x_inner_pos_rho, y_inner_pos_rho)
    knots_outer_xy_rho           = np.vstack([X_outer_pos_rho.ravel(), Y_outer_pos_rho.ravel()]).T
    knots_inner_xy_rho           = np.vstack([X_inner_pos_rho.ravel(), Y_inner_pos_rho.ravel()]).T
    knots_xy_rho                 = np.vstack((knots_outer_xy_rho, knots_inner_xy_rho))
    knots_id_in_domain_rho       = [row for row in range(len(knots_xy_rho)) if (minX < knots_xy_rho[row,0] < maxX and minY < knots_xy_rho[row,1] < maxY)]
    knots_xy_rho                 = knots_xy_rho[knots_id_in_domain_rho]
    knots_x_rho                  = knots_xy_rho[:,0]
    knots_y_rho                  = knots_xy_rho[:,1]
    k_rho                        = len(knots_id_in_domain_rho)

    # ----------------------------------------------------------------------------------------------------------------
    # Copula Splines

    # Basis Parameters - for the Gaussian and Wendland Basis

    radius_from_knots = np.repeat(radius, k) # influence radius from a knot

    # Generate the weight matrices
    # Weight matrix generated using Gaussian Smoothing Kernel
    gaussian_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
    for site_id in np.arange(Ns):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
                                        XB = knots_xy)
        # influence coming from each of the knots
        weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
        gaussian_weight_matrix[site_id, :] = weight_from_knots

    # Weight matrix generated using wendland basis
    wendland_weight_matrix = np.full(shape = (Ns,k), fill_value = np.nan)
    for site_id in np.arange(Ns):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
                                        XB = knots_xy)
        # influence coming from each of the knots
        weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
        wendland_weight_matrix[site_id, :] = weight_from_knots

    # Gaussian weight matrix specific to the rho/range surface
    gaussian_weight_matrix_rho = np.full(shape = (Ns, k_rho), fill_value = np.nan)
    for site_id in np.arange(Ns):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
                                                    XB = knots_xy_rho)
        # influence coming from each of the knots
        weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
        gaussian_weight_matrix_rho[site_id, :] = weight_from_knots

    # # constant weight matrix
    # constant_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
    # for site_id in np.arange(Ns):
    #     # Compute distance between each pair of the two collections of inputs
    #     d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
    #                                     XB = knots_xy)
    #     # influence coming from each of the knots
    #     weight_from_knots = np.repeat(1, k)/k
    #     constant_weight_matrix[site_id, :] = weight_from_knots

    # ----------------------------------------------------------------------------------------------------------------
    # Setup For the Marginal Model - GEV(mu, sigma, ksi)

    # Stationary marginal GEV parameters used to *initialize* the chain.
    # If UPDATE_GEV=True, the sampler updates a single global (mu, sigma, ksi)
    # and broadcasts constant surfaces each iteration.
    mu_matrix    = np.full(shape = (Ns, Nt), fill_value = 40.0)
    sigma_matrix = np.full(shape = (Ns, Nt), fill_value = 30.0)
    ksi_matrix   = np.full(shape = (Ns, Nt), fill_value = 0.14)

    # ----------------------------------------------------------------------------------------------------------------
    # Setup For the Copula/Data Model - X_star = R^phi * g(Z)

    # Covariance K for Gaussian Field g(Z) --------------------------------------------------------------------------
    nu        = 0.5 # exponential kernel for matern with nu = 1/2
    sigsq     = 1.0 # sill for Z
    sigsq_vec = np.repeat(sigsq, Ns) # hold at 1

    # Scale Mixture R^phi --------------------------------------------------------------------------------------------
    ## phi and gamma
    gamma          = 0.5 # this is the gamma that goes in rlevy, gamma_at_knots
    delta          = 0.0 # this is the delta in levy, stays 0
    alpha          = 0.5
    gamma_at_knots = np.repeat(gamma, k)
    gamma_vec      = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha),
                            axis = 1)**(1/alpha) # bar{gamma}, axis = 1 to sum over K knots


    # %% Estimate Parameter -----------------------------------------------------------------------------------------------
    # Estimate Parameter    -----------------------------------------------------------------------------------------------

    if start_iter == 1:

        # ----------------------------------------------------------------------------------------------------------------
        # Data Model Parameters - X_star = R^phi * g(Z)

        # Knot coefficients for the Gaussian-field range surface.
        # In this stationary setup k_rho=1, so this is a single scalar.
        range_at_knots = np.array([0.2] * k_rho)
        # range_vec = gaussian_weight_matrix @ range_at_knots

        # Scale Mixture R^phi --------------------------------------------------------------------------------------------

        # Knot coefficients for the RW mixing exponent phi(s).
        # In this stationary setup k=1, so this is a single scalar.
        phi_at_knots = np.array([0.4] * k)
        phi_vec      = gaussian_weight_matrix @ phi_at_knots

        if norm_pareto == 'standard':
            R_at_knots = np.full(shape = (k, Nt), fill_value = np.nan)
            for t in np.arange(Nt):
                # only use non-missing values
                miss_index_1t = np.where(miss_matrix[:,t] == True)[0]
                obs_index_1t  = np.where(miss_matrix[:,t] == False)[0]
                R_at_knots[:,t] = (np.min(qRW(pgev(Y[obs_index_1t,t],
                                                   mu_matrix[obs_index_1t,t], sigma_matrix[obs_index_1t,t], ksi_matrix[obs_index_1t,t]),
                                            phi_vec[obs_index_1t], gamma_vec[obs_index_1t]))/1.5)**(1/phi_at_knots)
        else: # norm_pareto == 'shifted':
            # Calculate Rt in Parallel, only use non-missing values
            comm.Barrier()
            miss_index_1t = np.where(miss_matrix[:,rank] == True)[0]
            obs_index_1t  = np.where(miss_matrix[:,rank] == False)[0]
            X_1t       = qRW(pgev(Y[obs_index_1t,rank], mu_matrix[obs_index_1t,rank], sigma_matrix[obs_index_1t,rank], ksi_matrix[obs_index_1t,rank]),
                                phi_vec[obs_index_1t], gamma_vec[obs_index_1t])
            R_1t       = np.array([np.median(X_1t)**2] * k)
            R_gathered = comm.gather(R_1t, root = 0)
            R_at_knots = np.array(R_gathered).T if rank == 0 else None
            R_at_knots = comm.bcast(R_at_knots, root = 0)


    # %% Plot Parameter Surface
    # Plot Parameter Surface

    if rank == 0 and start_iter == 1:
        # 0. Grids for plots
        plotgrid_res_x = 100
        plotgrid_res_y = 100
        plotgrid_res_xy = plotgrid_res_x * plotgrid_res_y
        plotgrid_x = np.linspace(minX,maxX,plotgrid_res_x)
        plotgrid_y = np.linspace(minY,maxY,plotgrid_res_y)
        plotgrid_X, plotgrid_Y = np.meshgrid(plotgrid_x, plotgrid_y)
        plotgrid_xy = np.vstack([plotgrid_X.ravel(), plotgrid_Y.ravel()]).T

        gaussian_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy, k), fill_value = np.nan)
        for site_id in np.arange(plotgrid_res_xy):
            # Compute distance between each pair of the two collections of inputs
            d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)),
                                            XB = knots_xy)
            # influence coming from each of the knots
            weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
            gaussian_weight_matrix_for_plot[site_id, :] = weight_from_knots

        gaussian_weight_matrix_for_plot_rho = np.full(shape = (plotgrid_res_xy, k_rho), fill_value = np.nan)
        for site_id in np.arange(plotgrid_res_xy):
            # Compute distance between each pair of the two collections of inputs
            d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)),
                                                        XB = knots_xy_rho)
            # influence coming from each of the knots
            weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
            gaussian_weight_matrix_for_plot_rho[site_id, :] = weight_from_knots

        wendland_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy,k), fill_value = np.nan)
        for site_id in np.arange(plotgrid_res_xy):
            # Compute distance between each pair of the two collections of inputs
            d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)),
                                            XB = knots_xy)
            # influence coming from each of the knots
            weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
            wendland_weight_matrix_for_plot[site_id, :] = weight_from_knots

        # # weight from knot plots --------------------------------------------------------------------------------------
        # # visualize the weights coming from a knot

        # # Define the colors for the colormap (white to red)
        # # Create a LinearSegmentedColormap
        # colors = ["#ffffff", "#ff0000"]
        # n_bins = 50  # Number of discrete color bins
        # cmap_name = "white_to_red"
        # colormap = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        # min_w = 0
        # max_w = 1
        # n_ticks = 11  # (total) number of ticks
        # ticks = np.linspace(min_w, max_w, n_ticks).round(3)

        # idx = 5
        # gaussian_weights_for_plot = gaussian_weight_matrix_for_plot[:,idx]
        # wendland_weights_for_plot = wendland_weight_matrix_for_plot[:,idx]

        # fig, axes = plt.subplots(1,2)
        # state_map.boundary.plot(ax=axes[0], color = 'black', linewidth = 0.5)
        # heatmap = axes[0].imshow(wendland_weights_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
        #                     cmap = colormap, vmin = min_w, vmax = max_w,
        #                     interpolation='nearest',
        #                     extent = [minX, maxX, maxY, minY])
        # axes[0].invert_yaxis()
        # # axes[0].scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
        # axes[0].scatter(knots_x, knots_y, s = 30, color = 'white', marker = '+')
        # axes[0].set_xlim(minX, maxX)
        # axes[0].set_ylim(minY, maxY)
        # axes[0].set_aspect('equal', 'box')
        # axes[0].title.set_text('wendland weights knot ' + str(idx))

        # state_map.boundary.plot(ax=axes[1], color = 'black', linewidth = 0.5)
        # heatmap = axes[1].imshow(gaussian_weights_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
        #                     cmap = colormap, vmin = min_w, vmax = max_w,
        #                     interpolation='nearest',
        #                     extent = [minX, maxX, maxY, minY])
        # axes[1].invert_yaxis()
        # # axes[1].scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
        # axes[1].scatter(knots_x, knots_y, s = 30, color = 'white', marker = '+')
        # axes[1].set_xlim(minX, maxX)
        # axes[1].set_ylim(minY, maxY)
        # axes[1].set_aspect('equal', 'box')
        # axes[1].title.set_text('gaussian weights knot ' + str(idx))

        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
        # fig.colorbar(heatmap, cax = cbar_ax, ticks = ticks)
        # plt.savefig('Plot_knot_weights.pdf')
        # plt.show()
        # plt.close()
        # # -------------------------------------------------------------------------------------------------------------

        # 1. Station, Knots
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 8)

        # Plot knots and circles
        for i in range(k):
            circle_i = plt.Circle((knots_xy[i, 0], knots_xy[i, 1]), radius_from_knots[i],
                                color='r', fill=True, fc='grey', ec='None', alpha=0.2)
            ax.add_patch(circle_i)

        # Scatter plot for sites and knots
        ax.scatter(sites_x, sites_y, marker='.', c='blue', label='sites')
        ax.scatter(knots_x, knots_y, marker='+', c='red', label='knot', s=300)

        # Plot spatial domain rectangle
        space_rectangle = plt.Rectangle(xy=(minX, minY), width=maxX-minX, height=maxY-minY,
                                        fill=False, color='black')
        ax.add_patch(space_rectangle)

        # Set ticks and labels
        ax.set_xticks(np.linspace(minX, maxX, num=3))
        ax.set_yticks(np.linspace(minY, maxY, num=5))
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('Longitude', fontsize=40)
        plt.ylabel('Latitude', fontsize=40)

        # Set limits to match the exact range of your data
        plt.xlim([0, 10])
        plt.ylim([0, 10])

        # Plot boundary
        ax.set_aspect('equal', 'box')  # Ensures 1:1 ratio for data units

        # Adjust the position of the legend to avoid overlap with the plot
        box = ax.get_position()
        legend_elements = [matplotlib.lines.Line2D([0], [0], marker= '.', linestyle='None', color='b', label='Site'),
                        matplotlib.lines.Line2D([0], [0], marker='+', linestyle = "None", color='red', label='Knot Center',  markersize=30),
                        matplotlib.lines.Line2D([0], [0], marker = 'o', linestyle = 'None', label = 'Knot Radius', markerfacecolor = 'grey', markersize = 30, alpha = 0.2),
                        matplotlib.lines.Line2D([], [], color='None', marker='s', linestyle='None', markeredgecolor = 'black', markersize=30, label='Spatial Domain')]
        plt.legend(handles = legend_elements, bbox_to_anchor=(1.01,1.01), fontsize = 40)

        # Save or show plot
        plt.savefig('Plot_stations.pdf', bbox_inches="tight")
        plt.show()
        plt.close()

        # 3. phi surface
        # heatplot of phi surface
        phi_vec_for_plot = (gaussian_weight_matrix_for_plot @ phi_at_knots).round(3)
        graph, ax = plt.subplots()
        heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
                            cmap ='bwr', interpolation='nearest', extent = [minX, maxX, maxY, minY])
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        plt.savefig('Plot_initial_heatmap_phi_surface.pdf')
        # plt.show()
        plt.close()

        # 4. Plot range surface
        # heatplot of range surface
        range_vec_for_plot = gaussian_weight_matrix_for_plot_rho @ range_at_knots
        graph, ax = plt.subplots()
        heatmap = ax.imshow(range_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
                            cmap ='bwr', interpolation='nearest', extent = [minX, maxX, maxY, minY])
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        # plt.show()
        plt.savefig('Plot_initial_heatmap_rho_surface.pdf')
        # plt.show()
        plt.close()

    # %% MCMC Parameters
    # MCMC Parameters
    ##########################################################################################################
    ########### MCMC Parameters ##############################################################################
    ##########################################################################################################

    # ----------------------------------------------------------------------------------------------------------------
    # Block Update Specification

    if norm_pareto == 'standard':
        phi_block_idx_size   = 1
        range_block_idx_size = 1

    if norm_pareto == 'shifted':
        phi_block_idx_size = 4
        range_block_idx_size = 4

    # Create Coefficient Index Blocks - each block size does not exceed size specified above

    ## phi
    phi_block_idx_dict = {}
    lst = list(range(k))
    for i in range(0, k, phi_block_idx_size):
        start_index = i
        end_index   = i + phi_block_idx_size
        key         = 'phi_block_idx_'+str(i//phi_block_idx_size+1)
        phi_block_idx_dict[key] = lst[start_index:end_index]

    ## range
    range_block_idx_dict = {}
    lst = list(range(k_rho))
    for i in range(0, k_rho, range_block_idx_size):
        start_index = i
        end_index   = i + range_block_idx_size
        key         = 'range_block_idx_'+str(i//range_block_idx_size+1)
        range_block_idx_dict[key] = lst[start_index:end_index]

    # ----------------------------------------------------------------------------------------------------------------
    # Adaptive Update: tuning constants

    c_0 = 1
    c_1 = 0.8
    offset = 3 # the iteration offset: trick the updater thinking chain is longer
    # r_opt_1d = .41
    # r_opt_2d = .35
    # r_opt = 0.234 # asymptotically
    r_opt = .35
    adapt_size = 10

    # ----------------------------------------------------------------------------------------------------------------
    # Adaptive Update: Proposal Variance Scalar and Covariance Matrix

    if start_iter == 1: # initialize the adaptive update necessities
        # with no trial run
        R_log_cov              = np.tile(((2.4**2)/k)*np.eye(k)[:,:,None], reps = (1,1,Nt))
        phi_cov                = 1e-2 * np.identity(k)
        range_cov              = 0.5  * np.identity(k)

        # with trial run
        if proposal_cov.R_log_cov is not None:                R_log_cov  = proposal_cov.R_log_cov
        if proposal_cov.phi_cov is not None:                  phi_cov    = proposal_cov.phi_cov
        if proposal_cov.range_cov is not None:                range_cov  = proposal_cov.range_cov

        # make parameter block (for block updates)
        ## phi
        phi_block_cov_dict = {}
        for key in phi_block_idx_dict.keys():
            start_idx                    = phi_block_idx_dict[key][0]
            end_idx                      = phi_block_idx_dict[key][-1]+1
            phi_block_cov_dict[key] = phi_cov[start_idx:end_idx, start_idx:end_idx]

        ## range rho
        range_block_cov_dict = {}
        for key in range_block_idx_dict.keys():
            start_idx                      = range_block_idx_dict[key][0]
            end_idx                        = range_block_idx_dict[key][-1]+1
            range_block_cov_dict[key] = range_cov[start_idx:end_idx, start_idx:end_idx]

        if rank == 0: # Handle phi, range, GEV on Worker 0
            # proposal variance scalar
            sigma_m_sq = {}
            for key in phi_block_idx_dict.keys():
                sigma_m_sq[key] = (2.4**2)/len(phi_block_idx_dict[key])
            for key in range_block_idx_dict.keys():
                sigma_m_sq[key] = (2.4**2)/len(range_block_idx_dict[key])
            if UPDATE_GEV: 
                sigma_m_sq.update({'mu'    : 1.0})
                sigma_m_sq.update({'sigma' : 1.0})
                sigma_m_sq.update({'ksi'   : 1e-5})

            # proposal covariance matrix
            Sigma_0 = {}
            Sigma_0.update(phi_block_cov_dict)
            Sigma_0.update(range_block_cov_dict)

        # Rt: each Worker_t propose k-R(t)s at time t
        if rank == 0:
            if norm_pareto == 'shifted':
                sigma_m_sq_Rt_list = [np.mean(np.diag(R_log_cov[:,:,t])) for t in range(Nt)]
            if norm_pareto == 'standard':
                sigma_m_sq_Rt_list = [(np.diag(R_log_cov[:,:,t])) for t in range(Nt)]
        else:
            sigma_m_sq_Rt_list = None
        sigma_m_sq_Rt = comm.scatter(sigma_m_sq_Rt_list, root = 0) if size>1 else sigma_m_sq_Rt_list[0]
    else:
        # start_iter != 1, pickle load the Proposal Variance Scalar, Covariance Matrix

        ## Proposal Variance Scalar for Rt
        if rank == 0:
            with open('sigma_m_sq_Rt_list.pkl', 'rb') as file:
                sigma_m_sq_Rt_list = pickle.load(file)
        else:
            sigma_m_sq_Rt_list = None
        if size != 1: sigma_m_sq_Rt = comm.scatter(sigma_m_sq_Rt_list, root = 0)

        ## Proposal Variance Scalar and Covariance Matrix for other variables
        if rank == 0:
            with open('sigma_m_sq.pkl','rb') as file:
                sigma_m_sq = pickle.load(file)
            with open('Sigma_0.pkl', 'rb') as file:
                Sigma_0    = pickle.load(file)

    # ----------------------------------------------------------------------------------------------------------------
    # Adaptive Update: Counter

    ## Counter for Rt
    if norm_pareto == 'shifted':
        num_accepted_Rt_list = [0] * size if rank == 0 else None
        if size != 1: num_accepted_Rt = comm.scatter(num_accepted_Rt_list, root = 0)

    if norm_pareto == 'standard':
        num_accepted_Rt_list = [[0] * k] * size if rank == 0 else None
        if size > 1: num_accepted_Rt = comm.scatter(num_accepted_Rt_list, root = 0)
        if size == 1: num_accepted_Rt = num_accepted_Rt_list[0]

    ## Counter for other variables
    if rank == 0:
        num_accepted = {}
        for key in phi_block_idx_dict.keys():
            num_accepted[key] = 0
        for key in range_block_idx_dict.keys():
            num_accepted[key] = 0
        if UPDATE_GEV: 
            num_accepted.update({'mu'    : 0})
            num_accepted.update({'sigma' : 0})
            num_accepted.update({'ksi'   : 0})

    # 8. Storage for Traceplots -----------------------------------------------

    if rank == 0:
        if start_iter == 1:
            Y_trace                   = np.full(shape = (n_iters, Ns, Nt), fill_value = np.nan)
            loglik_trace              = np.full(shape = (n_iters, 1), fill_value = np.nan) # overall likelihood
            loglik_detail_trace       = np.full(shape = (n_iters, 5), fill_value = np.nan) # detail likelihood
            R_trace_log               = np.full(shape = (n_iters, k, Nt), fill_value = np.nan) # log(R)
            phi_knots_trace           = np.full(shape = (n_iters, k), fill_value = np.nan) # phi_at_knots
            range_knots_trace         = np.full(shape = (n_iters, k_rho), fill_value = np.nan) # range_at_knots
            if UPDATE_GEV: GEV_trace  = np.full(shape = (n_iters, 3), fill_value = np.nan)    
        else:
            Y_trace                   = np.load('Y_trace.npy')
            loglik_trace              = np.load('loglik_trace.npy')
            loglik_detail_trace       = np.load('loglik_detail_trace.npy')
            R_trace_log               = np.load('R_trace_log.npy')
            phi_knots_trace           = np.load('phi_knots_trace.npy')
            range_knots_trace         = np.load('range_knots_trace.npy')
            if UPDATE_GEV: GEV_trace  = np.load('GEV_trace.npy')
    else:
        Y_trace                   = None
        loglik_trace              = None
        loglik_detail_trace       = None
        R_trace_log               = None
        phi_knots_trace           = None
        range_knots_trace         = None
        if UPDATE_GEV: GEV_trace  = None

    # Initialize Parameters -------------------------------------------------------------------------------------

    if start_iter == 1:
        # Initialize at the truth/at other values
        Y_init                   = Y                   if rank == 0 else None
        R_matrix_init_log        = np.log(R_at_knots)  if rank == 0 else None
        phi_knots_init           = phi_at_knots        if rank == 0 else None
        range_knots_init         = range_at_knots      if rank == 0 else None
        if UPDATE_GEV: GEV_init  = np.array([40.0, 30.0, 0.14]) if rank == 0 else None

        if rank == 0: # store initial value into first row of traceplot
            Y_trace[0,:,:]                 = Y_init
            R_trace_log[0,:,:]             = R_matrix_init_log # matrix (k, Nt)
            phi_knots_trace[0,:]           = phi_knots_init
            range_knots_trace[0,:]         = range_knots_init
            if UPDATE_GEV: GEV_trace[0,:]  = GEV_init
    else:
        last_iter = start_iter - 1
        Y_init                   = Y_trace[last_iter,:,:]                 if rank == 0 else None
        R_matrix_init_log        = R_trace_log[last_iter,:,:]             if rank == 0 else None
        phi_knots_init           = phi_knots_trace[last_iter,:]           if rank == 0 else None
        range_knots_init         = range_knots_trace[last_iter,:]         if rank == 0 else None
        if UPDATE_GEV: GEV_init  = GEV_trace[last_iter,:]                 if rank == 0 else None

    # Set Current Values
    ## ---- log(R) --------------------------------------------------------------------------------------------
    # note: directly comm.scatter an numpy nd array along an axis is tricky,
    #       hence we first "redundantly" broadcast an entire R_matrix then split
    R_matrix_init_log = comm.bcast(R_matrix_init_log, root = 0) # matrix (k, Nt)
    R_current_log     = np.array(R_matrix_init_log[:,rank]) # vector (k,)
    R_vec_current     = wendland_weight_matrix @ np.exp(R_current_log)

    ## ---- phi ------------------------------------------------------------------------------------------------
    phi_knots_current = comm.bcast(phi_knots_init, root = 0)
    phi_vec_current   = gaussian_weight_matrix @ phi_knots_current

    ## ---- range_vec (length_scale) ---------------------------------------------------------------------------
    range_knots_current = comm.bcast(range_knots_init, root = 0)
    range_vec_current   = gaussian_weight_matrix_rho @ range_knots_current
    K_current           = ns_cov(range_vec = range_vec_current,
                                 sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
    cholesky_matrix_current = scipy.linalg.cholesky(K_current, lower = False)

    ## ---- GEV covariate coefficients --> GEV surface ----------------------------------------------------------
    if UPDATE_GEV:
        GEV_current          = comm.bcast(GEV_init, root = 0)
        Loc_matrix_current   = np.full(shape = (Ns, Nt), fill_value = GEV_current[0])
        Scale_matrix_current = np.full(shape = (Ns, Nt), fill_value = GEV_current[1])
        Shape_matrix_current = np.full(shape = (Ns, Nt), fill_value = GEV_current[2])
    if not UPDATE_GEV:
        Loc_matrix_current    = mu_matrix
        Scale_matrix_current  = sigma_matrix
        Shape_matrix_current  = ksi_matrix

    ## ---- Y (Ns, Nt) ------------------------------------------------------------------------------------------------

    Y = comm.bcast(Y_init, root = 0)

    ## ---- X_star ----------------------------------------------------------------------------------------------------

    miss_vec_1t   = miss_matrix[:,rank]
    miss_index_1t = np.where(miss_vec_1t == True)[0]
    obs_index_1t  = np.where(miss_vec_1t == False)[0]

    if start_iter == 1: # initial imputation
        X_star_1t_current                = np.full(shape = (Ns,), fill_value = np.nan) # contain missing values
        X_star_1t_current[obs_index_1t]  = qRW(pgev(Y[:,rank][obs_index_1t],
                                                Loc_matrix_current[obs_index_1t,rank],
                                                Scale_matrix_current[obs_index_1t,rank],
                                                Shape_matrix_current[obs_index_1t,rank]),
                                            phi_vec_current[obs_index_1t], gamma_vec[obs_index_1t])
        X_star_1t_miss, Y_1t_miss        = impute_1t(miss_index_1t, obs_index_1t,
                                                    Y[:,rank], X_star_1t_current,
                                                    Loc_matrix_current[:, rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                    phi_vec_current, gamma_vec, R_vec_current, K_current)
        X_star_1t_current[miss_index_1t] = X_star_1t_miss
        Y[:,rank][miss_index_1t]         = Y_1t_miss # this will modify Y[:,rank] because Y_1t_current shallow copy

        Y_gathered = comm.gather(Y[:,rank], root = 0)
        if rank == 0:
            Y_trace[0,:,:] = np.array(Y_gathered).T

        assert len(np.where(np.isnan(Y[:,rank]))[0]) == 0
    else:
        X_star_1t_current = qRW(pgev(Y[:,rank],
                                        Loc_matrix_current[:,rank],
                                        Scale_matrix_current[:,rank],
                                        Shape_matrix_current[:,rank]),
                                    phi_vec_current, gamma_vec)


    # %% Metropolis-Hasting Updates
    # Metropolis-Hasting Updates
    #####################################################################################################################
    ########### Metropolis-Hasting Updates ##############################################################################
    #####################################################################################################################

    comm.Barrier() # Blocking before the update starts

    if rank == 0:
        start_time = time.time()
        print('started on:', strftime('%Y-%m-%d %H:%M:%S', localtime(time.time())))
    lik_1t_current = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current,
                                                                Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                                phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
    prior_1t_current = np.sum(scipy.stats.levy.logpdf(np.exp(R_current_log), scale = gamma) + R_current_log)
    if not np.isfinite(lik_1t_current) or not np.isfinite(prior_1t_current):
        print('initial values lead to none finite likelihood')
        print('rank:',rank)
        print('lik_1t_current:',lik_1t_current)
        print('prior_1t_current of R:',prior_1t_current)

    for iter in range(start_iter, n_iters):
        # %% Update Rt
        ###########################################################
        #### ----- Update Rt ----- Parallelized Across Nt time ####
        ###########################################################
        if norm_pareto == 'standard':
            for i in range(k):
                # propose a Rt at knot i with truncation
                change_indices   = np.array([i])
                unchange_indices = np.array([x for x in range(k) if x not in change_indices])

                s_in_r       = np.where(wendland_weight_matrix[:,change_indices].ravel() != 0)[0]
                S_k_log      = R_current_log[unchange_indices]
                ub_trunc     = np.log(np.min((X_star_1t_current[s_in_r]**(1/phi_vec_current[s_in_r]) - wendland_weight_matrix[s_in_r,:][:,unchange_indices] @ np.exp(S_k_log)) / wendland_weight_matrix[s_in_r,:][:,change_indices].ravel()))
                ub           = (ub_trunc - R_current_log[change_indices]) / np.sqrt(sigma_m_sq_Rt[i])
                # lb           = (np.log(0) - R_current_log[change_indices]) / np.sqrt(sigma_m_sq_Rt[i])
                lb           = np.array([-np.inf])
                RV_truncnorm = scipy.stats.truncnorm(a = lb, b = ub,
                                                     loc = R_current_log[change_indices],
                                                     scale = np.sqrt(sigma_m_sq_Rt[i]))

                R_proposal_log                 = R_current_log.copy()
                R_proposal_log[change_indices] = RV_truncnorm.rvs(size = len(change_indices), random_state = random_generator)
                hasting_denom_log              = RV_truncnorm.logpdf(x = R_proposal_log[change_indices])[0] # g(log(S') | log(S))

                # note that for Rt, X_star doesn't change, so ub_trunc_new doesn't change
                ub_new = (ub_trunc - R_proposal_log[change_indices]) / np.sqrt(sigma_m_sq_Rt[i])
                RV_truncnorm_new = scipy.stats.truncnorm(a = lb, b = ub_new,
                                                         loc = R_proposal_log[change_indices],
                                                         scale = np.sqrt(sigma_m_sq_Rt[i]))
                hasting_num_log = RV_truncnorm_new.logpdf(x = R_current_log[change_indices])[0] # g(log(S) | log(S'))

                # Conditional log likelihood at Current
                R_vec_current = wendland_weight_matrix @ np.exp(R_current_log)
                if iter == 1: # otherwise lik_1t_current will be inherited
                    lik_1t_current = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current,
                                                                Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                                phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
                # log-prior density
                prior_1t_current = np.sum(scipy.stats.levy.logpdf(np.exp(R_current_log), scale = gamma) + R_current_log)

                # Conditional log likelihood at Proposal
                R_vec_proposal = wendland_weight_matrix @ np.exp(R_proposal_log)
                lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current,
                                                                Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                                phi_vec_current, gamma_vec, R_vec_proposal, cholesky_matrix_current)
                prior_1t_proposal = np.sum(scipy.stats.levy.logpdf(np.exp(R_proposal_log), scale = gamma) + R_proposal_log)

                u = random_generator.uniform()
                if not all(np.isfinite([lik_1t_proposal, prior_1t_proposal, hasting_num_log, lik_1t_current, prior_1t_current, hasting_denom_log])):
                    ratio = 0
                else:
                    ratio = np.exp(lik_1t_proposal + prior_1t_proposal + hasting_num_log -
                                   lik_1t_current - prior_1t_current - hasting_denom_log)
                if not np.isfinite(ratio):
                    ratio = 0
                if u > ratio:
                    Rt_accepted = False
                    R_update_log = R_current_log
                else:
                    Rt_accepted = True
                    R_update_log = R_proposal_log
                    num_accepted_Rt[i] += 1

                R_update_log_gathered = comm.gather(R_update_log, root = 0)
                if rank == 0:
                    R_trace_log[iter,:,:] = np.vstack(R_update_log_gathered).T

                R_current_log = R_update_log
                R_vec_current = wendland_weight_matrix @ np.exp(R_current_log)

                if Rt_accepted:
                    lik_1t_current = lik_1t_proposal

                comm.Barrier()

        if norm_pareto == 'shifted':
            # Propose a R at time "rank", on log-scale
            # Propose a R using adaptive update
            R_proposal_log = np.sqrt(sigma_m_sq_Rt)*random_generator.normal(loc = 0.0, scale = 1.0, size = k) + R_current_log
            # R_proposal_log = np.sqrt(sigma_m_sq_Rt)*np.repeat(random_generator.normal(loc = 0.0, scale = 1.0, size = 1), k) + R_current_log # spatially cst R(t)

            # Conditional log likelihood at Current
            R_vec_current = wendland_weight_matrix @ np.exp(R_current_log)
            if iter == 1: # otherwise lik_1t_current will be inherited
                lik_1t_current = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current,
                                                            Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                            phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
            # log-prior density
            prior_1t_current = np.sum(scipy.stats.levy.logpdf(np.exp(R_current_log), scale = gamma) + R_current_log)
            # prior_1t_current = prior_1t_current/k # if R(t) is spatially constant

            # Conditional log likelihood at Proposal
            R_vec_proposal = wendland_weight_matrix @ np.exp(R_proposal_log)
            # if np.any(~np.isfinite(R_vec_proposal**phi_vec_current)): print("Negative or zero R, iter=", iter, ", rank=", rank, R_vec_proposal[0], phi_vec_current[0])
            lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current,
                                                            Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                            phi_vec_current, gamma_vec, R_vec_proposal, cholesky_matrix_current)
            prior_1t_proposal = np.sum(scipy.stats.levy.logpdf(np.exp(R_proposal_log), scale = gamma) + R_proposal_log)
            # prior_1t_proposal = prior_1t_proposal/k # if R(t) is spatially constant

            # Gather likelihood calculated across time
            # no need of R(t) because each worker takes care of one

            # Accept or Reject
            u = random_generator.uniform()
            if not all(np.isfinite([lik_1t_proposal,prior_1t_proposal, lik_1t_current, prior_1t_current])):
                ratio = 0
                # print('iter:', iter, 'updating Rt', 't:', rank)
                # print('lik_1t_proposal:', lik_1t_proposal, 'prior_1t_proposal:', prior_1t_proposal)
                # print('lik_1t_current:', lik_1t_current, 'prior_1t_current:', prior_1t_current)
            else:
                ratio = np.exp(lik_1t_proposal + prior_1t_proposal - lik_1t_current - prior_1t_current)
            if not np.isfinite(ratio):
                ratio = 0 # Force a rejection
            if u > ratio: # Reject
                Rt_accepted = False
                R_update_log = R_current_log
            else: # Accept, u <= ratio
                Rt_accepted = True
                R_update_log = R_proposal_log
                num_accepted_Rt += 1

            # Store the result
            R_update_log_gathered = comm.gather(R_update_log, root=0)
            if rank == 0:
                R_trace_log[iter,:,:] = np.vstack(R_update_log_gathered).T

            # Update the current values
            R_current_log = R_update_log
            R_vec_current = wendland_weight_matrix @ np.exp(R_current_log)

            # Update the likelihood (to ease computation below)
            if Rt_accepted:
                lik_1t_current = lik_1t_proposal

            comm.Barrier() # block for R_t updates

        # %% Update phi
        ###################################################################################
        ####   Update phi_at_knots   ######################################################
        ###################################################################################

        if norm_pareto == 'standard': # asymetric proposal
            ub_idx               = np.where(np.log(R_vec_current) > 0)[0]
            lb_idx               = np.where(np.log(R_vec_current) < 0)[0]
            log_R_pos            = np.log(R_vec_current[ub_idx])
            log_R_neg            = np.log(R_vec_current[lb_idx])

            for key in phi_block_idx_dict.keys():
                change_indices = np.array(phi_block_idx_dict[key])
                unchange_indices = np.array([x for x in range(k) if x not in change_indices])
                phi_k = phi_knots_current[unchange_indices]

                # calculate truncation at each time
                ubs_1t = np.append((np.log(X_star_1t_current[ub_idx])/log_R_pos - gaussian_weight_matrix[ub_idx,:][:,unchange_indices] @ phi_k) / gaussian_weight_matrix[ub_idx,:][:,change_indices].ravel(), 1)
                lbs_1t = np.append((np.log(X_star_1t_current[lb_idx])/log_R_neg - gaussian_weight_matrix[lb_idx,:][:,unchange_indices] @ phi_k) / gaussian_weight_matrix[lb_idx,:][:,change_indices].ravel(), 0)
                ub_1t  = np.min(ubs_1t)
                lb_1t  = np.max(lbs_1t)
                assert ub_1t >= lb_1t

                ub_1t_gathered = comm.gather(ub_1t, root = 0)
                lb_1t_gathered = comm.gather(lb_1t, root = 0)

                # Asymmetric Proposal
                if rank == 0:
                    # truncation abscissae of the phi at knot that changed
                    ub_trunc = np.min(ub_1t_gathered)
                    lb_trunc = np.max(lb_1t_gathered)

                    # transform to ub,lb in standard deviations for scipy truncnorm
                    ub = (ub_trunc - phi_knots_current[change_indices]) / np.sqrt(sigma_m_sq[key])
                    lb = (lb_trunc - phi_knots_current[change_indices]) / np.sqrt(sigma_m_sq[key])
                    RV_truncnorm = scipy.stats.truncnorm(a = lb, b = ub,
                                                        loc = phi_knots_current[change_indices],
                                                        scale = np.sqrt(sigma_m_sq[key]))

                    # proposal from truncated normal
                    phi_knots_proposal                 = phi_knots_current.copy()
                    phi_knots_proposal[change_indices] = RV_truncnorm.rvs(size = len(change_indices), random_state = random_generator)

                    # (log of) Hasting ratio denominator, the g(phi' | phi) i.e. phi is current, phi' is proposed
                    hasting_denom_log = RV_truncnorm.logpdf(x = phi_knots_proposal[change_indices])

                else:
                    phi_knots_proposal = None

                phi_knots_proposal     = comm.bcast(phi_knots_proposal, root = 0)
                phi_vec_proposal       = gaussian_weight_matrix @ phi_knots_proposal

                # Conditional log likelihood at proposal
                phi_out_of_range = any(phi <= 0 for phi in phi_knots_proposal) or any(phi > 1 for phi in phi_knots_proposal) # U(0,1] prior
                if phi_out_of_range:
                    lik_1t_proposal = -np.inf
                else:
                    X_star_1t_proposal = qRW(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                            phi_vec_proposal, gamma_vec)
                    lik_1t_proposal    = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal,
                                            Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                            phi_vec_proposal, gamma_vec, R_vec_current, cholesky_matrix_current)

                # Gather likelihood calculated across time
                lik_current_gathered  = comm.gather(lik_1t_current, root = 0)
                lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

                # Hasting numerator, ratio g(phi | phi')
                ubs_1t_new = np.append((np.log(X_star_1t_proposal[ub_idx])/log_R_pos - gaussian_weight_matrix[ub_idx,:][:,unchange_indices] @ phi_k) / gaussian_weight_matrix[ub_idx,:][:,change_indices].ravel(), 1)
                lbs_1t_new = np.append((np.log(X_star_1t_proposal[lb_idx])/log_R_neg - gaussian_weight_matrix[lb_idx,:][:,unchange_indices] @ phi_k) / gaussian_weight_matrix[lb_idx,:][:,change_indices].ravel(), 0)
                ub_1t_new  = np.min(ubs_1t_new)
                lb_1t_new  = np.max(lbs_1t_new)
                ub_1t_new_gathered = comm.gather(ub_1t_new, root = 0)
                lb_1t_new_gathered = comm.gather(lb_1t_new, root = 0)

                if rank == 0:
                    # truncation abscissae of the phi at knot that changed
                    ub_new_trunc = np.min(ub_1t_new_gathered)
                    lb_new_trunc = np.max(lb_1t_new_gathered)

                    # transform to ub,lb in standard deviations for scipy truncnorm
                    ub_new = (ub_new_trunc - phi_knots_proposal[change_indices]) / np.sqrt(sigma_m_sq[key])
                    lb_new = (lb_new_trunc - phi_knots_proposal[change_indices]) / np.sqrt(sigma_m_sq[key])

                    RV_truncnorm_new = scipy.stats.truncnorm(a = lb_new, b = ub_new,
                                                            loc = phi_knots_proposal[change_indices],
                                                            scale = np.sqrt(sigma_m_sq[key]))

                    # (log of) Hasting ratio denominator, the g(phi | phi') i.e. phi is current, phi' is proposed
                    hasting_num_log = RV_truncnorm_new.logpdf(x = phi_knots_current[change_indices])

                # Handle prior and Accept/Reject on worker 0
                if rank == 0:
                    # use Beta(5,5) prior on each one of the parameters in the block
                    lik_current  = sum(lik_current_gathered)  + np.sum(scipy.stats.beta.logpdf(phi_knots_current,  a = 5, b = 5))
                    lik_proposal = sum(lik_proposal_gathered) + np.sum(scipy.stats.beta.logpdf(phi_knots_proposal, a = 5, b = 5))

                    # Hasting ratio
                    lik_proposal += sum(hasting_num_log)
                    lik_current  += sum(hasting_denom_log)

                    # Accept or Reject
                    u     = random_generator.uniform()
                    ratio = np.exp(lik_proposal - lik_current)
                    if not np.isfinite(ratio):
                        ratio = 0
                        if rank == 0:
                            print('likelihood ratio not finite')
                            print('lik_proposal:', lik_proposal)
                            print('lik_current:', lik_current)
                    if u > ratio: # Reject
                        phi_accepted     = False
                        phi_vec_update   = phi_vec_current
                        phi_knots_update = phi_knots_current
                    else: # Accept, u <= ratio
                        phi_accepted              = True
                        phi_vec_update            = phi_vec_proposal
                        phi_knots_update          = phi_knots_proposal
                        num_accepted[key] += 1

                    # Store the result
                    phi_knots_trace[iter,:] = phi_knots_update

                    # Update the current value
                    phi_vec_current   = phi_vec_update
                    phi_knots_current = phi_knots_update
                else: # broadcast to other workers
                    phi_accepted  = None
                phi_vec_current   = comm.bcast(phi_vec_current, root = 0)
                phi_knots_current = comm.bcast(phi_knots_current, root = 0)
                phi_accepted      = comm.bcast(phi_accepted, root = 0)

                # Update X_star and likelihood if accepted
                if phi_accepted:
                    X_star_1t_current = X_star_1t_proposal
                    lik_1t_current    = lik_1t_proposal

                comm.Barrier() # block for phi update

        if norm_pareto == 'shifted': # symmetric proposal
        # if norm_pareto == 'shifted' or 'standard':
            for key in phi_block_idx_dict.keys():
                change_indices = np.array(phi_block_idx_dict[key])

                # Propose new phi_block at the change_indices
                if rank == 0:
                    phi_knots_proposal                  = phi_knots_current.copy()
                    phi_knots_proposal[change_indices] += np.sqrt(sigma_m_sq[key]) * \
                                                            random_generator.multivariate_normal(np.zeros(len(change_indices)), Sigma_0[key])
                else:
                    phi_knots_proposal = None
                phi_knots_proposal     = comm.bcast(phi_knots_proposal, root = 0)
                phi_vec_proposal       = gaussian_weight_matrix @ phi_knots_proposal

                # Conditional log likelihood at proposal
                phi_out_of_range = any(phi <= 0 for phi in phi_knots_proposal) or any(phi > 1 for phi in phi_knots_proposal) # U(0,1] prior
                if phi_out_of_range:
                    lik_1t_proposal = -np.inf
                else:
                    X_star_1t_proposal = qRW(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                            phi_vec_proposal, gamma_vec)
                    lik_1t_proposal    = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal,
                                            Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                            phi_vec_proposal, gamma_vec, R_vec_current, cholesky_matrix_current)

                # Gather likelihood calculated across time
                lik_current_gathered  = comm.gather(lik_1t_current, root = 0)
                lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

                # Handle prior and Accept/Reject on worker 0
                if rank == 0:
                    # use Beta(5,5) prior on each one of the parameters in the block
                    lik_current  = sum(lik_current_gathered)  + np.sum(scipy.stats.beta.logpdf(phi_knots_current,  a = 5, b = 5))
                    lik_proposal = sum(lik_proposal_gathered) + np.sum(scipy.stats.beta.logpdf(phi_knots_proposal, a = 5, b = 5))

                    # Accept or Reject
                    u     = random_generator.uniform()
                    ratio = np.exp(lik_proposal - lik_current)
                    if not np.isfinite(ratio):
                        ratio = 0
                        if rank == 0:
                            print('likelihood ratio not finite')
                            print('lik_proposal:', lik_proposal)
                            print('lik_current:', lik_current)
                    if u > ratio: # Reject
                        phi_accepted     = False
                        phi_vec_update   = phi_vec_current
                        phi_knots_update = phi_knots_current
                    else: # Accept, u <= ratio
                        phi_accepted              = True
                        phi_vec_update            = phi_vec_proposal
                        phi_knots_update          = phi_knots_proposal
                        num_accepted[key] += 1

                    # Store the result
                    phi_knots_trace[iter,:] = phi_knots_update

                    # Update the current value
                    phi_vec_current   = phi_vec_update
                    phi_knots_current = phi_knots_update
                else: # broadcast to other workers
                    phi_accepted  = None
                phi_vec_current   = comm.bcast(phi_vec_current, root = 0)
                phi_knots_current = comm.bcast(phi_knots_current, root = 0)
                phi_accepted      = comm.bcast(phi_accepted, root = 0)

                # Update X_star and likelihood if accepted
                if phi_accepted:
                    X_star_1t_current = X_star_1t_proposal
                    lik_1t_current    = lik_1t_proposal

                comm.Barrier() # block for phi update

        # %% Update range
        #########################################################################################
        ####  Update range_at_knots  ############################################################
        #########################################################################################

        # Update range ACTUALLY in blocks
        for key in range_block_idx_dict.keys():
            change_indices = np.array(range_block_idx_dict[key])

            # Propose new range_block at the change indices
            if rank == 0:
                range_knots_proposal                  = range_knots_current.copy()
                range_knots_proposal[change_indices] += np.sqrt(sigma_m_sq[key])* \
                                                        random_generator.multivariate_normal(np.zeros(len(change_indices)), Sigma_0[key])
            else:
                range_knots_proposal = None
            range_knots_proposal     = comm.bcast(range_knots_proposal, root = 0)
            range_vec_proposal       = gaussian_weight_matrix_rho @ range_knots_proposal

            # Conditional log likelihood at proposal
            if any(range <= 0 for range in range_knots_proposal):
                lik_1t_proposal = -np.inf
            else:
                K_proposal = ns_cov(range_vec = range_vec_proposal,
                                    sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
                cholesky_matrix_proposal = scipy.linalg.cholesky(K_proposal, lower = False)
                lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current,
                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                    phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_proposal)

            # Gather likelihood calculated across time (no prior yet)
            lik_current_gathered   = comm.gather(lik_1t_current, root = 0)
            # like_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)
            lik_proposal_gathered  = comm.gather(lik_1t_proposal, root = 0)

            # Handle prior and Accept/Reject on worker 0
            if rank == 0:
                # use Half-Normal prior on each one of the range parameters in the block
                lik_current  = sum(lik_current_gathered)  + np.sum(scipy.stats.halfnorm.logpdf(range_knots_current, loc = 0, scale = 2))
                lik_proposal = sum(lik_proposal_gathered) + np.sum(scipy.stats.halfnorm.logpdf(range_knots_proposal, loc = 0, scale = 2))

                # Accept or Reject
                u     = random_generator.uniform()
                ratio = np.exp(lik_proposal - lik_current)
                if not np.isfinite(ratio):
                    ratio = 0 # Force a rejection
                if u > ratio: # Reject
                    range_accepted     = False
                    range_vec_update   = range_vec_current
                    range_knots_update = range_knots_current
                else:
                    range_accepted     = True
                    range_vec_update   = range_vec_proposal
                    range_knots_update = range_knots_proposal
                    num_accepted[key] += 1

                # Store the result
                range_knots_trace[iter,:] = range_knots_update

                # Update the current value
                range_vec_current   = range_vec_update
                range_knots_current = range_knots_update
            else: # Broadcast the update values
                range_accepted  = None
            range_vec_current   = comm.bcast(range_vec_current, root = 0)
            range_knots_current = comm.bcast(range_knots_current, root = 0)
            range_accepted      = comm.bcast(range_accepted, root = 0)

            # Update the K, cholesky_matrix, and likelihood
            if range_accepted:
                K_current               = K_proposal
                cholesky_matrix_current = cholesky_matrix_proposal
                lik_1t_current          = lik_1t_proposal

            comm.Barrier() # block for range_block updates

        # %% Update Missing Values
        #########################################################################################
        ####  Update missing values  ############################################################
        #########################################################################################

        # draw new Y_miss
        X_star_1t_miss, Y_1t_miss     = impute_1t(miss_index_1t, obs_index_1t,
                                                  Y[:,rank], X_star_1t_current,
                                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank],Shape_matrix_current[:,rank],
                                                    phi_vec_current, gamma_vec, R_vec_current, K_current)

        X_star_1t_current[miss_index_1t] = X_star_1t_miss
        Y[:,rank][miss_index_1t]      = Y_1t_miss

        lik_1t_current = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current,
                                                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                                    phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
        Y_gathered = comm.gather(Y[:,rank], root = 0)
        if rank == 0:
            Y_trace[iter,:,:] = np.array(Y_gathered).T

        comm.Barrier()

        if UPDATE_GEV:
            # %% Update mu 
            if rank == 0:
                mu_proposal  = GEV_current[0].copy()
                mu_proposal += np.sqrt(sigma_m_sq['mu']) * random_generator.normal(loc = 0.0, scale = 1.0, size = None)
            else:
                mu_proposal = None
            
            mu_proposal         = comm.bcast(mu_proposal, root = 0)
            Loc_matrix_proposal = np.full(shape = (Ns, Nt), fill_value = mu_proposal)
            
            # Conditional log likelihood at proposal
            X_star_1t_proposal    = qRW(pgev(Y[:,rank], Loc_matrix_proposal[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                        phi_vec_current, gamma_vec)
            lik_1t_proposal       = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal,
                                        Loc_matrix_proposal[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                        phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
            lik_current_gathered  = comm.gather(lik_1t_current, root = 0)
            lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

            # Handle prior and Accept/Reject on worker 0
            if rank == 0:
                lik_current   = sum(lik_current_gathered) + scipy.stats.norm.logpdf(GEV_current[0], loc = 40.0, scale = 10.0)
                lik_proposal  = sum(lik_proposal_gathered) + scipy.stats.norm.logpdf(mu_proposal, loc = 40.0, scale = 10.0)

                # Accept or Reject
                u     = random_generator.uniform()
                ratio = np.exp(lik_proposal - lik_current)
                if not np.isfinite(ratio):
                    ratio = 0 # Force a rejection
                if u > ratio: # Reject
                    mu_accepted         = False
                    mu_update           = GEV_current[0]
                else: # Accept, u <= ratio
                    mu_accepted         = True
                    mu_update           = mu_proposal
                    num_accepted['mu'] += 1

                # Store the result
                GEV_trace[iter,:] = np.array([mu_update, GEV_current[1], GEV_current[2]])

                # Update the current values
                GEV_current[0] = mu_update
            else:
                mu_accepted = None

            # Broadcast the update values
            mu_accepted  = comm.bcast(mu_accepted, root = 0)
            GEV_current  = comm.bcast(GEV_current, root = 0)
            if mu_accepted:
                Loc_matrix_current = np.full(shape = (Ns, Nt), fill_value = GEV_current[0])
                X_star_1t_current  = X_star_1t_proposal.copy()
                lik_1t_current     = lik_1t_proposal.copy()

            comm.Barrier()
                
            # %% Update sigma
            if rank == 0:
                sigma_proposal  = GEV_current[1].copy()
                sigma_proposal += np.sqrt(sigma_m_sq['sigma']) * random_generator.normal(loc = 0.0, scale = 1.0, size = None)
            else:
                sigma_proposal = None

            sigma_proposal         = comm.bcast(sigma_proposal, root = 0)
            Scale_matrix_proposal  = np.full(shape = (Ns, Nt), fill_value = sigma_proposal)

            # Conditional log likelihood at proposal
            if sigma_proposal <= 0:
                lik_1t_proposal = -np.inf
            else:
                X_star_1t_proposal    = qRW(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_current[:,rank]),
                                            phi_vec_current, gamma_vec)
                lik_1t_proposal       = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal,
                                            Loc_matrix_current[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_current[:,rank],
                                            phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
                
            # Gather likelihood calculated across time
            lik_current_gathered  = comm.gather(lik_1t_current, root = 0)
            lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

            # Handle prior and Accept/Reject on worker 0
            if rank == 0:
                lik_current   = sum(lik_current_gathered) + scipy.stats.norm.logpdf(GEV_current[1], loc = 30.0, scale = 10.0)
                lik_proposal  = sum(lik_proposal_gathered) + scipy.stats.norm.logpdf(sigma_proposal, loc = 30.0, scale = 10.0)

                # Accept or Reject
                u     = random_generator.uniform()
                ratio = np.exp(lik_proposal - lik_current)
                if not np.isfinite(ratio):
                    ratio = 0 # Force a rejection
                if u > ratio: # Reject
                    sigma_accepted         = False
                    sigma_update           = GEV_current[1]
                else: # Accept, u <= ratio
                    sigma_accepted         = True
                    sigma_update           = sigma_proposal
                    num_accepted['sigma'] += 1

                # Store the result
                GEV_trace[iter,:] = np.array([GEV_current[0], sigma_update, GEV_current[2]])

                # Update the current values
                GEV_current[1] = sigma_update
            else:
                sigma_accepted = None

            # Broadcast the update values
            sigma_accepted  = comm.bcast(sigma_accepted, root = 0)
            GEV_current     = comm.bcast(GEV_current, root = 0)
            if sigma_accepted:
                Scale_matrix_current = np.full(shape = (Ns, Nt), fill_value = GEV_current[1])
                X_star_1t_current    = X_star_1t_proposal.copy()
                lik_1t_current       = lik_1t_proposal.copy()

            comm.Barrier()

            # %% Update ksi
            if rank == 0:
                ksi_proposal  = GEV_current[2].copy()
                ksi_proposal += np.sqrt(sigma_m_sq['ksi']) * random_generator.normal(loc = 0.0, scale = 1.0, size = None)
            else:
                ksi_proposal = None

            ksi_proposal          = comm.bcast(ksi_proposal, root = 0)
            Shape_matrix_proposal = np.full(shape = (Ns, Nt), fill_value = ksi_proposal)

            # Conditional log likelihood at proposal
            if (ksi_proposal <= -0.5) or (ksi_proposal >= 0.5):
                lik_1t_proposal = -np.inf
            else:
                X_star_1t_proposal    = qRW(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_proposal[:,rank]),
                                            phi_vec_current, gamma_vec)
                lik_1t_proposal       = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal,
                                            Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_proposal[:,rank],
                                            phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)

            # Gather likelihood calculated across time
            lik_current_gathered  = comm.gather(lik_1t_current, root = 0)
            lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

            # Handle prior and Accept/Reject on worker 0
            if rank == 0:
                lik_current   = sum(lik_current_gathered) + scipy.stats.uniform.logpdf(GEV_current[2], loc = -0.5, scale = 1.0)
                lik_proposal  = sum(lik_proposal_gathered) + scipy.stats.uniform.logpdf(ksi_proposal, loc = -0.5, scale = 1.0)
            
                # Accept or Reject
                u     = random_generator.uniform()
                ratio = np.exp(lik_proposal - lik_current)
                if not np.isfinite(ratio):
                    ratio = 0 # Force a rejection
                if u > ratio: # Reject
                    ksi_accepted         = False
                    ksi_update           = GEV_current[2]
                else: # Accept, u <= ratio
                    ksi_accepted         = True
                    ksi_update           = ksi_proposal
                    num_accepted['ksi'] += 1

                print('iter:', iter, 
                      'current ksi:', GEV_current[2], 'proposal ksi:', ksi_proposal,
                      'lik_current:', round(lik_current, 3), 'lik_proposal:', round(lik_proposal, 3),
                      'accepted:', ksi_accepted)

                # Store the result
                GEV_trace[iter,:] = np.array([GEV_current[0], GEV_current[1], ksi_update])

                # Update the current values
                GEV_current[2] = ksi_update
            else:
                ksi_accepted = None

            # Broadcast the update values
            ksi_accepted  = comm.bcast(ksi_accepted, root = 0)
            GEV_current   = comm.bcast(GEV_current, root = 0)
            if ksi_accepted:
                Shape_matrix_current = np.full(shape = (Ns, Nt), fill_value = GEV_current[2])
                X_star_1t_current    = X_star_1t_proposal.copy()
                lik_1t_current       = lik_1t_proposal.copy()

            comm.Barrier()
            

        # %% After iteration likelihood
        ######################################################################
        #### ----- Keeping track of likelihood after this iteration ----- ####
        ######################################################################

        lik_final_1t_detail = marg_transform_data_mixture_likelihood_1t_detail(Y[:,rank], X_star_1t_current,
                                                Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
        lik_final_1t = sum(lik_final_1t_detail)
        lik_final_detail_gathered = comm.gather(lik_final_1t_detail, root = 0)
        lik_final_gathered = comm.gather(lik_final_1t, root = 0)
        if rank == 0:
            loglik_trace[iter,0] = round(sum(lik_final_gathered),3) # storing the overall log likelihood
            loglik_detail_trace[iter,:] = np.matrix(lik_final_detail_gathered).sum(axis=0) # storing the detail log likelihood

        comm.Barrier() # block for one iteration of update

        # %% Adaptive Update tunings
        #####################################################
        ###### ----- Adaptive Update autotunings ----- ######
        #####################################################

        if iter % adapt_size == 0:

            gamma1 = 1 / ((iter/adapt_size + offset) ** c_1)
            gamma2 = c_0 * gamma1

            # R_t
            if norm_pareto == 'shifted':
                sigma_m_sq_Rt_list = comm.gather(sigma_m_sq_Rt, root = 0)
                num_accepted_Rt_list = comm.gather(num_accepted_Rt, root = 0)
                if rank == 0:
                    for i in range(size):
                        r_hat = num_accepted_Rt_list[i]/adapt_size
                        num_accepted_Rt_list[i] = 0
                        log_sigma_m_sq_hat = np.log(sigma_m_sq_Rt_list[i]) + gamma2*(r_hat - r_opt)
                        sigma_m_sq_Rt_list[i] = np.exp(log_sigma_m_sq_hat)
                sigma_m_sq_Rt = comm.scatter(sigma_m_sq_Rt_list, root = 0)
                num_accepted_Rt = comm.scatter(num_accepted_Rt_list, root = 0)

            if norm_pareto == 'standard':
                for i in range(k):
                    r_hat              = num_accepted_Rt[i]/adapt_size
                    num_accepted_Rt[i] = 0
                    log_sigma_m_sq_hat = np.log(sigma_m_sq_Rt[i]) + gamma2 * (r_hat - r_opt)
                    sigma_m_sq_Rt[i]   = np.exp(log_sigma_m_sq_hat)
                comm.Barrier()
                sigma_m_sq_Rt_list     = comm.gather(sigma_m_sq_Rt, root = 0)


            # phi, range, and GEV
            if rank == 0:
                # phi block update
                for key in phi_block_idx_dict.keys():
                    start_idx          = phi_block_idx_dict[key][0]
                    end_idx            = phi_block_idx_dict[key][-1]+1
                    r_hat              = num_accepted[key]/adapt_size
                    num_accepted[key]  = 0
                    log_sigma_m_sq_hat = np.log(sigma_m_sq[key]) + gamma2 * (r_hat - r_opt)
                    sigma_m_sq[key]    = np.exp(log_sigma_m_sq_hat)
                    Sigma_0_hat        = np.array(np.cov(phi_knots_trace[iter-adapt_size:iter, start_idx:end_idx].T))
                    Sigma_0[key]       = Sigma_0[key] + gamma1 * (Sigma_0_hat - Sigma_0[key])

                # range block update
                for key in range_block_idx_dict.keys():
                    start_idx          = range_block_idx_dict[key][0]
                    end_idx            = range_block_idx_dict[key][-1]+1
                    r_hat              = num_accepted[key]/adapt_size
                    num_accepted[key]  = 0
                    log_sigma_m_sq_hat = np.log(sigma_m_sq[key]) + gamma2 * (r_hat - r_opt)
                    sigma_m_sq[key]    = np.exp(log_sigma_m_sq_hat)
                    Sigma_0_hat        = np.array(np.cov(range_knots_trace[iter-adapt_size:iter, start_idx:end_idx].T))
                    Sigma_0[key]       = Sigma_0[key] + gamma1 * (Sigma_0_hat - Sigma_0[key])

                if UPDATE_GEV:
                    # mu
                    r_hat = num_accepted['mu']/adapt_size
                    num_accepted['mu'] = 0
                    log_sigma_m_sq_hat = np.log(sigma_m_sq['mu']) + gamma2*(r_hat - r_opt)
                    sigma_m_sq['mu']   = np.exp(log_sigma_m_sq_hat)

                    # sigma
                    r_hat = num_accepted['sigma']/adapt_size
                    num_accepted['sigma'] = 0
                    log_sigma_m_sq_hat    = np.log(sigma_m_sq['sigma']) + gamma2*(r_hat - r_opt)
                    sigma_m_sq['sigma']   = np.exp(log_sigma_m_sq_hat)

                    # ksi
                    r_hat = num_accepted['ksi']/adapt_size
                    num_accepted['ksi'] = 0
                    log_sigma_m_sq_hat  = np.log(sigma_m_sq['ksi']) + gamma2*(r_hat - r_opt)
                    sigma_m_sq['ksi']   = np.exp(log_sigma_m_sq_hat)

        comm.Barrier() # block for adaptive update

        # %% Midway Printing, Drawings, and Savings
        ##############################################
        ###    Printing, Drawings, Savings       #####
        ##############################################

        if rank == 0: # Handle Drawing at worker 0
            # print(iter)
            if iter % 10 == 0:
                print(iter)
                # print(strftime('%Y-%m-%d %H:%M:%S', localtime(time.time())))
                end_time = time.time()
                print('elapsed: ', round(end_time - start_time, 1), ' seconds')
            if iter % 50 == 0 or iter == n_iters-1: # Save and print data every 50 iterations
                # %% Saving -------------------------------------------------------------------------------------------
                # Saving ----------------------------------------------------------------------------------------------

                ## Traceplots of parameters
                np.save('Y_trace', Y_trace)
                np.save('loglik_trace', loglik_trace)
                np.save('loglik_detail_trace', loglik_detail_trace)
                np.save('R_trace_log', R_trace_log)
                np.save('phi_knots_trace', phi_knots_trace)
                np.save('range_knots_trace', range_knots_trace)
                if UPDATE_GEV:
                    np.save('GEV_trace', GEV_trace)

                ## Adaptive Tuning "Parameter" for daisy-chainning the runs
                with open('iter.pkl', 'wb') as file:
                    pickle.dump(iter, file)

                with open('sigma_m_sq.pkl', 'wb') as file:
                    pickle.dump(sigma_m_sq, file)

                with open('Sigma_0.pkl', 'wb') as file:
                    pickle.dump(Sigma_0, file)

                with open('sigma_m_sq_Rt_list.pkl', 'wb') as file:
                    pickle.dump(sigma_m_sq_Rt_list, file)


                # %% Printing -----------------------------------------------------------------------------------------
                # Printing --------------------------------------------------------------------------------------------
                # Print traceplot thinned by 10
                xs       = np.arange(iter)
                xs_thin  = xs[0::10] # index 1, 11, 21, ...
                xs_thin2 = np.arange(len(xs_thin)) # index 1, 2, 3, ...

                loglik_trace_thin              = loglik_trace[0:iter:10,:]
                loglik_detail_trace_thin       = loglik_detail_trace[0:iter:10,:]
                R_trace_log_thin               = R_trace_log[0:iter:10,:,:]
                phi_knots_trace_thin           = phi_knots_trace[0:iter:10,:]
                range_knots_trace_thin         = range_knots_trace[0:iter:10,:]
                if UPDATE_GEV:
                    GEV_trace_thin                 = GEV_trace[0:iter:10,:]

                # ---- log-likelihood ----

                plt.subplots()
                plt.plot(xs_thin2, loglik_trace_thin)
                plt.title('traceplot for log-likelihood')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('loglikelihood')
                plt.savefig('Traceplot_loglik.pdf')
                plt.close()

                # ---- log-likelihood in details ----

                plt.subplots()
                for i in range(5):
                    plt.plot(xs_thin2, loglik_detail_trace_thin[:,i],label = i)
                    plt.annotate('piece ' + str(i), xy=(xs_thin2[-1], loglik_detail_trace_thin[:,i][-1]))
                plt.title('traceplot for detail log likelihood')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('log likelihood')
                plt.legend()
                plt.savefig('Traceplot_loglik_detail.pdf')
                plt.close()

                # ---- R_t ----

                for t in range(Nt):
                    label_by_knot = ['knot ' + str(knot) for knot in range(k)]
                    plt.subplots()
                    plt.plot(xs_thin2, R_trace_log_thin[:,:,t], label = label_by_knot)
                    plt.legend(loc = 'upper left')
                    plt.title('traceplot for log(Rt) at t=' + str(t))
                    plt.xlabel('iter thinned by 10')
                    plt.ylabel('log(Rt)s')
                    plt.savefig('Traceplot_Rt'+str(t)+'.pdf')
                    plt.close()

                # ---- phi ----

                plt.subplots()
                for i in range(k):
                    plt.plot(xs_thin2, phi_knots_trace_thin[:,i], label='knot ' + str(i))
                    plt.annotate('knot ' + str(i), xy=(xs_thin2[-1], phi_knots_trace_thin[:,i][-1]))
                plt.title('traceplot for phi')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('phi')
                plt.legend()
                plt.savefig('Traceplot_phi.pdf')
                plt.close()

                # ---- range ----

                plt.subplots()
                for i in range(k_rho):
                    plt.plot(xs_thin2, range_knots_trace_thin[:,i], label='knot ' + str(i))
                    plt.annotate('knot ' + str(i), xy=(xs_thin2[-1], range_knots_trace_thin[:,i][-1]))
                plt.title('traceplot for range')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('range')
                plt.legend()
                plt.savefig('Traceplot_range.pdf')
                plt.close()

                # ---- GEV ----
                if UPDATE_GEV:
                    ## location mu0 coefficients in blocks

                    # mu

                    plt.plot(xs_thin2, GEV_trace_thin[:,0], label = r'$\mu$')
                    plt.title(r'traceplot for $\mu$')
                    plt.xlabel('iter thinned by 10')
                    plt.ylabel(r'$\mu$')
                    plt.legend()
                    plt.savefig('Traceplot_mu.pdf')
                    plt.close()

                    # sigma

                    plt.plot(xs_thin2, GEV_trace_thin[:,1], label = r'$\sigma$')
                    plt.title(r'traceplot for $\sigma$')
                    plt.xlabel('iter thinned by 10')
                    plt.ylabel(r'$\sigma$')
                    plt.legend()
                    plt.savefig('Traceplot_sigma.pdf')
                    plt.close()

                    # ksi

                    plt.plot(xs_thin2, GEV_trace_thin[:,2], label = r'$\xi$')
                    plt.title(r'traceplot for $\xi$')
                    plt.xlabel('iter thinned by 10')
                    plt.ylabel(r'$\xi$')
                    plt.legend()
                    plt.savefig('Traceplot_ksi.pdf')
                    plt.close()

        comm.Barrier() # block for drawing


    # %% 11. End of MCMC Saving Traceplot
    # 11. End of MCMC Saving Traceplot ------------------------------------------------------------------------------
    if rank == 0:
        end_time = time.time()
        print('total time: ', round(end_time - start_time, 1), ' seconds')
        # print('true R: ', R_at_knots)
        np.save('Y_trace', Y_trace)
        np.save('loglik_trace', loglik_trace)
        np.save('loglik_detail_trace', loglik_detail_trace)
        np.save('R_trace_log', R_trace_log)
        np.save('phi_knots_trace', phi_knots_trace)
        np.save('range_knots_trace', range_knots_trace)
        if UPDATE_GEV:
            np.save('GEV_trace', GEV_trace)
