################################################################################
##                                                                            ##
## This code is amimed to Simulate recurrent neural network as described in:  ##
##                                                                            ##
## Sastre et. al. (2025).                                                    ##
## Cortical excitability inversely modulates fMRI connectivity                ##
##                                                                            ##
##  Author: Gabriele Mancini (gabriele.mancini@iit.it)                        ##
##                                                                            ##
################################################################################


import nest
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import nest.raster_plot
import pandas as pd
import statistics as sta
import random


n_trials = 5           ##NUMBER OF TRIALS
simtime = 60500
num_threads = 60
interval = int(simtime)
discarded = 500.

LFP_signal_1 = np.zeros((n_trials+1,interval-1))
LFP_signal_2 = np.zeros((n_trials+1,interval-1))

nest.Install('mymodule')

for trial in range(n_trials):

    nest.ResetKernel()
    nest.overwrite_files = True
    nest.local_num_threads = num_threads
    np.random.seed(int(time.time()))
    nest.rng_seed = int(time.time())

                     #! ===========
                     #! Parameters
                     #! ===========


                     # Parameters of the network

    order = 1000
    NE = 4 * order      #number of excitatory neurons
    NI = 1 * order          # number of inhibitory neurons
    N_neurons = NE + NI   # number of neurons in total


                         # Connectivity parameters


    epsilon = 0.2       # probability of connection
    epsilon_inter = 0.05 # probability of connection between different areas


    CE = int(epsilon * NE)  # number of excitatory synapses per neuron
    CI = int(epsilon * NI)  # number of inhibitory synapses per neuron


    C_tot = int(CI + CE)      # total number of synapses per neuron



                         # Parameters of Simulation


    simstep = 0.1                # ms
    delay = 1.                  # synaptic delay in ms
    delay_ratio = 2.          # synaptic delay ratio of inter-area/intra-area connections in ms


                     # Parameters of Neurons


    E_in= -80.0
    g_L=25.0


    #######################

    def variable_param(mean, std=None):
        if std is None:
            std = 0.1 * abs(mean)
        # Draw from normal distribution
        val = random.gauss(mean, std)
        # Apply bounds: for positive mean, between 0.5*mean and 1.5*mean; for negative mean, reversed bounds
        if mean > 0:
            lower, upper = 0.5 * mean, 1.5 * mean
        else:
            lower, upper = 1.5 * mean, 0.5 * mean
        # Clip value to bounds
        val = max(min(val, upper), lower)
        return float(val)


    excitatory_cell_params_1 = {
        "V_th": variable_param(-52.0, 0.),  # ~1 mV std
        "V_reset": variable_param(-59.0, 0.),
        "t_ref": variable_param(2.0, 0.),  # ~10% CV
        "g_L": variable_param(25.0, 0.),  # ~10%
        "C_m": variable_param(500.0, 0.),  # ~10%
        "E_ex": 0.0,
        "E_in": -80.0,
        "E_L": variable_param(-70.0, 0.),  ##
        "tau_rise_AMPA": variable_param(0.4, 0.01),  # ~12%
        "tau_decay_AMPA": variable_param(2.0, 0.01),  # std ~0.3 ms
        "tau_rise_GABA_A": variable_param(0.25, 0.01),
        "tau_decay_GABA_A": variable_param(5.0, 0.01),  # moderate variance
        "tau_m": variable_param(20.0, 0.01),
        "I_e": 0.0
    }

    inhibitory_cell_params_1 = {
        "V_th": variable_param(-52.0, 0.),  # ~1 mV std
        "V_reset": variable_param(-59.0, 0.),
        "t_ref": variable_param(1.0, 0.),
        "g_L": variable_param(20.0, 0.),
        "C_m": variable_param(200.0, 0.),
        "E_ex": 0.0,
        "E_in": -80.0,
        "E_L": variable_param(-70.0 , 0.),
        "tau_rise_AMPA": variable_param(0.2, 0.01),
        "tau_decay_AMPA": variable_param(1.0, 0.01),
        "tau_rise_GABA_A": variable_param(0.25, 0.01),
        "tau_decay_GABA_A": variable_param(5.0, 0.01),
        "tau_m": variable_param(10.0, 0.01),
        "I_e": 0.0
    }

    excitatory_cell_params_2 = {
        "V_th": variable_param(-52.0, 0.),  # ~1 mV std
        "V_reset": variable_param(-59.0, 0.),
        "t_ref": variable_param(2.0, 0.),  # ~10% CV
        "g_L": variable_param(25.0, 0.),  # ~10%
        "C_m": variable_param(500.0, 0.),  # ~10%
        "E_ex": 0.0,
        "E_in": -80.0,
        "E_L": variable_param(-70.0, 0.),
        "tau_rise_AMPA": variable_param(0.4, 0.01),  # ~12%
        "tau_decay_AMPA": variable_param(2.0, 0.01),  # std ~0.3 ms
        "tau_rise_GABA_A": variable_param(0.25, 0.01),
        "tau_decay_GABA_A": variable_param(5.0, 0.01),  # moderate variance
        "tau_m": variable_param(20.0, 0.01),
        "I_e": 0.0
    }

    inhibitory_cell_params_2 = {
        "V_th": variable_param(-52.0, 0.),  # ~1 mV std
        "V_reset": variable_param(-59.0, 0.),
        "t_ref": variable_param(1.0, 0.),
        "g_L": variable_param(20.0, 0.),
        "C_m": variable_param(200.0, 0.),
        "E_ex": 0.0,
        "E_in": -80.0,
        "E_L": variable_param(-70.0, 0.),
        "tau_rise_AMPA": variable_param(0.2, 0.01),
        "tau_decay_AMPA": variable_param(1.0, 0.01),
        "tau_rise_GABA_A": variable_param(0.25, 0.01),
        "tau_decay_GABA_A": variable_param(5.0, 0.01),
        "tau_m": variable_param(10.0, 0.01),
        "I_e": 0.0
    }

    # Parameters of Synapses
    initial_exc_exc = 0.178  # nS
    initial_exc_inh = 0.233  # nS
    initial_inh_inh = -2.70  # nS
    initial_inh_exc = -2.01  # nS


    pct = 0.0  # 1% decrease per trial (use 0.005 for 0.5%, etc.)

    scaling = (1 - pct)

    # adjust for the different manipulations
    exc_exc_recurrent_1 = initial_exc_exc * scaling
    exc_inh_recurrent_1 = initial_exc_inh * scaling
    inh_inh_recurrent_1 = initial_inh_inh * scaling
    inh_exc_recurrent_1 = initial_inh_exc * scaling

    exc_exc_recurrent_2 =  0.178   # nS
    exc_inh_recurrent_2 =  0.233   # nS
    inh_inh_recurrent_2 = -2.70    # n
    inh_exc_recurrent_2 = -2.01    # nS


    inter_exc_exc =  0.178
    inter_exc_inh =  0.233


    th_exc_external = 0.234     # nS
    th_inh_external = 0.33     # nS
    cc_exc_external = 0.187     # nS
    cc_inh_external = 0.254     # nS
                         # Parameters of Noise


             ##  CORTICO-CORTICAL
    OU_mean_CC_1 = 2.
    OU_mean_CC_2 = 2.
    OU_sigma = 0.1   #spikes/ms
    OU_tau = 0.1   #ms

            ##   THALAMO-CORTICAL

    OU_sigma_TH = 0.3  # spikes/ms
    OU_tau_TH = 80  # 1000/f   #ms
    OU_mean_TH = .3


                 ##############################################################################


                                      ### CREATION OF THE NETWORK ###


                 ###############################################################################


                         ### Populations ###


    nest.CopyModel("iaf_bw_2003_NMDA","exc_cell_1", excitatory_cell_params_1)
    nest.CopyModel("iaf_bw_2003_NMDA","inh_cell_1", inhibitory_cell_params_1)
    nest.CopyModel("iaf_bw_2003_NMDA","exc_cell_2", excitatory_cell_params_2)
    nest.CopyModel("iaf_bw_2003_NMDA","inh_cell_2", inhibitory_cell_params_2)

    exc_cell_1=nest.Create("exc_cell_1",NE)
    exc_cell_2=nest.Create("exc_cell_2",NE)

    inh_cell_1=nest.Create("inh_cell_1",NI)
    inh_cell_2=nest.Create("inh_cell_2",NI)



                     ### Noise ###

    # Thalamocortical and cortical-cortical inputs are defined as poisson generators

    # nest.CopyModel("sinusoidal_poisson_generator","thalamocortical_input",params=sin_params_A)
    # nest.CopyModel("sinusoidal_poisson_generator","thalamocortical_input_B",params=sin_params_B)

    nest.CopyModel("inhomogeneous_poisson_generator","thalamocortical_input")
    nest.CopyModel("inhomogeneous_poisson_generator", "cortical_input_1")
    nest.CopyModel("inhomogeneous_poisson_generator", "cortical_input_2")


             ####### Ornstein-Uhlenbeck process (solved by Euler method) ########

    step = simstep  # *2

    time_array = np.arange(step, simtime - step, step)

    OU_n = len(time_array)  # Number of time steps.

    # Define renormalized variables (to avoid recomputing these constants at every time step)

    OU_sigma_bis = OU_sigma * np.sqrt(2. / OU_tau)
    OU_sqrtdt = np.sqrt(step)
    OU_x_1 = np.zeros(OU_n)  # OU output
    OU_x_2 = np.zeros(OU_n)  # OU output

    OU_sigma_bis_TH = OU_sigma_TH * np.sqrt(2. / OU_tau_TH)
    OU_sqrtdt_TH = np.sqrt(step)
    OU_x_TH = np.zeros(OU_n)  # OU output

    for i in range(OU_n - 1):
        OU_x_1[i + 1] = OU_x_1[i] + step * ((OU_mean_CC_1 - OU_x_1[i]) / OU_tau) + \
                      OU_sigma_bis * OU_sqrtdt * np.random.randn()

        OU_x_2[i + 1] = OU_x_2[i] + step * ((OU_mean_CC_2 - OU_x_2[i]) / OU_tau) + \
                      OU_sigma_bis * OU_sqrtdt * np.random.randn()

        OU_x_TH[i + 1] = OU_x_TH[i] + step * ((OU_mean_TH - OU_x_TH[i]) / OU_tau_TH) + \
                         OU_sigma_bis_TH * OU_sqrtdt_TH * np.random.randn()

    # Final rate of the Poisson generators

    cc_params_1 = {"rate_times": time_array, 'rate_values': OU_x_1}
    cc_params_2 = {"rate_times": time_array, 'rate_values': OU_x_2}

    TH_params = {"rate_times": time_array, 'rate_values': OU_x_TH}

    noise_th = nest.Create("thalamocortical_input", params=TH_params)
    noise_cc_1 = nest.Create("cortical_input_1", params=cc_params_1)
    noise_cc_2 = nest.Create("cortical_input_2", params=cc_params_2)


                        ##  PARROT NEURONS ##

    parrot_th=nest.Create("parrot_neuron",NE)
    parrot_th_B=nest.Create("parrot_neuron",NE)

    parrot_cc_1=nest.Create("parrot_neuron",NE)
    parrot_cc_2=nest.Create("parrot_neuron",NE)


                            ### DEVICES ###

    espikes_1 = nest.Create("spike_recorder",params={'start': discarded})
    espikes_2 = nest.Create("spike_recorder",params={'start': discarded})
    ispikes_1 = nest.Create("spike_recorder",params={'start': discarded})
    ispikes_2 = nest.Create("spike_recorder",params={'start': discarded})

    multimeter_1=nest.Create('multimeter')
    multimeter_2=nest.Create('multimeter')

    multimeter_1.set(record_from=["V_m","g_ex","g_in","g_NMDA"])
    multimeter_2.set(record_from=["V_m","g_ex","g_in","g_NMDA"])


             ###############################  CONNECTING THE NETWORK ############################


    def variable_weight(mean, std_ratio=0.):
        std = abs(mean * std_ratio)
        return random.gauss(mean, std)  # Returns a float


    def variable_delay(mean, std_ratio=0.):
        std = abs(mean * std_ratio)
        return random.gauss(mean, std)  # Returns a float


    # === Recurrent connections ===
    nest.CopyModel("static_synapse", "exc_exc_rec_1",
                   {"weight": variable_weight(exc_exc_recurrent_1),
                    "delay": variable_delay(delay)})

    nest.CopyModel("static_synapse", "exc_inh_rec_1",
                   {"weight": variable_weight(exc_inh_recurrent_1),
                    "delay": variable_delay(delay)})

    nest.CopyModel("static_synapse", "inh_inh_rec_1",
                   {"weight": variable_weight(inh_inh_recurrent_1),
                    "delay": variable_delay(delay)})

    nest.CopyModel("static_synapse", "inh_exc_rec_1",
                   {"weight": variable_weight(inh_exc_recurrent_1),
                    "delay": variable_delay(delay)})

    nest.CopyModel("static_synapse", "exc_exc_rec_2",
                   {"weight": variable_weight(exc_exc_recurrent_2),
                    "delay": variable_delay(delay)})

    nest.CopyModel("static_synapse", "exc_inh_rec_2",
                   {"weight": variable_weight(exc_inh_recurrent_2),
                    "delay": variable_delay(delay)})

    nest.CopyModel("static_synapse", "inh_inh_rec_2",
                   {"weight": variable_weight(inh_inh_recurrent_2),
                    "delay": variable_delay(delay)})

    nest.CopyModel("static_synapse", "inh_exc_rec_2",
                   {"weight": variable_weight(inh_exc_recurrent_2),
                    "delay": variable_delay(delay)})

    # === Inter-population connections ===
    nest.CopyModel("static_synapse", "inter_exc_exc",
                   {"weight": variable_weight(inter_exc_exc),
                    "delay": variable_delay(delay_ratio * delay)})

    nest.CopyModel("static_synapse", "inter_exc_inh",
                   {"weight": variable_weight(inter_exc_inh),
                    "delay": variable_delay(delay_ratio * delay)})

    # === Thalamic inputs ===

    nest.CopyModel("static_synapse", "exc_exc_thalamo",
                   {"weight": variable_weight(th_exc_external),
                    "delay": variable_delay(delay)})

    nest.CopyModel("static_synapse", "exc_inh_thalamo",
                   {"weight": variable_weight(th_inh_external),
                    "delay": variable_delay(delay)})


    # === Cortical inputs ===
    nest.CopyModel("static_synapse", "exc_exc_cortical",
                   {"weight": variable_weight(cc_exc_external),
                    "delay": variable_delay(delay)})

    nest.CopyModel("static_synapse", "exc_inh_cortical",
                   {"weight": variable_weight(cc_inh_external),
                    "delay": variable_delay(delay)})

    ### CONNECT POPULATIONS ###

    conn_params_1= {'rule': 'pairwise_bernoulli', 'p': epsilon }
    conn_params_2= {'rule': 'symmetric_pairwise_bernoulli', 'p': epsilon, 'allow_autapses': False, 'make_symmetric': True }
    conn_params_inter= {'rule': 'pairwise_bernoulli', 'p': epsilon_inter }

    ### RECURRENT CONNECTIONS ###


    ## INTRA populations ##

    nest.Connect(exc_cell_1, exc_cell_1, conn_params_1, "exc_exc_rec_1")
    nest.Connect(exc_cell_2, exc_cell_2, conn_params_1, "exc_exc_rec_2")

    nest.Connect(exc_cell_1, inh_cell_1, conn_params_1, "exc_inh_rec_1")
    nest.Connect(exc_cell_2, inh_cell_2, conn_params_1, "exc_inh_rec_2")

    nest.Connect(inh_cell_1, exc_cell_1, conn_params_1, "inh_exc_rec_1")
    nest.Connect(inh_cell_2, exc_cell_2, conn_params_1, "inh_exc_rec_2")

    nest.Connect(inh_cell_1, inh_cell_1, conn_params_1, "inh_inh_rec_1")
    nest.Connect(inh_cell_2, inh_cell_2, conn_params_1, "inh_inh_rec_2")


     ##  external connections ##

    nest.Connect(noise_th, parrot_th)
    nest.Connect(noise_cc_1, parrot_cc_1)
    nest.Connect(noise_cc_2, parrot_cc_2)

    nest.Connect(parrot_th, exc_cell_1,conn_params_1, syn_spec="exc_exc_thalamo")
    nest.Connect(parrot_th, inh_cell_1,conn_params_1, syn_spec="exc_inh_thalamo")
    nest.Connect(parrot_th, exc_cell_2,conn_params_1, syn_spec="exc_exc_thalamo")
    nest.Connect(parrot_th, inh_cell_2,conn_params_1, syn_spec="exc_inh_thalamo")

    nest.Connect(parrot_cc_1, exc_cell_1,conn_params_1, syn_spec="exc_exc_cortical")
    nest.Connect(parrot_cc_1, inh_cell_1,conn_params_1, syn_spec="exc_inh_cortical")
    nest.Connect(parrot_cc_2, exc_cell_2,conn_params_1, syn_spec="exc_exc_cortical")
    nest.Connect(parrot_cc_2, inh_cell_2,conn_params_1, syn_spec="exc_inh_cortical")


      ##  INTER populations ##

    nest.Connect(exc_cell_1, exc_cell_2, conn_params_inter, "inter_exc_exc")      ### FEEDFORWARD E-E
    #nest.Connect(exc_cell_1, inh_cell_2, conn_params_inter, "inter_exc_inh")     ### FEEDFORWARD E-I
    #nest.Connect(exc_cell_2, exc_cell_1, conn_params_inter, "inter_exc_exc")      ### FEEDBACK E-E
    #nest.Connect(exc_cell_2, inh_cell_1, conn_params_inter, "inter_exc_inh")      ### FEEDBACK E-I


             ### DEVICES  ###

    nest.Connect(exc_cell_1, espikes_1)
    nest.Connect(exc_cell_2, espikes_2)
    nest.Connect(inh_cell_1, ispikes_1)
    nest.Connect(inh_cell_2, ispikes_2)
    
    nest.Connect(multimeter_1, exc_cell_1)
    nest.Connect(multimeter_2, exc_cell_2)


             ###############################  GO  ############################


    print("Simulating")

    nest.Simulate(simtime)


             ###############################  PLOT AND RESULTS  ############################

             ## AVERAGE RATE ###

    events_ex_A = espikes_1.n_events
    events_ex_B = espikes_2.n_events
    events_in_A = ispikes_1.n_events
    events_in_B = ispikes_2.n_events


    rate_ex_A = events_ex_A / (simtime-discarded) * 1000.0 / NE
    rate_ex_B = events_ex_B / (simtime-discarded) * 1000.0 / NE

    rate_in_A = events_in_A / (simtime-discarded) * 1000.0 / NI
    rate_in_B = events_in_B / (simtime-discarded) * 1000.0 / NI


    print(f"Excitatory rate Population 1   : {rate_ex_A:.2f} Hz")
    print(f"Excitatory rate Population 2   : {rate_ex_B:.2f} Hz")
    print(f"Inhibitory rate Population 1   : {rate_in_A:.2f} Hz")
    print(f"Inhibitory rate Population 2   : {rate_in_B:.2f} Hz")

    e_app_1 = espikes_1.get('events')
    i_app_1 = ispikes_1.get('events')

    sender_exc_spike_1 = e_app_1['senders']
    sender_inh_spike_1 = i_app_1['senders']
    time_exc_spikes_1 = e_app_1['times']
    time_inh_spikes_1 = i_app_1['times']

    e_app_2 = espikes_2.get('events')
    i_app_2 = ispikes_2.get('events')

    sender_exc_spike_2 = e_app_2['senders']
    sender_inh_spike_2 = i_app_2['senders']
    time_exc_spikes_2 = e_app_2['times']
    time_inh_spikes_2 = i_app_2['times']

    ##### COMPUTE SPIKES MATRIX #####

    bin_size = 3.0  # ms
    fac = int(bin_size / simstep)

    all_spikes_1 = np.zeros((N_neurons + 1, int((simtime - discarded) / simstep) + 1))

    for i in range(time_exc_spikes_1.size - 1):
        ind_i = int(sender_exc_spike_1[i + 1])
        ind_j = int((time_exc_spikes_1[i + 1] - discarded) / simstep)
        all_spikes_1[ind_i, ind_j] = 1.

    for i in range(time_inh_spikes_1.size - 1):
        ind_i = int(sender_inh_spike_1[i + 1]) - 4000
        ind_j = int((time_inh_spikes_1[i + 1] - discarded) / simstep)
        all_spikes_1[ind_i, ind_j] = 1.

    all_spikes_2 = np.zeros((N_neurons + 1, int((simtime - discarded) / simstep) + 1))

    for i in range(time_exc_spikes_2.size - 1):
        ind_i = int(sender_exc_spike_2[i + 1]) - 4000
        ind_j = int((time_exc_spikes_2[i + 1] - discarded) / simstep)
        all_spikes_2[ind_i, ind_j] = 1.

    for i in range(time_inh_spikes_2.size - 1):
        ind_i = int(sender_inh_spike_2[i + 1]) - 5000
        ind_j = int((time_inh_spikes_2[i + 1] - discarded) / simstep)
        all_spikes_2[ind_i, ind_j] = 1.

    binned_spikes_1 = np.add.reduceat(all_spikes_1, np.arange(0, all_spikes_1.shape[1], step=fac), 1)
    binned_spikes_2 = np.add.reduceat(all_spikes_2, np.arange(0, all_spikes_2.shape[1], step=fac), 1)

    x_1 = binned_spikes_1[1:, :]
    x_2 = binned_spikes_2[1:, :]


    bin_size = 2.0  # ms
    fac = int(bin_size / simstep)
    bin_size_sec = bin_size / 1000.0  # Convert to seconds
    duration_bins = int((simtime - discarded) / bin_size)

    # Arrays to hold population traces
    rate_trace_E1 = np.zeros(duration_bins)
    rate_trace_I1 = np.zeros(duration_bins)
    rate_trace_E2 = np.zeros(duration_bins)
    rate_trace_I2 = np.zeros(duration_bins)

    # Bin excitatory spikes (pop 1)
    for t in time_exc_spikes_1:
        if t > discarded:
            bin_idx = int((t - discarded) / bin_size)
            if bin_idx < duration_bins:
                rate_trace_E1[bin_idx] += 1


    # Bin inhibitory spikes (pop 1)
    for t in time_inh_spikes_1:
        if t > discarded:
            bin_idx = int((t - discarded) / bin_size)
            if bin_idx < duration_bins:
                rate_trace_I1[bin_idx] += 1

    # Bin excitatory spikes (pop 2)
    for t in time_exc_spikes_2:
        if t > discarded:
            bin_idx = int((t - discarded) / bin_size)
            if bin_idx < duration_bins:
                rate_trace_E2[bin_idx] += 1

    # Bin inhibitory spikes (pop 2)
    for t in time_inh_spikes_2:
        if t > discarded:
            bin_idx = int((t - discarded) / bin_size)
            if bin_idx < duration_bins:
                rate_trace_I2[bin_idx] += 1

    # Convert to Hz by dividing by (N_neurons * bin_size_sec)
    rate_trace_E1 /= (4000 * bin_size_sec)
    rate_trace_I1 /= (1000 * bin_size_sec)
    rate_trace_E2 /= (1000 * bin_size_sec)
    rate_trace_I2 /= (1000 * bin_size_sec)

    # Save to CSV files
    df_rates = pd.DataFrame({
        'rate_E1': rate_trace_E1,
        'rate_I1': rate_trace_I1,
        'rate_E2': rate_trace_E2,
        'rate_I2': rate_trace_I2
    })

    df_rates.to_csv(f'firing_rate_trace_trial{trial + 1}.csv', index=False)

    mm_1=multimeter_1.get('events')
    mm_2=multimeter_2.get('events')

    g_ex_1=mm_1['g_ex']
    g_in_1=mm_1['g_in']
    g_NMDA_1=mm_1['g_NMDA']
    senders_1=mm_1['senders']
    time_m_1=mm_1['times']
    V_mm_1=mm_1['V_m']

    g_ex_2=mm_2['g_ex']
    g_in_2=mm_2['g_in']
    g_NMDA_2=mm_2['g_NMDA']
    senders_2=mm_2['senders']
    time_m_2=mm_2['times']
    V_mm_2=mm_2['V_m']


    AMPA_current_1=(g_ex_1*V_mm_1)
    GABA_current_1=(g_in_1*(V_mm_1-E_in))
    NMDA_current_1=(g_NMDA_1*beta_e_1)*V_mm_1

    AMPA_current_2=(g_ex_2*V_mm_2)
    GABA_current_2=(g_in_2*(V_mm_2-E_in))
    NMDA_current_2=(g_NMDA_2*beta_e_2)*V_mm_2

    tot_current_1 = abs(AMPA_current_1) + abs(GABA_current_1) + abs(NMDA_current_1)
    tot_current_2 = abs(AMPA_current_2) + abs(GABA_current_2) + abs(NMDA_current_2)

    current_1 = {"times": time_m_1, 'senders': senders_1, "tot_current": tot_current_1}
    current_dict_1 = pd.DataFrame.from_dict(current_1)
    tot_current_pandas_1 = current_dict_1.groupby(["times"])["tot_current"].sum()
    LFP_signal_1[trial] = tot_current_pandas_1

    current_2 = {"times": time_m_2, 'senders': senders_2, "tot_current": tot_current_2}
    current_dict_2 = pd.DataFrame.from_dict(current_2)
    tot_current_pandas_2 = current_dict_2.groupby(["times"])["tot_current"].sum()
    LFP_signal_2[trial] = tot_current_pandas_2

data_1 = pd.DataFrame(LFP_signal_1)
data_1.to_csv('LFP_pop_1.csv')
data_2 = pd.DataFrame(LFP_signal_2)
data_2.to_csv('LFP_pop_2.csv')

data_1 = pd.DataFrame(LFP_signal_1)
data_1.to_csv('LFP_pop_1.csv')
data_2 = pd.DataFrame(LFP_signal_2)
data_2.to_csv('LFP_pop_2.csv')










         

            
        
