#!/usr/bin/env python3

"""
This script implements the Control Ratio (CoRa) method on the antithetic feedback motif (ATF), a simple model system with negative feedback control.

The script performs the following tasks:

-	Creating a model that includes feedback and the locally analogous no-feedback model.
-	Plotting the concentrations of different molecular species over time, to observe what happens in the systems and to check our model.
-	Calculating a CoRa point.
-	Varying a parameter in the model to explore the parameter space.
-	Calculating a CoRa line (get_CoRa_plot()).

US English
"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

import datetime


current_time = datetime.datetime.now()
print("Time at start of script: {}".format(current_time))

base_dir = os.path.abspath("..")
output_dir = os.path.join(base_dir, "output")

# Function to create output directory if it doesn't exist
def ensure_output_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_output_dir(output_dir)





def system_with_feedback(t, y, mu_U, mu_W, mu_Y, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y):
    """
    Models a antithetic feedback motif as a system of ordinary differential equations (ODEs)

    Parameters
    ----------
    t       :       array of float values
        the set of time points over which to analyze the system of ODEs
    y       :       array of float values
        the set of initial conditions (the initial concentrations of U, W, C and Y)

    mu_U, mu_W, mu_Y        :       floats
        synthesis rates of the species U, W and Y respectively
    
    gamma, gamma_U, gamma_W, gamma_Y         :       floats
        parameter values describing loss by dilution
    
    eta_plus        :       float
        binding rate of U and W (forming complex C)
    eta_0           :       float
        spontaneous unbinding rate of complex C
    eta_minus       :       float
        co-degradation rate of each molecule

    Returns
    -------
    dU_dt, dW_dt, dC_dt, dY_dt      :       
        ODEs describing the concentrations of species U, W, C and Y respectively
    """

    # Set initial conditions of the concentrations of all molecular species
    U, W, C, Y = y

    dU_dt = mu_U * Y - (gamma + gamma_U) * U - eta_plus * U * W + (eta_0 + gamma_W) * C
    dW_dt = mu_W - (gamma + gamma_W) * W - eta_plus * U * W + (eta_0 + gamma_U) * C
    dC_dt = eta_plus * U * W - (gamma + eta_0 + eta_minus + gamma_U + gamma_W) * C
    dY_dt = mu_Y * W - (gamma + gamma_Y) * Y
    #dY_dt = mu_Y * (W + C) - (gamma + gamma_Y) * Y

    return [dU_dt, dW_dt, dC_dt, dY_dt]


def system_without_feedback(t, y, mu_U, mu_W, mu_Y, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss):
    """
    Models a antithetic feedback motif without feedback as a system of ordinary differential equations (ODEs). The feedback from Y to U that was present in the original system
    is taken out of the system. A new species, Ystar, is used as replacement.

    Parameters
    ----------
    t       :       array of float values
        the set of time points over which to analyze the system of ODEs
    y       :       array of float values
        the set of initial conditions (the initial concentrations of U, W, C and Y)

    mu_U, mu_W, mu_Y, mu_Ystar        :       floats
        synthesis rates of the species U, W and Y respectively
    
    gamma, gamma_U, gamma_W, gamma_Y, gamma_Ystar         :       floats
        parameter values describing loss by dilution
    
    eta_plus        :       float
        binding rate of U and W (forming complex C)
    eta_0           :       float
        spontaneous unbinding rate of complex C
    eta_minus       :       float
        co-degradation rate of each molecule

    W_ss        :       float
        steady state concentration of W

    Returns
    -------
    dU_dt, dW_dt, dC_dt, dY_dt      :       
        ODEs describing the concentrations of species U, W, C and Y respectively
    """

    U, W, C, Y, Ystar = y

    # In the no-feedback system we modify dU_dt replacing Y by Y_star
    dU_dt = mu_U * Ystar - (gamma + gamma_U) * U - eta_plus * U * W + (eta_0 + gamma_W) * C
    dW_dt = mu_W - (gamma + gamma_W) * W - eta_plus * U * W + (eta_0 + gamma_U) * C
    dC_dt = eta_plus * U * W - (gamma + eta_0 + eta_minus + gamma_U + gamma_W) * C
    dY_dt = mu_Y * W - (gamma + gamma_Y) * Y
    #dY_dt = mu_Y * (W + C) - (gamma + gamma_Y) * Y

    # In the no-feedback system we add an extra ODE to describe the dynamics of Y_star
    dYstar_dt = mu_Ystar * W_ss - (gamma + gamma_Ystar) * Ystar
    #dYstar_dt = mu_Ystar * (W_ss + C_ss) - (gamma + gamma_Ystar) * Ystar

    return [dU_dt, dW_dt, dC_dt, dY_dt, dYstar_dt]


def calc_CoRa_point(ss, ss_NF, ss_pp, ss_pp_NF):
    """
    Calculates co-ratio (CoRa) by comparing the original and post perturbation (pp) states of the feedback and no-feedback systems

    Parameters
    ----------
    ss          :       float
        steady state value of the investigated species, for the original system with feedback before perturbation
    ss_NF       :       float
        steady state value of the investigated species, no-feedback (NF) system with feedback before perturbation
    ss_pp       :       float
        steady state value of the investigated species, original system with feedback post perturbation (pp)
    ss_pp_NF    :       float
        steady state value of the investigated species, no-feedback (NF) system post perturbation (pp)
    
    Returns
    -------
    CoRa        :       float
        CoRa point
    """
    try:
        if ss_NF == ss_pp_NF:
            return float('nan')
        CoRa = (np.log(ss_pp) - np.log(ss)) / (np.log(ss_pp_NF) - np.log(ss_NF))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        CoRa = float('nan')
    return CoRa


def get_concentration_plot(sol_FB, sol_NF, title):
    """
    Plots the concentration values of the feedback and no-feedback systems over time.

    Parameters
    ----------
    sol_FB              :           OdeSolution
        Solution object from solving the ODEs for the feedback system
    sol_NF              :           OdeSolution
        Solution object from solving the ODEs for the no-feedback system
    title               :           string
        Title for the plot and filename
    
    Returns
    -------
    None
    """
    print("entering plotting function")
    # Plot results
    plt.plot(sol_FB.t, sol_FB.y[0], label='U')
    plt.plot(sol_FB.t, sol_FB.y[1], label='W')
    plt.plot(sol_FB.t, sol_FB.y[2], label='C')
    plt.plot(sol_FB.t, sol_FB.y[3], label='Y')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    plt.title('Solution to the system of ODEs {}'.format(title))
    plt.savefig(os.path.join(output_dir, "concentration_plots", "Concentration_values_system_with_feedback_ATF_{}.png".format(title)))
    plt.show()
    plt.close()

    print("going to 2nd plot!")

    # Plot results
    plt.plot(sol_NF.t, sol_NF.y[0], label='U')
    plt.plot(sol_NF.t, sol_NF.y[1], label='W')
    plt.plot(sol_NF.t, sol_NF.y[2], label='C')
    plt.plot(sol_NF.t, sol_NF.y[3], label='Y')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    plt.title('Solution to the system of ODEs no-feedback system {}'.format(title))
    plt.savefig(os.path.join(output_dir, "concentration_plots", "Concentration_values_system_NO_feedback_ATF_{}.png".format(title)))
    plt.show()
    plt.close()


def get_CoRa_plot(perturbations, CoRa_values, title):
    """
    Plots the CoRa (Concentration Ratio) values against perturbations.

    Parameters
    ----------
    perturbations       :           list or numpy array
        Values of perturbations (mu_Y values) used in the analysis
    CoRa_values         :           list or numpy array
        Calculated CoRa values corresponding to the perturbations
    title               :           string
        Title for the plot and filename
    
    Returns
    -------
    None
    """
    print("entering CoRa plot function")
    # Plot results
    plt.plot(perturbations, CoRa_values)
    plt.xlabel('mu_Y value')
    plt.ylabel('CoRa Value')
    plt.title(title)
    plt.savefig(os.path.join(output_dir, "{}.png".format(title)))
    plt.show()
    plt.close()


def get_ss(t_span, args_FB, extra_args_NF, y0, y0_NF, t_eval, title):
    """
    Calculates the steady-state values of the feedback and locally analogous no-feedback systems, after evaluation time t_eval.

    Parameters
    ----------
    t_span              :           tuple
        begin and end time points of evaluation range
    args_FB             :           list
        parameter values for the feedback (and no-feedback) system
    extra_args_NF       :           list
        extra parameter values for the no-feedback system
    y0                  :           list
        initial conditions for feedback system
    y0_NF               :           list
        initial conditions for no-feedback system (includes one extra initial condition for YStar)
    t_eval              :           numpy array
        time points over which to analyze ODE
    title               :           string
        for plot and file name
    
    Returns
    -------
    Y_values_FB_system  :           numpy array
        Y values over time for the feedback system
    Y_values_NF_system  :           numpy array
        Y values over time for the no-feedback system
    U_ss                :           float
        Steady-state value of U in the feedback system
    W_ss                :           float
        Steady-state value of W in the feedback system
    C_ss                :           float
        Steady-state value of C in the feedback system
    Y_ss                :           float
        Steady-state value of Y in the feedback system
    Y_ss_NF             :           float
        Steady-state value of Y in the no-feedback system
    Ystar_ss_NF         :           float
        Steady-state value of YStar in the no-feedback system
    """
    print("Entering function >get_ss< xxxxxxxxxxxxxxxxxxxxxxxxxx")

    args_for_FB = tuple(args_FB)
    # Solve ODEs system with feedback
    sol_FB = solve_ivp(system_with_feedback, t_span, y0, args = args_for_FB, t_eval=t_eval)

    # Get the steady state values by selecting the final value of the investigated range. Make sure that the time span is long enough!!
    print(sol_FB.y[1][-1])
    U_ss = sol_FB.y[0][-1]
    W_ss = sol_FB.y[1][-1]
    C_ss = sol_FB.y[2][-1]
    Y_ss = sol_FB.y[3][-1]

    #steady_state_conditions = [U_ss, W_ss, C_ss, Y_ss]

    args_for_NF = args_for_FB + tuple(extra_args_NF) + (W_ss,)
    sol_NF = solve_ivp(system_without_feedback, t_span, y0_NF, args = args_for_NF, t_eval=t_eval)

    # Get steady states in NF system, these must be the same as in the FB system, before feedback
    print(sol_FB.y[1][-1])
    U_ss_NF = sol_NF.y[0][-1]
    W_ss_NF = sol_NF.y[1][-1]
    C_ss_NF = sol_NF.y[2][-1]
    Y_ss_NF = sol_NF.y[3][-1]
    Ystar_ss_NF = sol_NF.y[4][-1]

    print("The two values above must be the same (steady states before perturbation)")

    steady_state_conditions_NF = [U_ss_NF, W_ss_NF, C_ss_NF, Y_ss_NF, Ystar_ss_NF]
    print("steady state conditions NF are: {}".format(steady_state_conditions_NF))

    Y_values_FB_system = sol_FB.y[3]
    Y_values_NF_system = sol_NF.y[3]

    #title = ""
    print("We will plot now")
    get_concentration_plot(sol_FB, sol_NF, title)

    print("End of function >get_ss< xxxxxxxxxxxxxxxxxxxxxxxxxx")
    return Y_values_FB_system, Y_values_NF_system, U_ss, W_ss, C_ss, Y_ss, Y_ss_NF, Ystar_ss_NF


def check_steady_state(Y_ss, Y_ss_NF, ss_threshold = 0.9):
    """
    Checks whether the steady-state value of the no-feedback system (Y_ss_NF) falls within an acceptable range 
    around the steady-state value of the feedback system (Y_ss), based on a given threshold.

    Parameters
    ----------
    Y_ss                :           float
        Steady-state value of Y in the feedback system
    Y_ss_NF             :           float
        Steady-state value of Y in the no-feedback system
    ss_threshold        :           float, optional
        Threshold for determining the acceptable range for steady-state comparison, default is 0.9

    Returns
    -------
    bool
        True if Y_ss_NF falls within the acceptable range of Y_ss, False otherwise
    """
    print("Entering function >check_steady_state< ---------------------------")
    # TO DO: Figure out how to select the appropriate range in a mathematically correct and systematic way
    print("The value of the Y_ss variable here is {}".format(Y_ss))
    print("The value of the Y_ss_NF variable here is {}".format(Y_ss_NF))
    print(type(Y_ss))
    print(type(Y_ss_NF))

    Y_ss = np.float64(Y_ss)
    ss_threshold = np.float64(ss_threshold)

    bottom_range = ss_threshold * Y_ss
    top_range = (1/ss_threshold) * Y_ss

    print(Y_ss)
    print(Y_ss_NF)
    print("The value of the Y_ss_NF variable here is {}".format(Y_ss_NF))

    if bottom_range < Y_ss_NF < top_range:
        print("YES!")
        print("End of function >check_steady_state< ---------------------------")
        return True
    else:
        print("NO!")
        print("End of function >check_steady_state< ---------------------------")
        return False







def main():
    
    ### --------------------- Part I - set initial conditions and parameter values --------------------- ###
    print("### --------------------- Part I - set initial conditions and parameter values --------------------- ###")

    # Set initial conditions
    U0_init = 1.0
    W0_init = 1.0
    C0_init = 1.0
    Y0_init = 1.0
    y0_init = [U0_init, W0_init, C0_init, Y0_init]

    # Extra initial condition Ystar for no-feedback network
    Ystar0_init = 1.0
    y0_NF_init = [U0_init, W0_init, C0_init, Y0_init, Ystar0_init]

    # Time points where solution is computed. Make sure that the time span is large enough to reach steady state!!
    t_span = (0, 50000)  # from t=0 to t=10
    t_eval = np.linspace(t_span[0], t_span[1], 50000)

    # Parameters
    gamma = 1e-4        # min^-1
    gamma_U = 1e-4      # min^-1
    gamma_W = 1e-4      # min^-1
    mu_U = 0.125        # min^-1
    mu_W = 0.1          # nM min^-1
    mu_Y = 0.125        # min^-1
    eta_0 = 1e-4        # min^-1
    eta_plus = 0.0375   # nM^-1 min^-1
    eta_minus = 0.5     # min^-1
    gamma_Y = 1.0       # min^-1

    # Extra parameters system without feedback
    mu_Ystar = mu_Y #0.125 # Set equal to non-starred parameter
    gamma_Ystar = gamma_Y #1.0 # set equal to non-starred parameter

    current_mu_Y_value = mu_Y





    ### --------------------- Part II - Calculate steady states before perutbation (bp) and check if the feedback and no-feedback systems reach the same steady states (ss) --------------------- ###
    print("### --------------------- Part II - Calculate steady states before perutbation (bp) and check if the feedback and no-feedback systems reach the same steady states (ss) --------------------- ###")

    #C_ss = sol_FB.y[2][-1]
    #W_ss = solve_ivp(system_with_feedback, t_span, y0, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y), t_eval=t_eval)

    args_FB = [mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y]
    extra_args_NF = [mu_Ystar, gamma_Ystar]

    title = "Pre Perturbation"
    Y_values_FB_system, Y_values_NF_system, U_ss, W_ss, C_ss, Y_ss, Y_ss_NF, Ystar_ss_NF = get_ss(t_span, args_FB, extra_args_NF, y0_init, y0_NF_init, t_eval, title)

    # We do not pass on the threshold parameter here, because we will use the default value
    check_steady_state(Y_ss, Y_ss_NF)

    # Solve system without feedback
    #sol_NF = solve_ivp(system_without_feedback, t_span, y0_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss), t_eval=t_eval)
    #sol_NF = solve_ivp(system_without_feedback, t_span, y0_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss, C_ss), t_eval=t_eval)


    # Use steady state (ss) values as new initial conditions
    y0 = [U_ss, W_ss, C_ss, Y_ss]
    y0_NF = [U_ss, W_ss, C_ss, Y_ss, Ystar_ss_NF]

    title = "at Steady State"
    Y_values_FB_system, Y_values_NF_system, U_ss, W_ss, C_ss, Y_ss, Y_ss_NF, Ystar_ss_NF = get_ss(t_span, args_FB, extra_args_NF, y0, y0_NF, t_eval, title)
    ss_threshold = 0.999
    check_steady_state(Y_ss, Y_ss_NF, ss_threshold)
    
    # Store before perturbation (bp) steady state values
    Y_ss_bp = Y_ss
    Y_ss_bp_NF = Y_ss_NF





    ### --------------------- Part III - Calculate steady states post perutbation (pp) and calculate CoRa value --------------------- ###
    print("### --------------------- Part III - Calculate steady states post perutbation (pp) and calculate CoRa value --------------------- ###")

    # Use updated steady state (ss) values as new initial conditions
    y0 = [U_ss, W_ss, C_ss, Y_ss]
    y0_NF = [U_ss, W_ss, C_ss, Y_ss, Ystar_ss_NF]

    #CoRa_points_list = []

    # Calculate and show final time point
    current_time = datetime.datetime.now()
    print("Time at start of CoRa line calculation: {}".format(current_time))

    # Set the new mu value after perturbation
    perturbation = 1.05
    mu_after_perturbation = perturbation * mu_Y
    current_mu_Y_value = mu_after_perturbation

    # Replace mu_Y parameter value by the perturbed mu_Y value
    position = 2            # The position of "current_mu_Y_value" in the parameter set is 3rd, hence 2
    args_FB[position] = current_mu_Y_value
    print("The arguments (FB system only) we will pass on to the get_ss() function are: {} !!!!!!!!!!!!!!".format(args_FB))

    title = "after Perturbation"
    Y_values_FB_system, Y_values_NF_system, U_ss, W_ss, C_ss, Y_ss, Y_ss_NF, Ystar_ss_NF = get_ss(t_span, args_FB, extra_args_NF, y0, y0_NF, t_eval, title)
    ss_threshold = 0.999
    check_steady_state(Y_ss, Y_ss_NF, ss_threshold)         # Now the steady states don't necessarily match, because it is after perturbation

    # Store new (post perturbation) steady state values
    Y_ss_pp = Y_ss
    Y_ss_pp_NF = Y_ss_NF

    print("*******************************************************************************************We will now calculate a test CoRa point")
    print("The input values are Y_ss: {}    ;   Y_ss_NF: {}     ;   Y_ss_pp: {}  ;   Y_ss_pp_NF: {}".format(Y_ss, Y_ss_NF, Y_ss_pp, Y_ss_pp_NF))
    CoRa_test_point = calc_CoRa_point(Y_ss, Y_ss_NF, Y_ss_pp, Y_ss_pp_NF)
    print("A single CoRa point has been calculated: {} -----CoRa point--------------CoRa point------------CoRa point-----".format(CoRa_test_point))





    ### --------------------- Part IV - Vary parameter and calculate CoRa line --------------------- ###
    print("### --------------------- Part IV - Vary parameter and calculate CoRa line --------------------- ###")

    # Set list of different synthesis rate using log scale
    #mu_Y_values = np.logspace(0.01, 10000, 10)
    #mu_Y_values = [0.005, 0.01, 0.1, 1, 10, 100, 1000]
    mu_Y_values = [0.12, 0.122, 0.124, 0.125, 0.126, 0.128, 0.13]
    #mu_Y_values = [0.125, 0.125,0.125, 0.125, 0.125]

    CoRa_points_list = []
    mu_Y_values_list = []

    for mu_Y_value in mu_Y_values:

        current_mu_Y_value = mu_Y_value

        print("NEXT ITERATION IN LOOP ********************************************************")
        print("The current mu_Y_value in the loop is {}".format(current_mu_Y_value))

        print("The current mu_Y_value in the loop --- after perturbation --- is {}".format(current_mu_Y_value))
        title = ("after Perturbation_{}".format(current_mu_Y_value))

        args_FB = [mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y]
        extra_args_NF = [mu_Ystar, gamma_Ystar]

        position = 2    # Already defined earlier, but for clarity we will repeat here
        args_FB[position] = current_mu_Y_value
        print("The arguments (FB system only) we will pass on to the get_ss() function are: {} !!!!!!!!!!!!!!".format(args_FB))

        #print(y0_init)
        #print(y0_NF_init)
        Y_values_FB_system, Y_values_NF_system, U_ss, W_ss, C_ss, Y_ss, Y_ss_NF, Ystar_ss_NF = get_ss(t_span, args_FB, extra_args_NF, y0_init, y0_NF_init, t_eval, title)
        ss_threshold = 0.9
        
        steady_state_found = check_steady_state(Y_ss, Y_ss_NF, ss_threshold)

        
        title = ("at steady state {}".format(current_mu_Y_value))
        Y_values_FB_system, Y_values_NF_system, U_ss, W_ss, C_ss, Y_ss, Y_ss_NF, Ystar_ss_NF = get_ss(t_span, args_FB, extra_args_NF, y0, y0_NF, t_eval, title)
        ss_threshold = 0.999
        steady_state_found = check_steady_state(Y_ss, Y_ss_NF, ss_threshold)
        
        # Store before perturbation (bp) steady state values
        Y_ss_bp = Y_ss
        Y_ss_bp_NF = Y_ss_NF



        # Set the new mu value after perturbation
        perturbation = 1.05
        mu_after_perturbation = perturbation * mu_Y_value
        current_mu_Y_value = mu_after_perturbation

        # Replace mu_Y parameter value by the perturbed mu_Y value
        position = 2            # The position of "current_mu_Y_value" in the parameter set is 3rd, hence 2
        args_FB[position] = current_mu_Y_value


        title = ("after Perturbation {}".format(current_mu_Y_value))
        Y_values_FB_system, Y_values_NF_system, U_ss, W_ss, C_ss, Y_ss, Y_ss_NF, Ystar_ss_NF = get_ss(t_span, args_FB, extra_args_NF, y0, y0_NF, t_eval, title)
        ss_threshold = 0.999
        check_steady_state(Y_ss, Y_ss_NF, ss_threshold)

        # Store new (post perturbation) steady state values
        Y_ss_pp = Y_ss
        Y_ss_pp_NF = Y_ss_NF
        

        print("We reached condition statement")
        if steady_state_found == True:
            print("CONDITION ACCEPTED, STEADY STATE PRESENT, CALCULATING CORA VALUE")
            mu_Y_values_list.append(mu_Y_value)

            # Store new (post perturbation) steady state values
            Y_ss_pp = Y_ss
            Y_ss_pp_NF = Y_ss_NF

            # Calculate CoRa point
            print("The Y_ss_bp value is {}".format(Y_ss_bp))
            print("The Y_ss_bp_NF value is {}".format(Y_ss_bp_NF))
            print("The Y_ss_pp value is {}".format(Y_ss_pp))
            print("The Y_ss_pp_NF value is {}".format(Y_ss_pp_NF))
            CoRa_point = calc_CoRa_point(Y_ss_bp, Y_ss_bp_NF, Y_ss_pp, Y_ss_pp_NF)
            print("CoRa point FOUND!! it is {}".format(CoRa_point))
            CoRa_points_list.append(CoRa_point)
            print("The CoRa value calculated for the current system is {} and has been added to the list".format(CoRa_point))

    title_CoRa_plot = "CoRa values of antithetic feedback motif for multiple perturbations"
    #get_CoRa_plot(perturbations_set, CoRa_points_list, title_CoRa_plot)
    get_CoRa_plot(mu_Y_values_list, CoRa_points_list, title_CoRa_plot)
    # Calculate and show final time point
    current_time = datetime.datetime.now()
    print("Time end start of script: {}".format(current_time))
    

if __name__ == "__main__":
    main()
