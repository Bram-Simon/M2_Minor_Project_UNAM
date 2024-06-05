#!/usr/bin/env python3

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




#def calc_CoRa(Y_fad, Y_fad_NF):
    """
    Calculates co-ratio (CoRa)
    """
    CoRa_values = []
    for i in range(len(Y_fad) - 1):
        #CoRa = (np.log(Y_fad[i + 1]) - np.log(Y_fad[i])) / (np.log(Y_fad_NF[i + 1]) - np.log(Y_fad_NF[i]))
        #CoRa = np.log(Y_fad[i + 1] / Y_fad[i]) / np.log(Y_fad_NF[i + 1] / Y_fad_NF[i + 1])          # FORMULA MAY NOT BE CORRECT YET!!!
        CoRa = (np.log(Y_fad[i + 1]) - np.log(Y_fad[i])) / (np.log(Y_fad_NF[i + 1]) - np.log(Y_fad_NF[i]))
        
        CoRa_values.append(CoRa)
    return CoRa_values


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

    CoRa = (np.log(ss_pp) - np.log(ss)) / (np.log(ss_pp_NF) - np.log(ss_NF))

    return CoRa


#def solve_system():



def get_concentration_plot(sol_FB, sol_NF, title):
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
    print("entering CoRa plot function")
    # Plot results
    plt.plot(perturbations, CoRa_values)
    plt.xlabel('Perturbation')
    plt.ylabel('CoRa Value')
    plt.title(title)
    plt.savefig(os.path.join(output_dir, "{}.png".format(title)))
    plt.show()
    plt.close()



def get_ss(t_span, args_FB, extra_args_NF, y0, y0_NF, t_eval):

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

    title = ""
    print("We will plot now")
    get_concentration_plot(sol_FB, sol_NF, title)

    return Y_values_FB_system, Y_values_NF_system, U_ss, W_ss, C_ss, Y_ss, Y_ss_NF, Ystar_ss_NF


def check_steady_state(Y_ss, Y_ss_NF, threshold = 0.99):
    print("Entering function >check_steady_state< ")
    # TO DO: Figure out how to select the appropriate range in a mathematically correct and systematic way
    bottom_range = threshold * Y_ss
    top_range = (1/threshold) * Y_ss

    print(Y_ss)
    print(Y_ss_NF)

    if Y_ss_NF > bottom_range and Y_ss_NF < top_range:
        print("YES!")

    #return ss_FB, ss_NF

"""
def get_ss_refined(t_span, args_FB, extra_args_NF, y0, y0_NF, t_eval):

    # Now we solve again, but using the steady state values as the initial conditions

    # Solve ODEs system with feedback
    sol_FB = solve_ivp(system_with_feedback, t_span, steady_state_conditions, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y), t_eval=t_eval)

    Ystar_ss = Y_ss
    #steady_state_conditions_NF = steady_state_conditions.append(Ystar_ss)


    # Solve system without feedback
    sol_NF = solve_ivp(system_without_feedback, t_span, steady_state_conditions_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss), t_eval=t_eval)


    Y_values_FB_system = sol_FB.y[3]
    Y_values_NF_system = sol_NF.y[3]



    title_ss = "steady state"
    print("We will plot now")
    get_concentration_plot(sol_FB, sol_NF, title_ss)"""





def main():
    
    # Set initial conditions
    U0 = 1.0
    W0 = 1.0
    C0 = 1.0
    Y0 = 1.0
    y0 = [U0, W0, C0, Y0]

    # Extra initial condition Ystar for no-feedback network
    Ystar0 = 1.0
    y0_NF = [U0, W0, C0, Y0, Ystar0]

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

    # Define range of mu_Y values
    #mu_Y = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

    # Range of mu_Y values on a logarithmic scale
    #mu_Y_values = np.logspace(-2, 2, 50)
    #cora_values_list = []

    current_mu_Y_value = mu_Y
    #for mu_Y_value in mu_Y_values:
        
    #current_mu_Y_value = mu_Y_value





    #C_ss = sol_FB.y[2][-1]
    #W_ss = solve_ivp(system_with_feedback, t_span, y0, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y), t_eval=t_eval)

    args_FB = [mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y]
    extra_args_NF = [mu_Ystar, gamma_Ystar]
    Y_values_FB_system, Y_values_NF_system, U_ss, W_ss, C_ss, Y_ss, Y_ss_NF, Ystar_ss_NF = get_ss(t_span, args_FB, extra_args_NF, y0, y0_NF, t_eval)

    check_steady_state(Y_ss, Y_ss_NF)

    # Solve system without feedback
    #sol_NF = solve_ivp(system_without_feedback, t_span, y0_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss), t_eval=t_eval)
    #sol_NF = solve_ivp(system_without_feedback, t_span, y0_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss, C_ss), t_eval=t_eval)


    # Use steady state (ss) values as new initial conditions
    y0 = [U_ss, W_ss, C_ss, Y_ss]
    y0_NF = [U_ss, W_ss, C_ss, Y_ss, Ystar_ss_NF]

    get_ss(t_span, args_FB, extra_args_NF, y0, y0_NF, t_eval)
    


    #title = None



    """ STILL TO IMPLEMENT IN FUNTIONS
    # Now we solve again, but using the steady state values as the initial conditions

    # Solve ODEs system with feedback
    sol_FB = solve_ivp(system_with_feedback, t_span, steady_state_conditions, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y), t_eval=t_eval)

    Ystar_ss = Y_ss
    #steady_state_conditions_NF = steady_state_conditions.append(Ystar_ss)


    # Solve system without feedback
    sol_NF = solve_ivp(system_without_feedback, t_span, steady_state_conditions_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss), t_eval=t_eval)


    Y_values_FB_system = sol_FB.y[3]
    Y_values_NF_system = sol_NF.y[3]



    title_ss = "steady state"
    print("We will plot now")
    get_concentration_plot(sol_FB, sol_NF, title_ss)
    """
  




    # Now we apply a perturbation and calculate again

    #mu_Y_perturbation = 0.5
    #current_mu_Y_value = mu_Y_perturbation



    # Solve ODEs system with feedback
    #sol_FB = solve_ivp(system_with_feedback, t_span, steady_state_conditions, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y), t_eval=t_eval)

    #Ystar_ss = Y_ss
    #steady_state_conditions_NF = steady_state_conditions.append(Ystar_ss)
    #steady_state_conditions_NF = [U_ss, W_ss, C_ss, Y_ss, Ystar_ss]
    #print("steady state conditions NF are: {}".format(steady_state_conditions_NF))

    # Solve system without feedback
    #sol_NF = solve_ivp(system_without_feedback, t_span, steady_state_conditions_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss), t_eval=t_eval)


    #Y_values_FB_system = sol_FB.y[3]
    #Y_values_NF_system = sol_NF.y[3]



    #title_after_perturbation = "after perturbation"
    #print("We will plot now")
    #get_concentration_plot(sol_FB, sol_NF, title_after_perturbation)




    # Get the steady state values post perturbation (pp) by selecting the final value of the investigated range. Make sure that the time span is long enough!!
    #print(sol_FB.y[1][-1])
    #U_ss_pp = sol_FB.y[0][-1]
    #W_ss_pp = sol_FB.y[1][-1]
    #C_ss_pp = sol_FB.y[2][-1]
    #Y_ss_pp = sol_FB.y[3][-1]

    #print(sol_NF.y[1][-1])
    #U_ss_pp_NF = sol_NF.y[0][-1]
    #W_ss_pp_NF = sol_NF.y[1][-1]
    #C_ss_pp_NF = sol_NF.y[2][-1]
    #Y_ss_pp_NF = sol_NF.y[3][-1]

    # Calculate and print CoRa point
    #CoRa_point = calc_CoRa_point(Y_ss, Y_ss_NF, Y_ss_pp, Y_ss_pp_NF)
    #print("The CoRa value calculated for the current system is {}".format(CoRa_point))



    #t_span = (0, 50000)  # from t=0 to t=10
    #t_eval = np.linspace(t_span[0], t_span[1], 50000)

    #perturbations_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #perturbation_span = (0.1, 1)
    #perturbations_set = np.linspace(perturbation_span[0], perturbation_span[1], 91)
    
    
    """ STILL TO IMPLEMENT IN FUNTIONS
    CoRa_points_list = []

    # Calculate and show final time point
    current_time = datetime.datetime.now()
    print("Time at start of CoRa line calculation: {}".format(current_time))

    # Set the new mu value after perturbation
    perturbation = 1.05
    mu_after_perturbation = perturbation * mu_Y
    current_mu_Y_value = mu_after_perturbation

    # Solve ODE system with and without feedback after applying the current perturbation
    sol_FB = solve_ivp(system_with_feedback, t_span, steady_state_conditions, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y), t_eval=t_eval)
    sol_NF = solve_ivp(system_without_feedback, t_span, steady_state_conditions_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss), t_eval=t_eval)
    
    # Extract new (post perturbation) steady state values
    Y_ss_pp = sol_FB.y[3][-1]
    Y_ss_pp_NF = sol_NF.y[3][-1]

    # Plot concentration graphs (to check if steady state was reached)
    title_after_perturbation = "after perturbation of {}".format(current_mu_Y_value)
    print(title_after_perturbation)
    get_concentration_plot(sol_FB, sol_NF, title_after_perturbation)

    # Calculate CoRa point
    CoRa_point = calc_CoRa_point(Y_ss, Y_ss_NF, Y_ss_pp, Y_ss_pp_NF)
    CoRa_points_list.append(CoRa_point)
    print("The CoRa value calculated for the current system is {} and has been added to the list".format(CoRa_point))

    title_CoRa_plot = "CoRa values of antithetic feedback motif for multiple perturbations"
    #get_CoRa_plot(perturbations_set, CoRa_points_list, title_CoRa_plot)
    get_CoRa_plot(perturbation, CoRa_points_list, title_CoRa_plot)
    # Calculate and show final time point
    current_time = datetime.datetime.now()
    print("Time end start of script: {}".format(current_time))


    # Set list of different synthesis rate using log scale
    mu_Y_values = np.logspace(-2, 2, 50)
    CoRa_points_list = []

    for mu_value in mu_Y_values:

        # Set the new mu value
        current_mu_Y_value = perturbation * mu_value

        # Solve ODE system with and without feedback after applying the current perturbation
        sol_FB = solve_ivp(system_with_feedback, t_span, steady_state_conditions, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y), t_eval=t_eval)
        sol_NF = solve_ivp(system_without_feedback, t_span, steady_state_conditions_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss), t_eval=t_eval)
        
        # Extract new (post perturbation) steady state values
        Y_ss_pp = sol_FB.y[3][-1]
        Y_ss_pp_NF = sol_NF.y[3][-1]

        # Plot concentration graphs (to check if steady state was reached)
        title_after_perturbation = "after perturbation of {} NEW".format(current_mu_Y_value)
        print(title_after_perturbation)
        get_concentration_plot(sol_FB, sol_NF, title_after_perturbation)

        # Calculate CoRa point
        CoRa_point = calc_CoRa_point(Y_ss, Y_ss_NF, Y_ss_pp, Y_ss_pp_NF)
        CoRa_points_list.append(CoRa_point)
        print("The CoRa value calculated for the current system is {} and has been added to the list".format(CoRa_point))

    title_CoRa_plot = "CoRa values of antithetic feedback motif for multiple perturbations NEW"
    get_CoRa_plot(mu_Y_values, CoRa_points_list, title_CoRa_plot)

    # Calculate and show final time point
    current_time = datetime.datetime.now()
    print("Time end start of script: {}".format(current_time))
    """




"""
    for perturbation in perturbations_set:
        
        # Set the new mu value
        current_mu_Y_value = perturbation

        # Solve ODE system with and without feedback after applying the current perturbation
        sol_FB = solve_ivp(system_with_feedback, t_span, steady_state_conditions, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y), t_eval=t_eval)
        sol_NF = solve_ivp(system_without_feedback, t_span, steady_state_conditions_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss), t_eval=t_eval)
        
        # Extract new (post perturbation) steady state values
        Y_ss_pp = sol_FB.y[3][-1]
        Y_ss_pp_NF = sol_NF.y[3][-1]

        # Plot concentration graphs (to check if steady state was reached)
        title_after_perturbation = "after perturbation of {}".format(current_mu_Y_value)
        print(title_after_perturbation)
        get_concentration_plot(sol_FB, sol_NF, title_after_perturbation)

        # Calculate CoRa point
        CoRa_point = calc_CoRa_point(Y_ss, Y_ss_NF, Y_ss_pp, Y_ss_pp_NF)
        CoRa_points_list.append(CoRa_point)
        print("The CoRa value calculated for the current system is {} and has been added to the list".format(CoRa_point))

    title_CoRa_plot = "CoRa values of antithetic feedback motif for multiple perturbations"
    get_CoRa_plot(perturbations_set, CoRa_points_list, title_CoRa_plot)

    # Calculate and show final time point
    current_time = datetime.datetime.now()
    print("Time end start of script: {}".format(current_time))
"""








if __name__ == "__main__":
    main()



