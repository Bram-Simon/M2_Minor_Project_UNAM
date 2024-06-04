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

    U, W, C, Y = y

    dU_dt = mu_U * Y - (gamma + gamma_U) * U - eta_plus * U * W + (eta_0 + gamma_W) * C
    dW_dt = mu_W - (gamma + gamma_W) * W - eta_plus * U * W + (eta_0 + gamma_U) * C
    dC_dt = eta_plus * U * W - (gamma + eta_0 + eta_minus + gamma_U + gamma_W) * C
    dY_dt = mu_Y * W - (gamma + gamma_Y) * Y
    #dY_dt = mu_Y * (W + C) - (gamma + gamma_Y) * Y

    return [dU_dt, dW_dt, dC_dt, dY_dt]




def system_without_feedback(t, y, mu_U, mu_W, mu_Y, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss):

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




def calc_CoRa(Y_fad, Y_fad_NF):
    CoRa_values = []
    for i in range(len(Y_fad) - 1):
        #CoRa = (np.log(Y_fad[i + 1]) - np.log(Y_fad[i])) / (np.log(Y_fad_NF[i + 1]) - np.log(Y_fad_NF[i]))
        #CoRa = np.log(Y_fad[i + 1] / Y_fad[i]) / np.log(Y_fad_NF[i + 1] / Y_fad_NF[i + 1])          # FORMULA MAY NOT BE CORRECT YET!!!
        CoRa = (np.log(Y_fad[i + 1]) - np.log(Y_fad[i])) / (np.log(Y_fad_NF[i + 1]) - np.log(Y_fad_NF[i]))
        
        CoRa_values.append(CoRa)
    return CoRa_values


def calc_CoRa_point(ss, ss_NF, ss_pp, ss_pp_NF):
    
    CoRa = (np.log(ss_pp) - np.log(ss)) / (np.log(ss_pp_NF) - np.log(ss_NF))

    return CoRa



# Initial conditions
U0 = 1.0
W0 = 1.0
C0 = 1.0
Y0 = 1.0
y0 = [U0, W0, C0, Y0]

# extra initial condition Ystar for no-feedback network
Ystar0 = 1.0
y0_NF = [U0, W0, C0, Y0, Ystar0]

# Time points where solution is computed. Make sure that the time span is large enough to reach steady state!!
t_span = (0, 50000)  # from t=0 to t=10
t_eval = np.linspace(t_span[0], t_span[1], 50000)

# Parameters
gamma = 1e-4  # min^-1
gamma_U = 1e-4  # min^-1
gamma_W = 1e-4  # min^-1
mu_U = 0.125  # min^-1
mu_W = 0.1  # nM min^-1
mu_Y = 0.125  # min^-1
eta_0 = 1e-4  # min^-1
eta_plus = 0.0375  # nM^-1 min^-1
eta_minus = 0.5  # min^-1
gamma_Y = 1.0  # min^-1


# Extra parameters system without feedback      QUESTION: these parameters are introduced in the equations (supplementary info CoRa paper). However, they are not mentioned in S6 "Parameter values ..."
# Therefore, where can we find the values used for these parameters to create the figure in the paper? - MY THOUGHT: are they by definition the same as for the Y (not Ystar) parameters?
mu_Ystar = mu_Y #0.125 # Set equal to non-starred parameter
gamma_Ystar = gamma_Y #1.0 # set equal to non-starred parameter




# Define range of mu_Y values
#mu_Y = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# Range of mu_Y values on a logarithmic scale
mu_Y_values = np.logspace(-2, 2, 50)


cora_values_list = []

current_mu_Y_value = mu_Y
#for mu_Y_value in mu_Y_values:
    
#current_mu_Y_value = mu_Y_value

# Solve ODEs system with feedback
sol_FB = solve_ivp(system_with_feedback, t_span, y0, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y), t_eval=t_eval)

# Get the steady state values by selecting the final value of the investigated range. Make sure that the time span is long enough!!
print(sol_FB.y[1][-1])
U_ss = sol_FB.y[0][-1]
W_ss = sol_FB.y[1][-1]
C_ss = sol_FB.y[2][-1]
Y_ss = sol_FB.y[3][-1]

steady_state_conditions = [U_ss, W_ss, C_ss, Y_ss]

#C_ss = sol_FB.y[2][-1]
#W_ss = solve_ivp(system_with_feedback, t_span, y0, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y), t_eval=t_eval)


# Solve system without feedback
sol_NF = solve_ivp(system_without_feedback, t_span, y0_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss), t_eval=t_eval)
#sol_NF = solve_ivp(system_without_feedback, t_span, y0_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss, C_ss), t_eval=t_eval)


# Get steady states in NF system, these must be the same as in the FB system, before feedback
print(sol_FB.y[1][-1])
U_ss_NF = sol_NF.y[0][-1]
W_ss_NF = sol_NF.y[1][-1]
C_ss_NF = sol_NF.y[2][-1]
Y_ss_NF = sol_NF.y[3][-1]

print("The two values above must be the same (steady states before perturbation)")


Y_values_FB_system = sol_FB.y[3]
Y_values_NF_system = sol_NF.y[3]

CoRa_value_set = calc_CoRa(Y_values_FB_system, Y_values_NF_system)

# QUESTION: We are taking the mean now, but how are the different CoRa values for different time points corresponding to each individual mu_Y combined?
avg_CoRa = np.mean(CoRa_value_set)

cora_values_list.append(avg_CoRa)



# Plot results
plt.plot(sol_FB.t, sol_FB.y[0], label='U')
plt.plot(sol_FB.t, sol_FB.y[1], label='W')
plt.plot(sol_FB.t, sol_FB.y[2], label='C')
plt.plot(sol_FB.t, sol_FB.y[3], label='Y')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.title('Solution to the system of ODEs feedback system')
plt.savefig(os.path.join(output_dir, "Concentration_values_system_with_feedback_ATF.png"))
plt.show()
plt.close()

# Plot results
plt.plot(sol_NF.t, sol_NF.y[0], label='U')
plt.plot(sol_NF.t, sol_NF.y[1], label='W')
plt.plot(sol_NF.t, sol_NF.y[2], label='C')
plt.plot(sol_NF.t, sol_NF.y[3], label='Y')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.title('Solution to the system of ODEs no-feedback system')
plt.savefig(os.path.join(output_dir, "Concentration_values_system_NO_feedback_ATF.png"))
plt.show()
plt.close()

"""
print(sol_FB.y[3])
print("------------------")
print(sol_NF.y[3])

Y_values_FB_system = sol_FB.y[3]
Y_values_NF_system = sol_NF.y[3]

CoRa_values = calc_CoRa(Y_values_FB_system, Y_values_NF_system)

print(CoRa_values)

plt.plot(sol_NF.t[:-1], CoRa_values)
plt.xlabel('Time')
plt.ylabel('CoRa values')
plt.legend()
plt.title('CoRa values of antithetic feedback motif')
plt.savefig(os.path.join(output_dir, 'CoRa_values_antithetic_feedback_network.png'))
plt.show()
plt.close()
"""





# Now we solve again, but using the steady state values as the initial conditions

# Solve ODEs system with feedback
sol_FB = solve_ivp(system_with_feedback, t_span, steady_state_conditions, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y), t_eval=t_eval)

Ystar_ss = Y_ss
#steady_state_conditions_NF = steady_state_conditions.append(Ystar_ss)
steady_state_conditions_NF = [U_ss, W_ss, C_ss, Y_ss, Ystar_ss]
print("steady state conditions NF are: {}".format(steady_state_conditions_NF))

# Solve system without feedback
sol_NF = solve_ivp(system_without_feedback, t_span, steady_state_conditions_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss), t_eval=t_eval)


Y_values_FB_system = sol_FB.y[3]
Y_values_NF_system = sol_NF.y[3]

CoRa_value_set = calc_CoRa(Y_values_FB_system, Y_values_NF_system)

# QUESTION: We are taking the mean now, but how are the different CoRa values for different time points corresponding to each individual mu_Y combined?
avg_CoRa = np.mean(CoRa_value_set)

cora_values_list.append(avg_CoRa)



# Plot results
plt.plot(sol_FB.t, sol_FB.y[0], label='U')
plt.plot(sol_FB.t, sol_FB.y[1], label='W')
plt.plot(sol_FB.t, sol_FB.y[2], label='C')
plt.plot(sol_FB.t, sol_FB.y[3], label='Y')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.title('Solution to the system of ODEs feedback system in steady state')
plt.savefig(os.path.join(output_dir, "Concentration_values_system_with_feedback_ATF_in_steady_state.png"))
plt.show()
plt.close()

# Plot results
plt.plot(sol_NF.t, sol_NF.y[0], label='U')
plt.plot(sol_NF.t, sol_NF.y[1], label='W')
plt.plot(sol_NF.t, sol_NF.y[2], label='C')
plt.plot(sol_NF.t, sol_NF.y[3], label='Y')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.title('Solution to the system of ODEs no-feedback system in steady state')
plt.savefig(os.path.join(output_dir, "Concentration_values_system_NO_feedback_ATF_in_steady_state.png"))
plt.show()
plt.close()








# Now we apply a perturbation and calculate again

mu_Y_perturbation = 0.5
current_mu_Y_value = mu_Y_perturbation



# Solve ODEs system with feedback
sol_FB = solve_ivp(system_with_feedback, t_span, steady_state_conditions, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y), t_eval=t_eval)

Ystar_ss = Y_ss
#steady_state_conditions_NF = steady_state_conditions.append(Ystar_ss)
steady_state_conditions_NF = [U_ss, W_ss, C_ss, Y_ss, Ystar_ss]
print("steady state conditions NF are: {}".format(steady_state_conditions_NF))

# Solve system without feedback
sol_NF = solve_ivp(system_without_feedback, t_span, steady_state_conditions_NF, args=(mu_U, mu_W, current_mu_Y_value, gamma, gamma_U, gamma_W, eta_plus, eta_0, eta_minus, gamma_Y, mu_Ystar, gamma_Ystar, W_ss), t_eval=t_eval)


Y_values_FB_system = sol_FB.y[3]
Y_values_NF_system = sol_NF.y[3]

CoRa_value_set = calc_CoRa(Y_values_FB_system, Y_values_NF_system)

# QUESTION: We are taking the mean now, but how are the different CoRa values for different time points corresponding to each individual mu_Y combined?
avg_CoRa = np.mean(CoRa_value_set)

cora_values_list.append(avg_CoRa)



# Plot results
plt.plot(sol_FB.t, sol_FB.y[0], label='U')
plt.plot(sol_FB.t, sol_FB.y[1], label='W')
plt.plot(sol_FB.t, sol_FB.y[2], label='C')
plt.plot(sol_FB.t, sol_FB.y[3], label='Y')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.title('Solution to the system of ODEs feedback system after_perturbation')
plt.savefig(os.path.join(output_dir, "Concentration_values_system_with_feedback_ATF_after_perturbation.png"))
plt.show()
plt.close()

# Plot results
plt.plot(sol_NF.t, sol_NF.y[0], label='U')
plt.plot(sol_NF.t, sol_NF.y[1], label='W')
plt.plot(sol_NF.t, sol_NF.y[2], label='C')
plt.plot(sol_NF.t, sol_NF.y[3], label='Y')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.title('Solution to the system of ODEs no-feedback system after_perturbation')
plt.savefig(os.path.join(output_dir, "Concentration_values_system_NO_feedback_ATF_after_perturbation.png"))
plt.show()
plt.close()



# Get the steady state values post perturbation (pp) by selecting the final value of the investigated range. Make sure that the time span is long enough!!
print(sol_FB.y[1][-1])
U_ss_pp = sol_FB.y[0][-1]
W_ss_pp = sol_FB.y[1][-1]
C_ss_pp = sol_FB.y[2][-1]
Y_ss_pp = sol_FB.y[3][-1]

print(sol_NF.y[1][-1])
U_ss_pp_NF = sol_NF.y[0][-1]
W_ss_pp_NF = sol_NF.y[1][-1]
C_ss_pp_NF = sol_NF.y[2][-1]
Y_ss_pp_NF = sol_NF.y[3][-1]


CoRa_point = calc_CoRa_point(Y_ss, Y_ss_NF, Y_ss_pp, Y_ss_pp_NF)
print("The CoRa value calculated for the current system is {}".format(CoRa_point))






current_time = datetime.datetime.now()
print("Time end start of script: {}".format(current_time))



"""
print(cora_values_list)

plt.figure(figsize=(10, 6))
#plt.plot(mu_Y_values, cora_values_list, marker='o', linestyle='-', label='CoRa')
plt.plot(sol_FB.t[-1], cora_values_list, marker='o', linestyle='-', label='CoRa')
plt.xscale('log')
plt.yscale('linear')
plt.xlabel('muY')
plt.ylabel('CoRa values')
plt.legend()
plt.title('CoRa values of antithetic feedback motif against muY concentration')
plt.grid(True, which="both", ls="--")
plt.savefig(os.path.join(output_dir, 'CoRa_values_antithetic_feedback_network_CoRa_against_muY_concentration.png'))
plt.show()
plt.close()
"""



