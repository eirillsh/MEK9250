import pandas            as pd
import matplotlib.pyplot as plt

length_factor = 0.125
timestep = 1/1000 # Timestep size [s]
scheme_type = "implicit"

velocity_element_degree = 2 # Degree of velocity finite element function space
if velocity_element_degree == 1:
    filename_prefix = scheme_type + "_linear_V_elements_dt=" + str(timestep) + "_mesh_LF=" + str(length_factor) + "_"
elif velocity_element_degree == 2:
    filename_prefix = scheme_type + "_quadratic_V_elements_dt=" + str(timestep) + "_mesh_LF=" + str(length_factor) + "_"

df = pd.read_pickle(filename_prefix + "data_pickle.pkl")

# from IPython import embed;embed()

plt.figure(1)
df['C_D'].plot()
plt.title("Drag Coefficient")
plt.xlabel(r"Time $t$ [s]")
plt.show()

plt.figure(2)
df['C_L'].plot()
plt.title("Lift Coefficient")
plt.xlabel(r"Time $t$ [s]")
plt.show()

plt.figure(3)
df['delta_P'].plot()
plt.title("Pressure Difference")
plt.xlabel(r"Time $t$ [s]")
plt.show()