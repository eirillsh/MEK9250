import pandas            as pd
import matplotlib.pyplot as plt

sinusoidal = False
if sinusoidal:
    case_str = "case_2D-3_"
else:
    case_str = "case_2D-2_"

df = pd.read_pickle(case_str + "data_pickle.pkl")

from IPython import embed;embed()

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