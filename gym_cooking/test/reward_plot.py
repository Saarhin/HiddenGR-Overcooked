import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import outset as otst

input_file = "./policies_Summary_seed3/rewards.csv"

file = pd.read_csv(input_file)
file_array = np.array(file)

# adapted from https://matplotlib.org/stable/gallery/
x,y = file_array[:,0], file_array[:,1]

# print(np.max(file_array[:,1]))
# plt.plot(file_array[:,0], file_array[:,1])
# plt.show()



# 3 axes grid: source plot and two zoom frames
grid = otst.OutsetGrid([(3000, -25, 3500, -15)])  # frame coords
grid.broadcast(
    plt.plot,  # run plotter over all axes
    x,
    y,  # line coords
    c="mediumblue",
    zorder=-1,
)  # kwargs forwarded to plt.plot

plt.xlabel("Training Episodes")
plt.ylabel("Return")




grid.marqueeplot()  # set axlims and render marquee annotations
plt.show()
