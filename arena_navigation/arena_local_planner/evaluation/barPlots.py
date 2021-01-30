import matplotlib
import matplotlib.pyplot as plt
import numpy as np


dwa = [20, 34, 30]
teb = [25, 32, 34]
mpc = [25, 35, 33]
cadrl = [25, 32, 80.4]
arena2d = [40, 24, 80]
sarl = [27, 12, 14]

dwa01 = [60, 54, 60]
teb01 = [65, 52, 74]
mpc01 = [65, 65, 63]
cadrl01 = [95, 92, 90.4]
arena2d01 = [80, 84, 90]
sarl01 = [77, 42, 74]

dwa05 = [10, 24, 10]
teb05 = [15, 22, 24]
mpc05 = [15, 15, 23]
cadrl05 = [15, 22, 60.4]
arena2d05 = [20, 14, 10]
sarl05 = [17, 2, 4]

barWidth = 0.1  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(len(dwa))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]


aluminum = np.array([6.4e-5 , 3.01e-5 , 2.36e-5, 3.0e-5, 7.0e-5, 4.5e-5, 3.8e-5,
                     4.2e-5, 2.62e-5, 3.6e-5])
copper = np.array([4.5e-5 , 1.97e-5 , 1.6e-5, 1.97e-5, 4.0e-5, 2.4e-5, 1.9e-5, 
                   2.41e-5 , 1.85e-5, 3.3e-5 ])
steel = np.array([3.3e-5 , 1.2e-5 , 0.9e-5, 1.2e-5, 1.3e-5, 1.6e-5, 1.4e-5, 
                  1.58e-5, 1.32e-5 , 2.1e-5])


# Calculate the average
aluminum_mean = np.mean(aluminum)
copper_mean = np.mean(copper)
steel_mean = np.mean(steel)

# Calculate the standard deviation
aluminum_std = np.std(aluminum)
copper_std = np.std(copper)
steel_std = np.std(steel)
error1 = 12
error2 = 5
error3 = 7
error4 = 0.8
error5 = 1.6
error6 = 2.5


fig, ax = plt.subplots()
rects1 = ax.bar(r1, dwa01, width=barWidth, label='dwa',  color="lightsteelblue",yerr=error1, alpha=0.5, ecolor='black',capsize=2)
rects2 = ax.bar(r2, teb01, width=barWidth, label='teb',  color="lightpink",yerr=error2, alpha=0.5, ecolor='black',capsize=2)
rects3 = ax.bar(r3, mpc01, width=barWidth, label='mpc',  color="ivory",yerr=error3, alpha=0.5, ecolor='black',capsize=2)
rects4 = ax.bar(r4, cadrl01, width=barWidth, label='cadrl',  color="plum",yerr=error4, alpha=0.5, ecolor='black',capsize=2)
rects5 = ax.bar(r5, arena2d01, width=barWidth, label='arena2d',  color="lightcyan",yerr=error5, alpha=0.5, ecolor='black',capsize=2)
rects6 = ax.bar(r6, sarl01, width=barWidth, label='sarl',  color="palegreen",yerr=error6, alpha=0.5, ecolor='black',capsize=2)

# rects1 = ax.bar(r1, dwa, width=barWidth, label='dwa', color="cornflowerblue",yerr=error)
# rects2 = ax.bar(r2, teb, width=barWidth, label='teb', color="lightcoral",yerr=error)
# rects3 = ax.bar(r3, mpc, width=barWidth, label='mpc', color="khaki",yerr=error)
# rects4 = ax.bar(r4, cadrl, width=barWidth, label='cadrl', color="violet",yerr=error)
# rects5 = ax.bar(r5, arena2d, width=barWidth, label='arena2d', color="cyan",yerr=error)
# rects6 = ax.bar(r6, sarl, width=barWidth, label='sarl', color="lightgreen",yerr=error)


# rects1 = ax.bar(r1, dwa05, width=barWidth, label='dwa',  color="blue",yerr=error)
# rects2 = ax.bar(r2, teb05, width=barWidth, label='teb', color="red",yerr=error)
# rects3 = ax.bar(r3, mpc05, width=barWidth, label='mpc',  color="yellow",yerr=error)
# rects4 = ax.bar(r4, cadrl05, width=barWidth, label='cadrl',  color="darkviolet",yerr=error)
# rects5 = ax.bar(r5, arena2d05, width=barWidth, label='arena2d', color="teal",yerr=error)
# rects6 = ax.bar(r6, sarl05, width=barWidth, label='sarl',  color="lime",yerr=error)






# Add xticks on the middle of the group bars
plt.xlabel('Number of dyn. Obstacles', fontsize=15)
plt.xticks([r + barWidth for r in range(len(dwa))], ['5', '10', '20'])
plt.ylabel('Success Rate', fontsize=15)
plt.title("Success Rate over Number of dynamic Obstacles", fontweight='bold', fontsize=16)
 
# Create legend & Showt graphic
plt.legend(loc='upper right')
plt.savefig('plots/success111.png')
plt.show()

