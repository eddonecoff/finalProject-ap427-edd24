"""
Final Project
Name:
Date:
NetID(s):
"""

import matplotlib.pyplot as plt


"""

Actual Function

"""

##def fx(x,y):
##   fx =(2-x-y)*x
##   return fx
##
##def fy(x,y):
##   fy =(3-3*x-y)*y
##   return fy

"""

Example Function

"""

def fx(x,y):
   fx = 2*x - x**2 - x*y
   return fx

def fy(x,y):
   fy = -y + x*y
   return fy

x = []
y = []

"""

Modeling population changes

"""

def pop(x0, y0, dt, time):
	# Initial values
	x.append(x0)
	y.append(y0)

	# Calculate populations at each timestep
	for i in range(time):
		x.append(x[i] + (fx(x[i],y[i])) * dt)
		y.append(y[i] + (fy(x[i],y[i])) * dt)

	return(x,y)

"""

Plotting population changes

"""

x, y = pop(5, 3, 0.01, 1000)
a, b = pop(4, 3, 0.01, 1000)

fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(wspace = 0.5, hspace = 0.3)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(x, 'r-', label='predator')
ax1.plot(y, 'b-', label='prey')
ax1.set_title("Dynamics in time")
ax1.set_xlabel("time")
ax1.grid()
ax1.legend(loc='best')

ax2.plot(x, y, color="blue")
ax2.set_xlabel("x")
ax2.set_ylabel("y")  
ax2.set_title("Phase space")
ax2.grid()


"""
Determine Steady States
"""
fp = []

def find_fixed_points(r):
    for x in range(r):
        for y in range(r):
            if ((fx(x,y) == 0) and (fy(x,y) == 0)):
                fp.append((x,y))
                print('The system has a fixed point in %s,%s' % (x,y))
    return fp

print(find_fixed_points(200))


"""

RK4

"""
# plt.figure()
# 	fig, ax = plt.subplots()
# 	ax.plot(N, errorRK4, label = "Error vs. N")
# 	ax.set_xscale("log")
# 	ax.set_yscale("log")
# 	plt.title("Runge-Kutta 4: Error vs. Number of Timesteps")
# 	plt.xlabel("N (timesteps)")
# 	plt.ylabel("Error")
# 	plt.savefig("RK4.png", bbox_inches = "tight")
# 	plt.close("all")
