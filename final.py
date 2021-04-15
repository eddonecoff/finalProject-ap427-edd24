"""
Final Project
Name:
Date:
NetID(s):
"""

import matplotlib.pyplot as plt


# def dx(x,y):
#     dx = x(e1-a1*x - c1*y)
#     return dx

# def dy(x,y):
#     dy = y(e2-a2*y-c2*x)

def fx(x,y):
   fx =(2-x-y)*x
   return fx

def fy(x,y):
   fy =(3-3*x-y)*y
   return fy

x = []
y = []

def pop(x0, y0, dt, time):
	# Initial values
	x.append(x0)
	y.append(y0)

	# Calculate populations at each timestep
	for i in range(time):
		x.append(x[i] + (fx(x[i],y[i])) * dt)
		y.append(y[i] + (fy(x[i],y[i])) * dt)

	return(x,y)

x, y = pop(5, 3, 0.01, 1000)
v, w = pop(1, 1, 0.01, 1000)

#plot
fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(wspace = 0.5, hspace = 0.3)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(x, 'r-', label='predator')
ax1.plot(y, 'b-', label='prey')
ax1.plot(v, 'g-', label='predator')
ax1.plot(w, 'y-', label='prey')
ax1.set_title("Dynamics in time")
ax1.set_xlabel("time")
ax1.grid()
ax1.legend(loc='best')

ax2.plot(x, y, color="blue")
ax2.set_xlabel("x")
ax2.set_ylabel("y")  
ax2.set_title("Phase space")
ax2.grid()

plt.show()

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