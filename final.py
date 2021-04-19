"""
Final Project
Names: Arvind Parthasarthy, Ethan Donecoff
Date:
NetID(s): ap427, edd24
"""

import matplotlib.pyplot as plt
import random
import numpy as np

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

"""

Another Function

"""

# This system shows a source at (0,0)
def fx1(x,y):
	fx1 = (1-x-y)*x
	return fx1

def fy1(x,y):
	fy1 = (0.75-y-0.5*x)*y
	return fy1

# This system shows a saddle at (0,4) 
def fx2(x,y):
	fx1 = (6-2*x-3*y)*x
	return fx1

def fy2(x,y):
	fy1 = (1-x-y)*y
	return fy1

def fx3(x,y):
	fx1 = 0.2*x+y
	return fx1

def fy3(x,y):
	fy1 = 0.2*y-x
	return fy1

"""

Modeling population changes

"""

def pop(x0, y0, dt, time):
	x = []
	y = []

	# Initial values
	x.append(x0)
	y.append(y0)

	# Calculate populations at each timestep
	for i in range(-time, time):
		x.append(x[i+time] + (fx3(x[i+time],y[i+time])) * dt)
		y.append(y[i+time] + (fy3(x[i+time],y[i+time])) * dt)

	return(x,y)

# p1, p2 = pop(-0.5, -0.5, 0.1, 10)
# print(len(p1))

"""

List of Initial Populations

"""
def init_pops(numPoints, xRange, yRange):

	init_pops= []

	for i in range(numPoints):
		x = random.random()*xRange
		y = random.random()*yRange
		init_pops.append((x,y))

	return(init_pops)

"""

Plotting population changes

"""

# x1, y1 = pop(0, 4, 0.01, 1000)
# x2, y2 = pop(4, 0, 0.01, 1000)
# timelist = [0.01*i for i in range(1000)]

# fig = plt.figure(figsize=(15,5))
# fig.subplots_adjust(wspace = 0.5, hspace = 0.3)
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)

# ax1.plot(timelist, x1, 'r-', label='predator')
# ax1.plot(timelist, y1, 'b-', label='prey')
# ax1.plot(timelist, x2, label = "predator 2")
# ax1.plot(timelist, y2, label = "prey 2")

init_pops1 = init_pops(100,5,5)
init_pops2 = init_pops(100,0.5,0.5)

# for i in range(len(init_pops)):
# 	a, b = init_pops[i]
# 	x, y = pop(a, b, 0.01, 1000)
# 	ax1.plot(timelist, x)
# 	ax1.plot(timelist, y)

# 	ax2.plot(x, y)

# ax1.set_title("Dynamics in time")
# ax1.set_xlabel("time")
# ax1.grid()
# ax1.legend(loc='best')

# ax2.plot(x1, y1, color="blue")
# ax2.plot(x2, y2)
# ax2.set_xlabel("x")
# ax2.set_ylabel("y")  
# ax2.set_title("Phase space")
# ax2.grid()

# plt.show()


"""

Determine Steady States

"""
fp = []

def find_fixed_points(r):
   for x in range(r):
       for y in range(r):
           if ((fx1(x,y) == 0) and (fy1(x,y) == 0)):
               fp.append((x,y))
               print('The system has a fixed point in %s,%s' % (x,y))
   return fp

print(find_fixed_points(1000))


"""

RK4

"""

# vector function for growth rate of both populations
def fxy(x,y):
  	return np.array([[fx(x,y)], [fy(x,y)]])

def fxy1(x,y):
	return np.array([[fx1(x,y)], [fy1(x,y)]])

def fxy2(x,y):
	return np.array([[fx2(x,y)], [fy2(x,y)]])

def fxy3(x,y):
	return np.array([[fx3(x,y)], [fy3(x,y)]])

# RK4 simulation of competitive species system
def rk4(N,a,b,dt):
   x0 = np.array([[a],[b]])
   xn = np.zeros((2,N+1))
   xn[:,0,None] = x0
   t = [dt*n for n in range(N+1)]
   
   for n in range(N):
      tn = t[n]
      a = xn[0][n]
      b = xn[1][n]
      k1 = dt*fxy(a,b)
      k2 = dt*fxy(a+(k1[0][0])/2, b+(k1[1][0])/2)
      k3 = dt*fxy(a+(k2[0][0])/2, b+(k2[1][0])/2)
      k4 = dt*fxy(a+(k3[0][0]), b+(k3[1][0]))

      xn[:,n+1,None] = xn[:,n,None] + 1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4
         
   return t, xn

def rk4_1(N,a,b,dt):
   x0 = np.array([[a],[b]])
   xn = np.zeros((2,N+1))
   xn[:,0,None] = x0
   t = [dt*n for n in range(N+1)]
   
   for n in range(N):
      tn = t[n]
      a = xn[0][n]
      b = xn[1][n]
      k1 = dt*fxy1(a,b)
      k2 = dt*fxy1(a+(k1[0][0])/2, b+(k1[1][0])/2)
      k3 = dt*fxy1(a+(k2[0][0])/2, b+(k2[1][0])/2)
      k4 = dt*fxy1(a+(k3[0][0]), b+(k3[1][0]))

      xn[:,n+1,None] = xn[:,n,None] + 1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4
         
   return t, xn


def rk4_2(N,a,b,dt):
   x0 = np.array([[a],[b]])
   xn = np.zeros((2,N+1))
   xn[:,0,None] = x0
   t = [dt*n for n in range(N+1)]
   
   for n in range(N):
      tn = t[n]
      a = xn[0][n]
      b = xn[1][n]
      k1 = dt*fxy2(a,b)
      k2 = dt*fxy2(a+(k1[0][0])/2, b+(k1[1][0])/2)
      k3 = dt*fxy2(a+(k2[0][0])/2, b+(k2[1][0])/2)
      k4 = dt*fxy2(a+(k3[0][0]), b+(k3[1][0]))

      xn[:,n+1,None] = xn[:,n,None] + 1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4
         
   return t, xn

def rk4_3(N,a,b,dt):
   x0 = np.array([[a],[b]])
   xn = np.zeros((2,N+1))
   xn[:,0,None] = x0
   t = [dt*n for n in range(N+1)]
   
   for n in range(N):
      tn = t[n]
      a = xn[0][n]
      b = xn[1][n]
      k1 = dt*fxy3(a,b)
      k2 = dt*fxy3(a+(k1[0][0])/2, b+(k1[1][0])/2)
      k3 = dt*fxy3(a+(k2[0][0])/2, b+(k2[1][0])/2)
      k4 = dt*fxy3(a+(k3[0][0]), b+(k3[1][0]))

      xn[:,n+1,None] = xn[:,n,None] + 1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4
         
   return t, xn

# Example function

# Phase-Plane Portrait
plt.figure()
fig, ax = plt.subplots()
plt.title("Phase-Plane Portrait Using RK4")
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")

# Draw portrait for 20 random initial pops using RK4
for i in range(len(init_pops1)):
	a, b = init_pops1[i]
	t, xn = rk4(1000, a, b, 0.01)
	ax.plot(xn[0], xn[1])

plt.savefig("rk4ppp.png", bbox_inches = "tight")
plt.close("all")

# Example population plot with initial (5,3)
t, xn = rk4(1000, 5, 3, 0.01)
plt.figure()
fig, ax = plt.subplots()
ax.plot(t, xn[0], label = "Prey Population")
ax.plot(t, xn[1], label = "Predator Population")
ax.legend(loc = "upper right")
plt.title("Species Population over Time")
plt.xlabel("Time")
plt.ylabel("Population")
plt.savefig("rk4pop.png", bbox_inches = "tight")
plt.close("all")


# fxy1: nodal source near (0,0)

# Phase-Plane Portrait
plt.figure()
fig, ax = plt.subplots()
plt.title("Phase-Plane Portrait Using RK4")
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")

# Draw portrait for 20 random initial pops using RK4
for i in range(len(init_pops2)):
	a1, b1 = init_pops2[i]
	t1, xn1 = rk4_1(1000, a1, b1, 0.01)
	ax.plot(xn1[0], xn1[1])

plt.savefig("rk4_1ppp.png", bbox_inches = "tight")
plt.close("all")

# Example population plot with initial (5,3)
t1, xn1 = rk4_1(1000, 5, 3, 0.01)
plt.figure()
fig, ax = plt.subplots()
ax.plot(t1, xn1[0], label = "Prey Population")
ax.plot(t1, xn1[1], label = "Predator Population")
ax.legend(loc = "upper right")
plt.title("Species Population over Time")
plt.xlabel("Time")
plt.ylabel("Population")
plt.savefig("rk4_1pop.png", bbox_inches = "tight")
plt.close("all")

# fxy2: 

# Phase-Plane Portrait
plt.figure()
fig, ax = plt.subplots()
plt.title("Phase-Plane Portrait Using RK4")
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")

# Draw portrait for 20 random initial pops using RK4
for i in range(len(init_pops1)):
	a2, b2 = init_pops1[i]
	t2, xn2 = rk4_2(1000, a2, b2, 0.01)
	ax.plot(xn2[0], xn2[1])

plt.savefig("rk4_2ppp.png", bbox_inches = "tight")
plt.close("all")

# Example population plot with initial (5,3)
t2, xn2 = rk4_2(1000, 5, 3, 0.01)
plt.figure()
fig, ax = plt.subplots()
ax.plot(t2, xn2[0], label = "Prey Population")
ax.plot(t2, xn2[1], label = "Predator Population")
ax.legend(loc = "upper right")
plt.title("Species Population over Time")
plt.xlabel("Time")
plt.ylabel("Population")
plt.savefig("rk4_2pop.png", bbox_inches = "tight")
plt.close("all")

# fxy3: experimental

# Phase-Plane Portrait
plt.figure()
fig, ax = plt.subplots()
plt.title("Phase-Plane Portrait Using RK4")
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")

# Draw portrait for 20 random initial pops using RK4
for i in range(len(init_pops1)):
	a3, b3 = init_pops1[i]
	t3, xn3 = rk4_3(1000, a3, b3, 0.01)
	ax.plot(xn3[0], xn3[1])

plt.savefig("rk4_3ppp.png", bbox_inches = "tight")
plt.close("all")

# Example population plot with initial (5,3)
t3, xn3 = rk4_3(1500, 5, 3, 0.01)
plt.figure()
fig, ax = plt.subplots()
ax.plot(t3, xn3[0], label = "Predator Population")
ax.plot(t3, xn3[1], label = "Prey Population")
ax.legend(loc = "upper right")
plt.title("Species Population over Time")
plt.xlabel("Time")
plt.ylabel("Population")
plt.savefig("rk4_3pop.png", bbox_inches = "tight")
plt.close("all")