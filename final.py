"""
Final Project
Names: Arvind Parthasarthy, Ethan Donecoff
Date: April 27, 2021
NetID(s): ap427, edd24
"""

import matplotlib.pyplot as plt
import random
import numpy as np

# Functions 0, 1, 2, and 3 represent the four competitive species systems
# of non-linear ODEs
"""

Function 0

"""
def fx(x,y):
   fx = 2*x - x**2 - x*y
   return fx

def fy(x,y):
   fy = -y + x*y
   return fy

"""

Function 1

"""
def fx1(x,y):
	fx1 = (1-x-y)*x
	return fx1

def fy1(x,y):
	fy1 = (0.75-y-0.5*x)*y
	return fy1

"""

Function 2

"""
def fx2(x,y):
	fx1 = (6-2*x-3*y)*x
	return fx1

def fy2(x,y):
	fy1 = (1-x-y)*y
	return fy1

"""

Function 3

"""
def fx3(x,y):
	fx1 = 0.2*x+y
	return fx1

def fy3(x,y):
	fy1 = 0.2*y-x
	return fy1

"""
init_pops

This function generates a list of random initial populations.

INPUTS:
numPoints: The number of initial populations generated
xRange: Range of initial x populations
yRange: Range of initial y populations

OUTPUTS:
init_pops: List of initial populations
"""
def init_pops(numPoints, xRange, yRange):

	init_pops= []

	for i in range(numPoints):
		x = random.random()*xRange
		y = random.random()*yRange
		init_pops.append((x,y))

	return(init_pops)

"""
find_eqpts

This function finds the equilibrium points (location where fx = fy = 0)
for a system of ODEs by checking all (x,y) in range [0,r]x[0,r], 
incrementing by 0.01.


INPUTS:
xfunc: function for x'
yfunc: function for y'
r: range of non-negative (x,y) values that will be checked

OUTPUTS:
ep: list of non-negative equilibrium points

"""
ep = []

def find_eqpts(xfunc, yfunc, r):
   for x in range(100*r):
       for y in range(100*r):
           if ((xfunc(0.01*x,0.01*y) == 0) and (yfunc(0.01*x,0.01*y) == 0)):
               ep.append((0.01*x,0.01*y))
               print('The system has an equilibrium point at (%s,%s)' % (0.01*x,0.01*y))
   return ep


# Vector function (np array) for growth rate of both populations. 
# fxy corresponds to "Function 0", fxy1 correponds to "Function 1", etc.
def fxy(x,y):
  	return np.array([[fx(x,y)], [fy(x,y)]])

def fxy1(x,y):
	return np.array([[fx1(x,y)], [fy1(x,y)]])

def fxy2(x,y):
	return np.array([[fx2(x,y)], [fy2(x,y)]])

def fxy3(x,y):
	return np.array([[fx3(x,y)], [fy3(x,y)]])

"""
RK4

This function uses RK4 to approximate x(t) and y(t).

INPUTS:
func: Vector function (np.array) that will be simulated
N: Number of subintervals
a: Initial x population x(0)
b: initial y population y(0)
dt: timestep

OUTPUTS:
t: List of times
xn: List of (x,y) population at each time
"""

def rk4(func,N,a,b,dt):
   x0 = np.array([[a],[b]])
   xn = np.zeros((2,N+1))
   xn[:,0,None] = x0
   t = [dt*n for n in range(N+1)]
   
   for n in range(N):
      tn = t[n]
      a = xn[0][n]
      b = xn[1][n]
      k1 = dt*func(a,b)
      k2 = dt*func(a+(k1[0][0])/2, b+(k1[1][0])/2)
      k3 = dt*func(a+(k2[0][0])/2, b+(k2[1][0])/2)
      k4 = dt*func(a+(k3[0][0]), b+(k3[1][0]))

      xn[:,n+1,None] = xn[:,n,None] + 1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4
         
   return t, xn

"""
main function
"""
if __name__ == '__main__':

	"""

	Define two sets of initial populations to be used below for
	modeling the four different systems.

	"""

	# Initial populations in 5x5 square in first quadrant
	init_pops1 = init_pops(100,5,5)

	# Initial populations in 0.5x0.5 square in first quadrant
	init_pops2 = init_pops(100,1,1)


	# Find equilibrium points for each system
	print("First system equilibrium points: ")
	find_eqpts(fx, fy, 5)
	print("\nSecond system equilibrium points: ")
	find_eqpts(fx1, fy1, 5)
	print("\nThird system equilibrium points: ")
	find_eqpts(fx2, fy2, 5)
	print("\nFourth system equilibrium points: ")
	find_eqpts(fx3, fy3, 5)

	# fx, fy Plots

	# Phase-Plane Portrait
	plt.figure()
	fig, ax = plt.subplots()
	plt.title("Phase-Plane Portrait Using RK4")
	plt.xlabel("Prey Population")
	plt.ylabel("Predator Population")

	# Draw portrait for 20 random initial pops using RK4
	for i in range(len(init_pops1)):
		a, b = init_pops1[i]
		t, xn = rk4(fxy, 1000, a, b, 0.01)
		ax.plot(xn[0], xn[1])

	plt.savefig("rk4_0ppp.png", bbox_inches = "tight")
	plt.close("all")

	# Example population plot with initial (5,3)
	t, xn = rk4(fxy, 1000, 5, 3, 0.01)
	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(t, xn[0], label = "Prey Population")
	ax.plot(t, xn[1], label = "Predator Population")
	ax.legend(loc = "upper right")
	plt.title("Species Population over Time")
	plt.xlabel("Time")
	plt.ylabel("Population")
	plt.savefig("rk4_0pop.png", bbox_inches = "tight")
	plt.close("all")


	# fx1, fy1 Plots

	# Phase-Plane Portrait
	plt.figure()
	fig, ax = plt.subplots()
	plt.title("Phase-Plane Portrait Using RK4")
	plt.xlabel("Prey Population")
	plt.ylabel("Predator Population")

	# Draw portrait for 20 random initial pops using RK4
	for i in range(len(init_pops2)):
		a1, b1 = init_pops2[i]
		t1, xn1 = rk4(fxy1, 1000, a1, b1, 0.01)
		ax.plot(xn1[0], xn1[1])

	plt.savefig("rk4_1ppp.png", bbox_inches = "tight")
	plt.close("all")

	# Example population plot with initial (5,3)
	t1, xn1 = rk4(fxy1, 1000, 5, 3, 0.01)
	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(t1, xn1[0], label = "Population 1")
	ax.plot(t1, xn1[1], label = "Population 2")
	ax.legend(loc = "upper right")
	plt.title("Species Population over Time")
	plt.xlabel("Time")
	plt.ylabel("Population")
	plt.savefig("rk4_1pop.png", bbox_inches = "tight")
	plt.close("all")

	# fx2, fy2 Plots

	# Phase-Plane Portrait
	plt.figure()
	fig, ax = plt.subplots()
	plt.title("Phase-Plane Portrait Using RK4")
	plt.xlabel("Prey Population")
	plt.ylabel("Predator Population")

	# Draw portrait for 20 random initial pops using RK4
	for i in range(len(init_pops1)):
		a2, b2 = init_pops1[i]
		t2, xn2 = rk4(fxy2, 1000, a2, b2, 0.01)
		ax.plot(xn2[0], xn2[1])

	plt.savefig("rk4_2ppp.png", bbox_inches = "tight")
	plt.close("all")

	# Example population plot with initial (5,3)
	t2, xn2 = rk4(fxy2, 1000, 5, 3, 0.01)
	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(t2, xn2[0], label = "Population 1")
	ax.plot(t2, xn2[1], label = "Population 2")
	ax.legend(loc = "upper right")
	plt.title("Species Population over Time")
	plt.xlabel("Time")
	plt.ylabel("Population")
	plt.savefig("rk4_2pop.png", bbox_inches = "tight")
	plt.close("all")

	# fx3, fy3 Plots

	# Phase-Plane Portrait
	plt.figure()
	fig, ax = plt.subplots()
	plt.title("Phase-Plane Portrait Using RK4")
	plt.xlabel("Prey Population")
	plt.ylabel("Predator Population")

	# Draw portrait for 20 random initial pops using RK4
	for i in range(len(init_pops1)):
		a3, b3 = init_pops1[i]
		t3, xn3 = rk4(fxy3, 1000, a3, b3, 0.01)
		ax.plot(xn3[0], xn3[1])

	plt.savefig("rk4_3ppp.png", bbox_inches = "tight")
	plt.close("all")

	# Example population plot with initial (5,3)
	t3, xn3 = rk4(fxy3, 1500, 5, 3, 0.01)
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