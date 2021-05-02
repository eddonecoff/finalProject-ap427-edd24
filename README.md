# Analysis of Competitive Species Systems of Nonlinear ODEs

Authors: Ethan Donecoff, Arvind Parthasarathy

Math260 Final Project, Spring 2021

## Introduction

For the purposes of our analysis, we investigated systems of two functions denoted x(t) and y(t) for the population of species x and y at time t. Since the population growth rate of one species depends on its own population and the population of the other species, these systems are nonlinear. Given equations for the rate of change of both populations, we used Fourth Order Runge Kutta methods to approximate functions for each species' population over time. For each system, we generated a population vs. time and a phase-plane portrait to showcase the long-term behavior of each system.

## Written Report

The written report `Final Report.pdf` details the analysis of each system, including the mathematical underpinning for calculating the long-term behavior of each system. This includes finding equilibrium points where the rate of change of both populations is 0. Long-term behavior can be then determined by analyzing the eigenvalues of the Jacobian Matrix at each equilibrium point. 

## Python Code

The main file `final.py` contains all python code used for generating the plots shown in the written report. The file begins with 4 functions, each defining a nonlinear system used for analysis. The following functions are also found within the file:

*init_pops*: This function produces a set of random initial population conditions within a specified (non-negative) range.

*find_eqpts*: This function produces a set of all non-negative equilibrium points within a specified range. Since it is impossible to scan all real numbers, it searches for equilibrium points starting at (0,0) and incrementing by 0.01 for each (x,y) in range.

*rk4*: This function computes the simulated population vs. time for each species (x(t) and y(t)) using RK4, given initial conditions x(0) and y(0). The function also allows the user to specify the number of subintervals and the length of the timestep used in the algorithm. 

Finally, the file contains a main function that prints the positive equilibrium points for each system and generates plots given sample initial conditions and specified initial populations. 

### Plots

For each system, two .png files are generated. They are named rk4_npop.png and rk4_nppp.png, for n = 0,1,2,3.

rk4_npop.png shows the population vs. time for each species, while rk4n-ppp shows the phase-plane portrait for the system.

For example, rk4_1pop.png is the population vs. time plot for the system labeled "Function 1", and rk4_1ppp.png is the phase-plane portrait. 

#### Sources

The mathematical basis for finding equilibrium points and analyzing their stability was discussed in class and is also found in any typical course on ODEs. It may also be found in countless textbooks and online resources, including <a href="https://math.libretexts.org/Courses/East_Tennesee_State_University/Book%3A_Differential_Equations_for_Engineers_(Lebl)_Cintron_Copy/8%3A_Nonlinear_Equations/8.2%3A_Stability_and_classi%EF%AC%81cation_of_isolated_critical_points">LibreTexts</a>
or in sections 10.1 and 10.2 of the book "Differential Equations with Boundary Value Problems, Second Edition" by Polking, Boggess and Arnold (This is the book used for Math356 at Duke!).