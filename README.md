# HOW TO RUN PROJECT

1) Clone Repo on Your Device
2) Create Virtual Environment
	- python -m venv .venv
	- Windows: .venv\Scripts\activate
	- Mac/Linux: source .venv/bin/activate
3) Install Dependencies
	- pip install -r requirements.txt
4) Run main.py

## Introduction

In the name of God, the most Compassionate, the most Merciful, due to the virtue of seeking knowledge we have built a modest MHD Z Pinch simulator for your enjoyment today

- This project implements a real-time interactive 3D Z-Pinch plasma column simulator based on a simplified single-fluid resistive MHD model with hybrid Eulerian–Lagrangian coupling.
- The simulator is designed to:
	- Model the evolution of a current-driven plasma column
	- Explore confinement and instability behaviour
	- Provide interactive runtime parameter control
	- Visualise grid-based field evolution alongside particle motion
	- Track diagnostics such as energy and confinement proxies
- The architecture prioritises modularity and clarity. Core components (state, grid, particles, solver, diagnostics, rendering, UI) are cleanly separated to allow future expansion, testing, and alternative numerical implementations.
- This project is educational and exploratory in nature. It is not intended to compete with production-level plasma codes, but to provide an intuitive, extensible research-grade framework for understanding Z-Pinch dynamics.

## Sim Architecture

The Structure is 9 files. That's it and you see the skeleton in order on the project subfolders:
1) constants.py
2) parameters.py
3) simstate.py
4) cic_interpolation.py
5) solvers.py
6) diagnostics.py
7) ui_ux.py
8) visualisation.py
9) main.py
	- just download folder and run this and boom whole thing. This will also be the order of the Jupyter notebook's internal sequencing.

What DBZPinch1105 Actually Produces:
- This is NOT a particle based simulation - this is a fluid field based one
- A 3D, time-dependent, resistive MHD simulation of a Z-pinch plasma column, with tracer particles for visualisation
- You start with a cylindrical column of plasma in the centre of the box
	- Uniform density and internal energy inside the cylinder
	- Near-vacuum outside
	- No initial flow
	- No initial magnetic field
	- Tracer particles sprinkled through the plasma to visualise fluid flow but currently stationary
	- The number of tracers and dimensions of the cylinder are initialisable parameters
- You can control these before or during the run to cause the Z Pinch to take place and evolve the plasma
	- Axial current → strength of the pinch
	- Resistivity → how ideal vs diffusive the plasma is
	- Viscosity → how violent vs smooth the motion is
	- External heating
	- External pressure
	- Gravity acting down in negative y direction
- What happens once running starts
	- current produces an azimuthal magnetic field
	- magnetic field strength is proportional to current and a current density is established being b and j respectively
	- The Lorentz force of j x b points radially inwards and plasma accelerates towards the axis and density increases near the centre
	- As plasma compresses, density rises, pressure rises and pressure gradients form outward forces, causing two forces being the magnetic pinch vs thermal pressure
	- Depending on parameters, you get steady confinement or oscillations or violent collapse or diffusion and decay.
	- Resistivity when low causes faster instability formation and higher resistivity means a weaker pinch and damped motion
	- The plasma should move radially and flow axially and form compressive waves, with the CFL auto adjusting the timestep for the sim
	- Tracer particles will reveal the motion by following the grid velocity smoothly via CIC and spiral inwards during pinch and get expelled axially if flows develop and bounce off walls if they hit boundaries.

## Conclusion

Thus is the outcome of our project here.
