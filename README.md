# S4_Pillar_Simulations
Data_Utils file:
  Has utility functions for the simulation. Can import things that save your data to a csv, and make different plots from csv or plot the geometry of a simulation. 

optimized_simulation:
  Contains the code to run a 2x2 grid of pillars under wavelength ranges from 3-5. The period size is 2.7u, with radii sizes ranging from 10%-90% of period size. Returns the transmission and reflection values from the experiment.

  Notes after Tuesday April 15th:
  -Ran the experiment, with the max transmission not being found. There are a few ways to fix this:
    1. Make the EA search more in the space
    2. Run the EA for longer
    3. Update the fitness function to be more in depth, more than 3 decimal points. 
    4. Maybe emphasize the max transmission more than 80% of max transmission

  -Should also graph the results to see if they are better than they seem. 

