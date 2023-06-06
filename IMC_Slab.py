import random
import math
import numpy as np
import mpmath as mp
import matplotlib as mpl
from matplotlib import pyplot as plt

# Assuming isotropically scattering, homogeneous slab with isotropic source.

# ? = double check

# Discretization and other system parameters/geometry
total_N = 20000 # Put a resonable number
total_length = 4 # Put a resonable number
total_cells = 10 # Put a resonable number
time_step_total = 10 # Put a reasonable number
dt = 2e-11 # Put a reasonable number ---> will track particles through 1e-7 seconds (time_step_total * dt)
dx = total_length / total_cells
N_census_particles = 0 # Moved into loop as well
N_source_particles_initial = 1000

# Constants
c = 3.0e10  # speed of light [cm/s]
h = 4.1356675e-18  # Plancs constant [KeV s] ?
pi = 3.1415
radiation_constant_a = 8 * pi ** 5 / (15 * h **3 * c ** 3) #[1 / keV^3 m^3] ?
gamma = 27
scattering_fraction = 0.944
scattering_fraction_compliment = 1 - scattering_fraction
b = (60 * radiation_constant_a * c * gamma * dt * scattering_fraction_compliment) / ((pi ** 4) * scattering_fraction) 
# Wollabar uses heat capacity, fcmimc uses 'b' and its tempurature independent (defined as it is above). Why is it time dependent?


# Arrays 
cell_temp = np.zeros(total_cells)
initial_temp = np.zeros(total_cells) + 0.001
initial_surface_temp = 1
sigma_p = np.zeros(total_cells) #Planck opacity / Planck mean absorption cross section
beta_factor = np.zeros(total_cells)
fleck_factor = np.zeros(total_cells)
#deposited_energy = np.zeros(total_cells) # end of time step energy deposition, moved into loop
cell_energy = np.zeros(total_cells) # energy radiated by cell per timestep 
cell_probability = np.zeros(total_cells) # for emmision of cell energy probability
N_cell_particles = np.zeros(total_cells)
internal_cell_position = np.linspace(0,total_length,total_cells + 1)
radiation_energy_density = np.zeros(total_cells)
energy_increase = np.zeros(total_cells)
properties = []

# Planck spectrum sampling
def planck_spectrum_sample():
    n = 1
    random1 = random.random()
    partial_sum = 1
    while True:
        if random1 <= 90 * partial_sum * (pi ** -4):
            random1 = random.random()
            random2 = random.random()
            random3 = random.random()
            random4 = random.random()
            x_for_frequency = -math.log(random1 * random2 * random3 * random4) / n
            break
        n += 1
        partial_sum += n ** -4
    return x_for_frequency
# redundancy in if statement, look at again
print(planck_spectrum_sample())

# Initilizing 
cell_temp[:] = initial_temp[:]  
current_time = 0

for step in range(0,time_step_total):
    
    end_of_timestep = step + dt
    
    ## -----------------------------------------------
    ## Update sigma_p, beta_factor, and fleck_factor
    ## based on new temperature
    ## -----------------------------------------------
    
    # Planck opacity
    sigma_p[:] = 15 * gamma * (pi ** -4) * (cell_temp[:] ** -3)
    
    # Beta factor 
    beta_factor[:] = (4 * radiation_constant_a * cell_temp[:] ** 3) / b
    
    # Fleck factor 
    fleck_factor[:] = 1 / (1 + beta_factor[:] * c * dt * sigma_p[:])
    
    ##------------------------------------------------
    ## How many particles do we source this step?
    ## What is the energy sources and their emission
    ## probabilities? 
    ##------------------------------------------------
    
    # Initilizing 
    N_source_particles = N_source_particles_initial 
    
    # keeping the system from blowing up with particles
    if (N_source_particles + N_census_particles) > total_N:
        N_source_particles = total_N - N_census_particles - total_cells - 1
        
    # Energy radiated by surface (E_s in f&c and E_B in wollabar)
    surface_energy = (radiation_constant_a * c * dt * initial_surface_temp ** 4) / 4
    
    # Energy radiated by cell (E_{j-1/2})
    cell_energy[:] = fleck_factor[:] * sigma_p[:] * radiation_constant_a * c * dx * dt * cell_temp[:] ** 4 
    
    # Total energy radiated (E_totd)
    total_energy = surface_energy + sum(cell_energy[:])
    
    # Emission probabilities
    surface_emission_probability = surface_energy / total_energy
    cell_probability[:] = np.cumsum(cell_energy[:] / sum(cell_energy[:]))
    
    # find number of particles emitted from cells and surface
    N_surface_particles = 0
    N_cell_particles = np.zeros(total_cells)
    
    for i in range(N_source_particles):
        if random.random() <= surface_emission_probability:
            N_surface_particles += 1
        else:
            other_random = random.random()
            for cell_number in range(total_cells):
                if other_random <= cell_probability[cell_number]:
                    N_cell_particles[cell_number] += 1
                    break
                
    # creating surface particles
    for i in range(N_surface_particles):
        
        energy = surface_energy / N_surface_particles # --> Uniformly weighted
        starting_energy = surface_energy / N_surface_particles # Used for a check later
        starting_x_position = 0 # Place at the left of the slab
        mu = math.sqrt(random.random()) # Sample direction
        starting_time = current_time + random.random() * dt 
        frequency = initial_surface_temp * planck_spectrum_sample() / h 
        properties.append([starting_time,0,starting_x_position,mu,frequency,energy,starting_energy])
        # Creating a list to pull properties from and use in particle loop
        
    # creating cell particles 
    for cell_number in range(total_cells):
        if N_cell_particles[cell_number] <= 0:
            continue # Gotta skip these guys
        energy = cell_energy[cell_number] / N_cell_particles[cell_number] 
        starting_energy = cell_energy[cell_number] / N_cell_particles[cell_number]
        for i in range(int(N_cell_particles[cell_number])):
            starting_x_position = internal_cell_position[cell_number] + random.random() * dx
            mu = random.uniform(-1,1) # isotropic
            starting_time = current_time + random.random() * dt
            frequency = cell_temp[cell_number] * abs(math.log(random.random())) / h
            properties.append([starting_time,cell_number,starting_x_position,mu,frequency,energy,starting_energy])

        
    for particle_number in range(len(properties)):
        
        (starting_time,cell_number,starting_x_position,mu,frequency,energy,starting_energy) = properties[particle_number]
        deposited_energy = np.zeros(total_cells) # end of time step energy deposition
        N_census_particles = 0
        
        while True:
            
            # Macroscopic cross section ##CHECK THIS EQUATION IF STUFF IS WEIRD
            sigma = gamma * (1 / (h * frequency) ** 3) * (1 - math.exp(- h * frequency / cell_temp[cell_number]))

            # Fleck factor times sigma
            sigma_f = fleck_factor[cell_number] * sigma
            
            # Their difference
            sigma_diff = sigma - sigma_f 
            
            # Distance to boundary (a cell with new temperature dependent sigma)
            if mu > 0:
                distance_to_boundary = (internal_cell_position[cell_number + 1] - starting_x_position) / mu
            else: 
                distance_to_boundary = (internal_cell_position[cell_number] - starting_x_position) / mu
            
            # How far till crash into stuff (distance to collision)
            distance_to_collision = abs(math.log(random.random())) / sigma_diff
            
            # Distance to end of timestep (distance to census)
            distance_to_census = c * (end_of_timestep - starting_time)
            
            # Choosing which distance to assign and follow
            distance = min(distance_to_boundary,distance_to_collision,distance_to_census)
            
            # Energy difference and depositided energy 
            new_energy = energy * mp.exp(-sigma_f * distance) # --> not always < energy, part of 
            # the issue, new energy is larger than starting energy
            if new_energy <= 0.01 * starting_energy: # --> optimizing 
                new_energy = 0 
            
            # Difference
            energy_loss = energy - new_energy
                
            deposited_energy[cell_number] += energy_loss # Have tried += and -= in efforts to fix
            # energy deposition issues, makes no difference
            
            if new_energy == 0:
                properties[particle_number][6] = -1 # kill later
                break
        
            # If distance to boundary event; change properties (cell number), or mark for later killing
            if distance == distance_to_boundary:
                if mu > 0:
                    if cell_number == total_cells - 1:
                        properties[particle_number][6] = -1
                        break
                    cell_number += 1
                if mu < 0:
                    if cell_number == 0:
                        properties[particle_number][6] = -1
                        break
                    cell_number -= 1
            
            # Advance temporal and spatial potisions 
            ## IF STUFF DOESNT WORK CHECK THE ORDER OF THIS VERSES OTHER CASES
            starting_x_position = starting_x_position + distance * mu
            starting_time = starting_time + distance * (c ** -1)
            energy = new_energy
                   
            # If distance to census, put particle in census and update properties (not starting energy)
            if distance == distance_to_census:
                N_census_particles += 1
                properties[particle_number] = [starting_time,cell_number,starting_x_position,mu,frequency,energy,starting_energy]
                break
            
            # If distance to collision, recalculate mu and frequency (absorption case treaded as pseudo scattering)
            if distance == distance_to_collision:
                mu = random.uniform(-1,1)
                frequency = cell_temp[cell_number] * abs(math.log(random.random())) / h
    
    # kill off particles with no energy or outside bounds before next time step
    ##for particle_properties in properties:    # Not working
    ##    if particle_properties[6] < 0: 
    ##        del particle_properties
    for particle_number in range(len(properties)-1,-1,-1):
        if properties[particle_number][6] < 0:
            del properties[particle_number]
    
    # Tallies deposited energy and calculate new cell temperature (Eq 4.28 / 94b)
    radiation_energy_density[:] = radiation_constant_a * cell_temp[:] ** 4
    
    energy_increase[:] = deposited_energy[:] / dx - fleck_factor[:] * c * sigma_p[:] * radiation_energy_density[:] * dt
    
    cell_temp[:] = cell_temp[:] + energy_increase[:] / b
    
    current_time = current_time + dt
    #Christopher - start with Eq4.28 and 3.8 to see why why energy increase is negative

# Plotting stuffs --> keep getting nan / -inf so a plot will show (if not getting divide by zero
# runtime issues) but is blank (cant show neg infinity energy on a plot)
#plt.subplot(1,2,1)
#plt.plot(np.linspace(0,4,10),cell_energy[:])
# Rename, this is just temperary until its working
#plt.title("Energy vs position of slab")
#plt.xlabel("Position ")
#plt.ylabel("Energy")

#plt.subplot(1,2,2)
plt.plot(np.linspace(0,4,10),cell_temp[:])
plt.title("Temperature vs Position in 4 cm Slab (1000 Histories Over --- s)")
plt.xlabel("Position [cm]")
plt.ylabel("Temperature [KeV]")

plt.show() 

