import S4
import numpy as np
import os
from data_utils import save_values_csv, load_and_plot_csv_view, plot_geometry, plot_difference_csv
from itertools import product
from tqdm import tqdm
import csv
#path to save plots
save_path = '../plots'

#Simulation parameters
wavelength_range = np.linspace(3,5,40) # wavelength from 3 to 5 microns
unit_size = 1 # 1 micron
pillar_height = 8 * unit_size #8 microns
silicon_height = 10 * unit_size
header = ['height', 'radii list', 'wavelength', 'transmission', 'reflection'] 
def append_simulation_results(path, height, radii_list, wavelength, transmission, reflection,header):
    file_exists = os.path.isfile(path)
    
    with open(path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'height':height, 'radii list':radii_list, 'wavelength':wavelength, 
                        'transmission':transmission, 'reflection':reflection})
    
#define varying radii and spaceing dynamically
def run_simulation(radii_list,pillar_height,substrate_height=0,flip_stacking=False, alt_calc=False):
    max_r = max(radii_list)
    min_r = min(radii_list)
    #pillar_spacing = spacing_ratio * (2.7) #consistent spacing across cells
    lattice_size = 5.4 #Ensuring unirform spacing in periodic grid

    #create s4 simulation object
    S = S4.New(Lattice=((lattice_size, 0), (0, lattice_size)), NumBasis = 100)
    
    #define materials
    S.SetMaterial('Vacuum', 1)
    S.SetMaterial('Plastic', 1.526+ 0j)
    S.SetMaterial('Silicon', Epsilon=11.7)
    
    #set up the layer stack
    if flip_stacking is False:
        S.AddLayer('Air', 0, 'Vacuum')
        S.AddLayer('Pillars', pillar_height, 'Vacuum')
        S.AddLayer('slab', 1, 'Silicon')
    else:
        S.AddLayer('bottom',0,'Vacuum')
        S.AddLayer('slab', substrate_height, 'Silicon')
        S.AddLayer('Pillars', pillar_height, 'Vacuum')
        S.AddLayerCopy('Air', 0, 'bottom')
    
    #Define 2x2 pillar grid with uniform spacing
    pillar_positions = [
        [lattice_size * 1/4, lattice_size * 1/4],
        [lattice_size * 1/4, lattice_size * 3/4],
        [lattice_size * 3/4, lattice_size * 1/4],
        [lattice_size * 3/4, lattice_size * 3/4],
    ]

    #add pillars
    for i, r in enumerate(radii_list):
        S.SetRegionCircle(Layer='Pillars', Material='Plastic', 
                        Center=(pillar_positions[i][0], pillar_positions[i][1]),
                        Radius = r)
    
    # initialize results storage
    transmission_values, reflection_values, cylinder_values, frequencies = [], [], [], []
    
    # set excitation plane wave
    S.SetExcitationPlanewave((0, 0), 1 + 0j, 0 + 0j)
    
    # Solve for transmission/reflection at each wavelength
    for wl in wavelength_range:
        freq = 1/ wl #convert wavelength to frequency
        S.SetFrequency(freq)
        frequencies.append(freq)
        if alt_calc is False:
            inc, ref = S.GetPowerFlux('Air', 0)
            fw, _ = S.GetPowerFlux('slab', 0)
            cyl_fw2, cyl_bw2 = S.GetPowerFlux('Pillars', pillar_height)
            cyl_fw1, cyl_bw1 = S.GetPowerFlux('Pillars', 0)

            #calculate absorption
            cyl_abs = np.abs((cyl_fw2 - cyl_fw1 - (cyl_bw1 - cyl_bw2)) / inc)
            reflectance = np.abs(-ref / inc)
            transmission = np.abs(fw / inc)
        else:
            inc, ref = S.GetPowerFlux('bottom', 0)
            fw, _ = S.GetPowerFlux('Air', 0)
            
            reflectance = np.abs(-ref/ inc)
            transmission = np.abs(fw / inc)
            cyl_fw2, cyl_bw2 = S.GetPowerFlux('Pillars', pillar_height)
            cyl_fw1, cyl_bw1 = S.GetPowerFlux('Pillars', 0)

            #calculate absorption
            cyl_abs = np.abs((cyl_fw2 - cyl_fw1 - (cyl_bw1 - cyl_bw2)) / inc)
            
        file_path = '../plots/grid_data/grid_test1.csv' 
        append_simulation_results(file_path, pillar_height, [r1,r2,r3,r4], wl,
                                    transmission, reflectance, header)
        #store results
        transmission_values.append(transmission)
        reflection_values.append(reflectance)
        cylinder_values.append(cyl_abs)
            
    # save and plot results
    #plot_geometry(S, lattice_size,pillar_height,substrate_height,z=False)
    #plot_geometry(S, lattice_size,pillar_height,substrate_height,z=True)
    save_values_csv(frequencies, transmission_values, reflection_values, cylinder_values, [], radii_list, save_path, pillar_height)
    load_and_plot_csv_view(save_path, radii_list,pillar_height)
    #plot_difference_csv(save_path, radii_list, pillar_height)

    return transmission_values, reflection_values, cylinder_values

if __name__ == "__main__":
    #example: run with different radii and spacing values
    #radii_sets = [
    #    np.linspace(0.2 * 2.7, 0.45 *2.7, 4),
    #    np.linspace(0.25 * 2.7, 0.5 * 2.7, 4)
    #]
    radii_min = 0.05 * 2.7  # Min radius
    radii_max = 0.45 * 2.7  # Max radius
    num_radii = 4
    np.random.seed(12)
    #radii_list = np.linspace(radii_min, radii_max, 4)

    radii_sets = [
        np.round(np.random.uniform(radii_min,radii_max,4),4),
        np.round(np.random.uniform(radii_min,radii_max,4),4),
        np.round(np.random.uniform(radii_min,radii_max,4),4)
        ]
    radiis = np.linspace(radii_max, radii_min, 10)
    heights = np.linspace(10,1,10)

    R1, R2, R3, R4, H = np.meshgrid(radiis, radiis, radiis, radiis, heights, indexing='ij')
    
    for r1, r2, r3, r4, h in tqdm(zip(R1.ravel(), R2.ravel(), R3.ravel(), R4.ravel(), H.ravel())):
        print(f"Running simulation with radii {r1, r2, r3, r4} and height {h}")
        run_simulation([r1,r2,r3,r4], h)
     
    #run_simulation(radii_list, 0.5, 0, flip_stacking=False, alt_calc=False)
    #run_simulation(radii_list, 0.5, 0.1, flip_stacking=False, alt_calc=True)
    #run_simulation(radii_list, 0.5, 10, flip_stacking=True, alt_calc =True)
    #run_simulation(radii_list, 0.5, 0.1, flip_stacking=True, alt_calc=True)
    
        
    

