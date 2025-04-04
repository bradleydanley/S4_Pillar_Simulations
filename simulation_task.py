#!/usr/bin/env python3
import S4
import numpy as np
import os
import csv
import argparse
import sys

# Simulation parameters (40 wavelengths from 3 to 5 microns)
wavelength_range = np.linspace(3, 5, 40)
header = ['height', 'radii list', 'wavelength', 'transmission', 'reflection']

def run_simulation(radii_list, pillar_height, substrate_height=0, flip_stacking=False, alt_calc=False):
    # Set lattice dimensions and create the S4 simulation object
    lattice_size = 5.4  # uniform spacing in the periodic grid
    S = S4.New(Lattice=((lattice_size, 0), (0, lattice_size)), NumBasis=100)
    
    # Define materials
    S.SetMaterial('Vacuum', 1)
    S.SetMaterial('Plastic', 1.526+0j)
    S.SetMaterial('Silicon', Epsilon=11.7)
    
    # Build layer stack
    if not flip_stacking:
        S.AddLayer('Air', 0, 'Vacuum')
        S.AddLayer('Pillars', pillar_height, 'Vacuum')
        S.AddLayer('slab', 1, 'Silicon')
    else:
        S.AddLayer('bottom', 0, 'Vacuum')
        S.AddLayer('slab', substrate_height, 'Silicon')
        S.AddLayer('Pillars', pillar_height, 'Vacuum')
        S.AddLayerCopy('Air', 0, 'bottom')
    
    # Define a 2x2 pillar grid (using the provided radii_list)
    pillar_positions = [
        [lattice_size * 1/4, lattice_size * 1/4],
        [lattice_size * 1/4, lattice_size * 3/4],
        [lattice_size * 3/4, lattice_size * 1/4],
        [lattice_size * 3/4, lattice_size * 3/4],
    ]
    
    for i, r in enumerate(radii_list):
        S.SetRegionCircle(Layer='Pillars', Material='Plastic',
                          Center=(pillar_positions[i][0], pillar_positions[i][1]),
                          Radius=r)
    
    # Set the excitation plane wave
    S.SetExcitationPlanewave((0, 0), 1+0j, 0+0j)
    
    # Collect simulation results (one row per wavelength)
    results = []
    for wl in wavelength_range:
        freq = 1 / wl
        S.SetFrequency(freq)
        if not alt_calc:
            inc, ref = S.GetPowerFlux('Air', 0)
            fw, _ = S.GetPowerFlux('slab', 0)
            cyl_fw2, cyl_bw2 = S.GetPowerFlux('Pillars', pillar_height)
            cyl_fw1, cyl_bw1 = S.GetPowerFlux('Pillars', 0)
            # Calculate values
            reflectance = np.abs(-ref / inc)
            transmission = np.abs(fw / inc)
        else:
            inc, ref = S.GetPowerFlux('bottom', 0)
            fw, _ = S.GetPowerFlux('Air', 0)
            reflectance = np.abs(-ref / inc)
            transmission = np.abs(fw / inc)
            cyl_fw2, cyl_bw2 = S.GetPowerFlux('Pillars', pillar_height)
            cyl_fw1, cyl_bw1 = S.GetPowerFlux('Pillars', 0)
        
        results.append({
            'height': pillar_height,
            'radii list': radii_list,
            'wavelength': wl,
            'transmission': transmission,
            'reflection': reflectance
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run a single simulation task.')
    parser.add_argument('--r1', type=float, required=True, help='Radius 1')
    parser.add_argument('--r2', type=float, required=True, help='Radius 2')
    parser.add_argument('--r3', type=float, required=True, help='Radius 3')
    parser.add_argument('--r4', type=float, required=True, help='Radius 4')
    parser.add_argument('--height', type=float, required=True, help='Pillar height')
    parser.add_argument('--output', type=str, default='output_temp.csv',
                        help='Output CSV file for simulation results')
    args = parser.parse_args()
    
    radii_list = [args.r1, args.r2, args.r3, args.r4]
    
    # Log the start of the simulation to stderr
    print(f"Starting simulation with radii: {radii_list} and height: {args.height}", 
          flush=True, file=sys.stderr)
    
    results = run_simulation(radii_list, args.height)
    
    # Log the end of the simulation to stderr
    print(f"Finished simulation with radii: {radii_list} and height: {args.height}", 
          flush=True, file=sys.stderr)
    
    # Write results to the output CSV file
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    main()

