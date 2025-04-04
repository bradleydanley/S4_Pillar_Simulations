import S4
import numpy as np
import csv
import matplotlib.pyplot as plt
import S4Utils.S4Utils as S4Utils
import os
from tqdm import tqdm



def save_values_csv(frequencies, transmission_values, reflection_values,cylinder_values, silicon_values, radii_list, path, height): #put cylinder_values before silicon
    folder_name ="Radii_" + "_".join(map(str,radii_list)) + "_" + str(height)
    full_path = os.path.join(path, folder_name)

    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, "transmission_data.csv")
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frequency', 'Transmission', 'Reflection', 'Cylinder Absorption', 'Silicon Absorption'])
        print("reflection:", len(reflection_values))
        print("frequencies:", len(frequencies))
        print("transmission:", len(transmission_values))
        print("cylinder:", len(cylinder_values))
        print("silicon:", len(silicon_values))
        while len(silicon_values) < 50:
            silicon_values.append(0)
            
        for freq, trans, refl, cyl, si in zip(frequencies, transmission_values, 
                                            reflection_values, cylinder_values, 
                                            silicon_values): #put cyl and cylinder_values before silicon
            
            """
            if isinstance(trans, tuple):
                writer.writerow([freq] + list(trans))
            else:"""
            writer.writerow([freq, trans, refl, cyl, si]) #put cyl before si

def load_and_plot_csv_view(file_path, radii_list, pillar_height):
    frequencies =[]
    transmission_values = []
    reflection_values = []
    cylinder_values = []
    silicon_values = []
    
    folder_name = "Radii_" + "_".join(map(str, radii_list)) + "_" + str(pillar_height) + "/transmission_data.csv"
    
    file_path = os.path.join(file_path, folder_name)
    print('file path is:', file_path)

    
    with open(file_path, mode='r') as file:
        print('got to the open section')
        reader = csv.reader(file)
        next(reader)
    
        for row in reader:
            freq = float(row[0])
            frequencies.append(freq)
            """
            transmission_value = complex(row[1].replace('(', '').replace(')', ''))
            transmission_values.append(transmission_value)
            reflection_values.append(complex(row[2].replace('(','').replace(')', '')))
            """
            transmission_value = complex(row[1])
            transmission_values.append(transmission_value)
            reflection_values.append(complex(row[2]))
            cylinder_values.append(complex(row[3]))
            silicon_values.append(complex(row[4])) #make 4
            
            
    frequencies = np.array(frequencies)
    transmission_values = np.array(transmission_values)
    reflection_values = np.array(reflection_values)
    cylinder_values = np.array(cylinder_values)
    silicon_values = np.array(silicon_values)
   
    print(transmission_values.real) 
    plt.plot(1/frequencies, transmission_values.real, label='Transmission', color = 'b')# marker ='o', linestyle='None', markersize=2)
    plt.plot(1/frequencies, reflection_values.real, label='Reflection', color = 'y')#marker = 'o', linestyle='None', markersize=2)
    plt.plot(1/frequencies, cylinder_values.real, label = 'Absorption (Cylinder)', color = 'cyan')#marker ='o', linestyle = 'None', markersize=2)
    plt.plot(1/frequencies, silicon_values.real, label = 'Absorption (Silicon)', color = 'pink')# marker ='o', linestyle = 'None', markersize=2)
    
 
    plt.xlabel('Wavelength')
    plt.ylabel('Value')
        
    title = 'Transmission v Frequency for radii of (' + ', '.join(map(str, radii_list[0:4])) + ')'
    plt.title(title)

    plt.legend(loc="upper right")
    plt.ylim(0,3)
    plt.axhline(y=1,color='red', linestyle='--', linewidth=2)
    folder = os.path.dirname(file_path)
    plot_filename = os.path.join(folder, 'transmission_plot.png')
    plt.savefig(plot_filename)
    plt.show()
    plt.close()
def load_and_plot_csv(file_path, radii_list, pillar_height):
    frequencies =[]
    transmission_values = []
    reflection_values = []
    cylinder_values = []
    silicon_values = []
    
    folder_name = "Radii_" + "_".join(map(str, radii_list)) + "_" + str(pillar_height) + "/transmission_data.csv"
    file_path = os.path.join(file_path, folder_name)


    
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
    
        for row in reader:
            freq = float(row[0])
            frequencies.append(freq)
            """
            transmission_value = complex(row[1].replace('(', '').replace(')', ''))
            transmission_values.append(transmission_value)
            reflection_values.append(complex(row[2].replace('(','').replace(')', '')))
            """
            transmission_value = complex(row[1])
            transmission_values.append(transmission_value)
            reflection_values.append(complex(row[2]))
            cylinder_values.append(complex(row[3]))
            silicon_values.append(complex(row[4])) #make 4
            
            
    frequencies = np.array(frequencies)
    transmission_values = np.array(transmission_values)
    reflection_values = np.array(reflection_values)
    cylinder_values = np.array(cylinder_values)
    silicon_values = np.array(silicon_values)
    
    plt.figure()
    plt.plot(1/frequencies, transmission_values.real, label='Transmission', color = 'b')# marker ='o', linestyle='None', markersize=2)
    plt.plot(1/frequencies, reflection_values.real, label='Reflection', color = 'y')#marker = 'o', linestyle='None', markersize=2)
    plt.plot(1/frequencies, cylinder_values.real, label = 'Absorption (Cylinder)', color = 'cyan')#marker ='o', linestyle = 'None', markersize=2)
    plt.plot(1/frequencies, silicon_values.real, label = 'Absorption (Silicon)', color = 'pink')# marker ='o', linestyle = 'None', markersize=2)
    
 
    plt.xlabel('Wavelength')
    plt.ylabel('Value')
        
    title = 'Transmission v Frequency for radii of (' + ', '.join(map(str, radii_list[0:4])) + ')'
    plt.title(title)

    plt.legend(loc="upper right")
    plt.ylim(0,1)
    
    folder = os.path.dirname(file_path)
    plot_filename = os.path.join(folder, 'transmission_plot.png')
    plt.savefig(plot_filename)
    #plt.show()
    plt.close()


def plot_geometry(S,lattice_size,ph,sh, z=False):    
    #plotting the geometry
    if z is False:
        xmin = 0
        xmax = lattice_size
        ymin = 0
        ymax = lattice_size
        height = sh + ph/2
        step = 0.01
        x_vals = np.linspace(xmin, xmax, 500)
        y_vals = np.linspace(ymin, ymax, 500)
        z_vals = np.linspace(0, height, 500)
        epsilon_real = S4Utils.GetSlice(S, x_vals, y_vals, height, 'xy', 'Epsilon').real 
        
        #from IPython import embed; embed()
        #epsilon_real = np.zeros((len(x_vals), len(y_vals)))
        """ 
        for i, x in tqdm(enumerate(x_vals)):
            for j, y in enumerate(y_vals):
                epsilon = S.GetEpsilon(z=0.3).real
                epsilon_real[i,j] = epsilon.real
        
        epsilon_real = S.GetEpsilon(x_vals, y_vals, 0.3).real
        """
        plt.imshow(epsilon_real.T, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
        
        xx,yy = np.meshgrid(x_vals,y_vals)
        plt.pcolormesh(xx,yy, epsilon_real)
        plt.colorbar(label='Dielectric Constant')
        plt.title('Dielectric Constant Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        plt.close()
    else:
        xmin = 0
        xmax = lattice_size
        zmin = 0
        zmax = ph + sh
        depth = lattice_size*1/4
        x_vals = np.linspace(xmin, xmax, 500)
        z_vals = np.linspace(zmin, zmax, 500)
        epsilon_real = S4Utils.GetSlice(S, x_vals, z_vals, depth, 'xz', 'Epsilon').real
        
        plt.imshow(epsilon_real.T, extent=(xmin, xmax, zmin, zmax), origin='lower', cmap='viridis')
        xx, zz = np.meshgrid(x_vals, z_vals)
        plt.pcolormesh(xx,zz,epsilon_real)
        plt.colorbar(label='Dielectric Constant')
        plt.title('Dielectric Constant Distribution')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.show()
        plt.close() 
    
def plot_difference_csv(file_path, radii_list, pillar_height):
    frequencies = []
    transmission_values = []
    reflection_values = []

    folder_name = "Radii_" + "_".join(map(str, radii_list)) + "_" + str(pillar_height) + "/transmission_data.csv"
    full_file_path = os.path.join(file_path, folder_name)
    
    with open(full_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            freq = float(row[0])
            frequencies.append(freq)
            transmission_values.append(complex(row[1]))
            reflection_values.append(complex(row[2]))
    
    frequencies = np.array(frequencies)
    transmission_values = np.array(transmission_values)
    reflection_values = np.array(reflection_values)
    
    wavelengths = 1 / frequencies

    mask = (wavelengths >= 3) & (wavelengths <= 5)
    wavelengths_filtered = wavelengths[mask]
    transmission_filtered = transmission_values.real[mask]
    reflection_filtered = reflection_values.real[mask]

    base_transmission = 0.70017237125
    base_reflection = 0.2998276287
    
    transmission_diff = transmission_filtered - base_transmission
    reflection_diff = reflection_filtered - base_reflection
    
    
    plt.figure()
    plt.plot(wavelengths_filtered, transmission_diff, label='Transmission Difference', color = 'blue')
    plt.plot(wavelengths_filtered, reflection_diff, label='Reflection Difference', color = 'orange')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Difference Value')
    plt.title('Difference in Transmission and Reflection vs Wavelength\nRadii: ' + ', '.join(map(str, radii_list)))
    plt.legend(loc='best')
    plt.grid(True)
    
    folder = os.path.dirname(full_file_path)
    plot_filename = os.path.join(folder, 'difference_plot.png')
    plt.savefig(plot_filename)
    #plt.show()
    plt.close() 
