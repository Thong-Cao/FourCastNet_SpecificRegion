import h5py
import numpy as np
import pandas as pd

def read_h5_file(file_path, csv_file_path, year):
    try:
        # Open the HDF5 file in read mode
        with h5py.File(file_path, 'r') as h5_file:
            # Accessing the middle data as you mentioned
            target_data = h5_file['fields'][:]

            # Read data from CSV file
            new_values = pd.read_excel(csv_file_path, header=None)

            # Check if the length of new_values matches the number of time steps (8760)
            if new_values.shape[0] == target_data.shape[0]:
                # Replace the middle slice with new_values
                middle_point = 22 # (full size -1)/2
                target_data[:, :, middle_point:middle_point+1, middle_point:middle_point+1] = new_values.values[:, :, np.newaxis, np.newaxis]
                print("Replacement successful.")
            else:
                print("Error: The length of the new data does not match the number of time steps (8760).")

            # Create a new HDF5 file to store the processed data
            with h5py.File(year + '.h5', 'w') as processed_file:
                # Create a new dataset named 'fields' and write the modified data to it
                processed_file.create_dataset('fields', data=target_data)

        print("Data written successfully." + year )

    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    # Provide the path to your HDF5 file and Excel file here
    lst = ["2020", "2021", "2022"]
    for year in lst:

        hdf5_file_path = r"/home/mcsp/Downloads/FourCastNet/data/model15/train/" + year + ".h5"
        csv_file_path = r"/home/mcsp/Downloads/FourCastNet/A2M/galmae-yeo/"+ year + ".xlsx"
        read_h5_file(hdf5_file_path, csv_file_path, year)
