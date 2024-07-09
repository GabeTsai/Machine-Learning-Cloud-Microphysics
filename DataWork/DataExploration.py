import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.graph_objects as go

file_path = '/home/groups/yzwang/gabriel_files/PNNLData/ENA_Jul_18/00d-01h-00m-00s-000ms.h5'

def get_dataset(dataset_name):
    with h5py.File(file_path, 'r') as f:
        dataset = f[dataset_name]
        print(dataset_name)
        print(f"Shape: {dataset.shape}")

        for attr in dataset.attrs:
            print(f"Attribute: {attr} = {dataset.attrs[attr]}\n")
        
        return np.squeeze(dataset[:])

def volume_render_plot(data):
    x, y, z = np.indices(data.shape)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    values = data.flatten()

    fig = go.Figure(data=go.Volume(
        x=x,
        y=y,
        z=z,
        value=values,
        isomin=np.min(values),
        isomax=np.max(values),
        opacity=0.1,  # Adjust for better visualization
        surface_count=5,  # Number of isosurfaces
        colorscale='Viridis'  # Color scale
    ))

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    fig.show()

def main():
    qc_autoconv_data = get_dataset('qc_autoconv')
    qc_autoconv_data = -np.log(qc_autoconv_data, where = qc_autoconv_data > 0)
    volume_render_plot(qc_autoconv_data)


if __name__ == "__main__":
    main()