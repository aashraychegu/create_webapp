import pandas as pd
import tomllib
import pathlib as pl
from pathlib import Path
from argparse import ArgumentParser
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import gradio as gr
from tqdm import tqdm
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import xarray as xr
# pyrefly: ignore [missing-import]
import dask
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")

parser = ArgumentParser() 
parser.add_argument("--folder", required= True)
args = parser.parse_args()
folder_path = Path(args.folder).resolve().absolute()
subfolders = sorted(list(folder_path.glob("*/")))

def mean_abs(x):
    return np.mean(np.abs(x))

def median_abs(x):
    return np.median(np.abs(x))

parse = tomllib.loads(Path("./mapping.toml").read_text())

display_table = []
data_field_table = []

field_mappings = {
    "Data Fields": "data_fields.png",
    "Absolute Residuals": "equation_residuals_absolute.png",
    "Relative Residuals": "equation_residuals.png",
    "Viscosity": "fields.png",
    "Effective Strain": "stresses/e_eff.png",
    "Extensional Stress (R_xx)": "stresses/R_xx.png",
    "Tau 1": "stresses/tau_1.png",
    "Tau 2": "stresses/tau_2.png",
}

for count, subfolder in tqdm(list(enumerate(subfolders))):
    resid_file = subfolder / "field_and_residual_data.npz"
    if not resid_file.exists(): continue
    data = np.load(resid_file)
    inversion_name = subfolder.name
    shelf, raw_velocity, raw_thickness = inversion_name.split("__") 
    velocity_product, velocity_start, velocity_end = parse[raw_velocity]
    thickness_product, thickness_start, thickness_end = parse[raw_thickness]
    res_x, res_y = data["residual_x"], data["residual_y"]
    mean_res_x, mean_res_y = mean_abs(res_x)  , mean_abs(res_y)
    med_res_x , med_res_y  = median_abs(res_x), median_abs(res_y)
    mean_res_x, mean_res_y, med_res_x , med_res_y = round(mean_res_x,2), round(mean_res_y,2), round(med_res_x,2), round(med_res_y,2)
    display_table.append([inversion_name, 
                          shelf,
                          velocity_product, 
                          velocity_start, 
                          velocity_end, 
                          thickness_product, 
                          thickness_start, 
                          thickness_end, 
                          mean_res_x, 
                          mean_res_y, 
                          med_res_x , 
                          med_res_y
                        ])
    
    plot_dir = subfolder / "plots"
    if plot_dir.exists():
        data_field_mappings = {"Inversion Name" : inversion_name,}
        for field, path in field_mappings.items():
            field_path = plot_dir / path
            if field_path.exists():
                data_field_mappings[field] = field_path
        data_field_table.append(data_field_mappings)
    

headers = [ "Inversion\nName", 
            "Shelf", 
            "Velocity\nProduct",
            "Velocity\nStart", 
            "Velocity\nEnd", 
            "Thickness\nProduct", 
            "Thickness\nStart", 
            "Thickness\nEnd", 
            "Mean-Abs\nX Residual", 
            "Mean-Abs\nY Residual", 
            "Median-Abs\nX Residual", 
            "Median-Abs\nY Residual", 
        ]

show_headers = [ "Inversion\nName", 
                 "Shelf", 
                 "Velocity\nEnd", 
                 "Mean-Abs\nX Residual", 
                 "Mean-Abs\nY Residual", 
                 "Median-Abs\nX Residual", 
                 "Median-Abs\nY Residual", 
            ]

dataframe = pd.DataFrame(display_table, columns=headers)
data_field_frame = pd.DataFrame(data_field_table)
data_fields = data_field_frame.columns.to_list()
data_fields.remove("Inversion Name")

all_inversions = data_field_frame["Inversion Name"].to_list()

def _valid_bbox(da):
    valid = da.notnull()
    return valid.any("y"), valid.any("x")


def _bounds_from_masks(da, xvalid, yvalid, pad=0.05):
    xv = da["x"].values[xvalid]
    yv = da["y"].values[yvalid]
    xmin, xmax = xv.min(), xv.max()
    ymin, ymax = yv.min(), yv.max()

    dx = (xmax - xmin) * pad
    dy = (ymax - ymin) * pad
    xlim = (float(xmin) - dx, float(xmax) + dx)
    ylim = (float(ymin) - dy, float(ymax) + dy)
    return (xmin, xmax, ymin, ymax), xlim, ylim


def _crop(da, xmin, xmax, ymin, ymax):
    xasc = bool(da["x"][0] <= da["x"][-1])
    yasc = bool(da["y"][0] <= da["y"][-1])
    xsl = slice(xmin, xmax) if xasc else slice(xmax, xmin)
    ysl = slice(ymin, ymax) if yasc else slice(ymax, ymin)
    return da.sel(x=xsl, y=ysl)


def get_paired_plot(name1, name2, fields, pad=0.05):
    zarr1_path = folder_path / name1 / "stress.zarr"
    zarr2_path = folder_path / name2 / "stress.zarr"
    zarr1 = xr.open_zarr(zarr1_path, consolidated=False)
    zarr2 = xr.open_zarr(zarr2_path, consolidated=False)
    figs = []
    for field in fields:
        mapping = field_mappings[field]
        if "stresses" in mapping:
            raw_field_key = field_mappings[field]
            field_key = raw_field_key.replace("stresses/", "").split(".")[0]
            field1 = zarr1[field_key]
            field2 = zarr2[field_key]

            # Batch all reads into one pass
            x1v, y1v = _valid_bbox(field1)
            x2v, y2v = _valid_bbox(field2)
            vmin1, vmax1, vmin2, vmax2, x1v, y1v, x2v, y2v = dask.compute(
                field1.min(), field1.max(),
                field2.min(), field2.max(),
                x1v, y1v, x2v, y2v,
            )
            vmin = float(min(vmin1, vmin2))
            vmax = float(max(vmax1, vmax2))

            b1, xlim1, ylim1 = _bounds_from_masks(field1, x1v.values, y1v.values, pad)
            b2, xlim2, ylim2 = _bounds_from_masks(field2, x2v.values, y2v.values, pad)

            crop1 = _crop(field1, *b1)
            crop2 = _crop(field2, *b2)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), constrained_layout=True)

            crop1.plot.imshow(ax=ax1, vmin=vmin, vmax=vmax)
            crop2.plot.imshow(ax=ax2, vmin=vmin, vmax=vmax)

            ax1.set_xlim(*xlim1); ax1.set_ylim(*ylim1)
            ax2.set_xlim(*xlim2); ax2.set_ylim(*ylim2)

            ax1.set_title(f"{name1} — {field}")
            ax2.set_title(f"{name2} — {field}")

            fig.suptitle(field)
            figs.append((field, fig))
            plt.close(fig)
        else:
            path1 = (folder_path / name1 / "plots" / field_mappings[field]).as_posix()
            path2 = (folder_path / name2 / "plots" / field_mappings[field]).as_posix()
            figs.append((field,(path1, path2)))
    return figs


def get_diff_plot(name1, name2, fields, pad=0.05):
    zarr1_path = folder_path / name1 / "stress.zarr"
    zarr2_path = folder_path / name2 / "stress.zarr"
    zarr1 = xr.open_zarr(zarr1_path, consolidated=False)
    zarr2 = xr.open_zarr(zarr2_path, consolidated=False)

    figs = []
    for field in fields:
        mapping = field_mappings[field]
        if "stresses" in mapping:
            raw_field_key = field_mappings[field]
            field_key = raw_field_key.replace("stresses/", "").split(".")[0]
            field1 = zarr1[field_key]
            field2 = zarr2[field_key]
            diff = field1 - field2

            xv, yv = _valid_bbox(diff)
            dmax, xv, yv = dask.compute(abs(diff).max(), xv, yv)
            vmax = float(dmax)

            b, xlim, ylim = _bounds_from_masks(diff, xv.values, yv.values, pad)
            crop = _crop(diff, *b)

            fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)

            crop.plot.imshow(ax=ax, cmap="seismic", vmin=-vmax, vmax=vmax)

            ax.set_xlim(*xlim); ax.set_ylim(*ylim)

            ax.set_title(f"{field}: {name1} − {name2}")
            figs.append((field,fig))
            plt.close(fig)
        else:
            path1 = (folder_path / name1 / "plots" / field_mappings[field]).as_posix()
            path2 = (folder_path / name2 / "plots" / field_mappings[field]).as_posix()
            figs.append((field,(path1, path2)))
    return figs


with gr.Blocks(fill_height = True, fill_width = True) as demo:

    with gr.Row():
        with gr.Column():
            inversion1 = gr.Dropdown(all_inversions, value = None, label = "Inversion 1")
        with gr.Column():
            inversion2 = gr.Dropdown(all_inversions, value = None, label = "Inversion 2")
        with gr.Column():
            selected_data_fields = gr.CheckboxGroup(choices = data_fields,value=data_fields, label = "Data Fields")
            def update_data_field_names(fields):
                field_string = "; ".join(fields)
                return (f"Show Fields Side by Side: {field_string}",f"Plot Fields: {field_string}",f"Plot the Difference for Fields: {field_string}")

    with gr.Tabs():
        with gr.Tab("Table"):
            col_selector = gr.CheckboxGroup(
                choices=dataframe.columns.tolist(),
                value=show_headers,                    
                label="Columns to display",
            )
            df_view = gr.Dataframe(
                value=dataframe[show_headers],            
                headers=show_headers,
            )

            def filter_columns(selected):
                cols = selected or show_headers           
                sub = dataframe[cols]
                return gr.Dataframe(value=sub, headers=sub.columns.tolist())

            col_selector.change(filter_columns, inputs=col_selector, outputs=df_view)

        with gr.Tab("Side By Side"):
            side_by_side = gr.Button("Show All Data Fields Side by Side")

            @gr.render(inputs=[inversion1, inversion2, selected_data_fields], triggers = [side_by_side.click])
            def show_side_by_side(name1: str, name2: str, fields: list, progress=gr.Progress()):
                if name1 is None or name2 is None:
                    return None
                for field in progress.tqdm(fields):
                    gr.Markdown(f"### Comparing: {name1}, {name2} - {field}: ")
                    with gr.Row():
                        path1 = (folder_path / name1 / "plots" / field_mappings[field]).as_posix()
                        path2 = (folder_path / name2 / "plots" / field_mappings[field]).as_posix()
                        with Image.open(path1) as img1, Image.open(path2) as img2:
                            gr.Image(img1)
                            gr.Image(img2)

        with gr.Tab("Comparison Plot"):
            comparison_plot = gr.Button("Plot All Data Fields")
            
            @gr.render(inputs=[inversion1, inversion2, selected_data_fields],triggers = [comparison_plot.click])
            def plot_comparisons(name1: str, name2: str, fields: list, progress = gr.Progress()):
                if name1 is None or name2 is None:
                    return None
                figs = get_paired_plot(name1, name2, fields)
                for field,fig in progress.tqdm(figs):
                    with gr.Column():
                        gr.Markdown(f"### Comparing: {name1}, {name2} - {field}: ")
                        if isinstance(fig,tuple):
                            with gr.Row():
                                path1, path2 = fig
                                with Image.open(path1) as img1, Image.open(path2) as img2:
                                    gr.Image(img1)
                                    gr.Image(img2)
                        else:
                            with gr.Row():
                                gr.Plot(fig)                        

        with gr.Tab("Difference Plot"):
            difference_plot = gr.Button("Plot the Difference for All Data Fields")
            
            @gr.render(inputs=[inversion1, inversion2, selected_data_fields],triggers = [difference_plot.click])
            def plot_differences(name1: str, name2: str, fields: list, progress = gr.Progress()):
                if name1 is None or name2 is None:
                    return None
                figs = get_diff_plot(name1, name2, fields)
                for field,fig in progress.tqdm(figs):
                    with gr.Column():
                        gr.Markdown(f"### Comparing: {name1}, {name2} - {field}: ")
                        if isinstance(fig,tuple):
                            with gr.Row():
                                path1, path2 = fig
                                with Image.open(path1) as img1, Image.open(path2) as img2:
                                    gr.Image(img1)
                                    gr.Image(img2)
                        else:
                            with gr.Row():
                                gr.Plot(fig) 
                        
    selected_data_fields.input(update_data_field_names,inputs = selected_data_fields,outputs = [side_by_side, comparison_plot, difference_plot])

demo.launch(share=True, allowed_paths = [folder_path.as_posix()])

# Top interface: Select 1, Select 2
# 4 tabs: dataframe, side by side, comparison_plot, diff_plot

def process_pair(path1: Path, path2: Path): pass
