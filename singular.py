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
parser.add_argument("--folder", required=True)
args = parser.parse_args()
folder_path = Path(args.folder).resolve().absolute()
subfolders = sorted(list(folder_path.glob("*/")))


def mean_abs(x):
    return np.mean(np.abs(x))


def median_abs(x):
    return np.median(np.abs(x))


parse = tomllib.loads(Path("./mapping.toml").read_text())

display_table = []

field_mappings = {
    "Data Fields": "data_fields.png",
    "Absolute Residuals": "equation_residuals_absolute.png",
    "Relative Residuals": "equation_residuals.png",
    "Equation Terms": "equation_terms.png",
    "Viscosity": "fields.png",
    "Effective Strain": "stresses/e_eff.png",
    "Extensional Stress (R_xx)": "stresses/R_xx.png",
    "Tau Effective": "stresses/tau_eff.png",
    "Tau 1": "stresses/tau_1.png",
    "Tau 2": "stresses/tau_2.png",
    "Mean Resid.": "residuals/mean",
    "Median Resid.": "residuals/median"
}

for count, subfolder in tqdm(list(enumerate(subfolders))):
    resid_file = subfolder / "field_and_residual_data.npz"
    if not resid_file.exists():
        continue
    data = np.load(resid_file)
    inversion_name = subfolder.name
    shelf, raw_velocity, raw_thickness = inversion_name.split("__")
    velocity_product, velocity_start, velocity_end = parse[raw_velocity]
    thickness_product, thickness_start, thickness_end = parse[raw_thickness]
    res_x, res_y = data["residual_x"], data["residual_y"]
    mean_res_x, mean_res_y = mean_abs(res_x), mean_abs(res_y)
    med_res_x, med_res_y = median_abs(res_x), median_abs(res_y)
    mean_res_x, mean_res_y, med_res_x, med_res_y = round(mean_res_x, 2), round(mean_res_y, 2), round(med_res_x, 2), round(med_res_y, 2)
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
                          med_res_x,
                          med_res_y
                          ])


headers = ["Inversion Name",
           "Shelf",
           "Velocity Product",
           "Velocity Start",
           "Velocity End",
           "Thickness Product",
           "Thickness Start",
           "Thickness End",
           "Mean-Abs X Residual",
           "Mean-Abs Y Residual",
           "Median-Abs X Residual",
           "Median-Abs Y Residual",
           ]


show_headers = ["Inversion Name",
                "Shelf",
                "Velocity End",
                "Mean-Abs X Residual",
                "Mean-Abs Y Residual",
                "Median-Abs X Residual",
                "Median-Abs Y Residual",
                ]

dataframe = pd.DataFrame(display_table, columns=headers)
data_fields = list(field_mappings.keys())

all_inversions = dataframe["Inversion Name"].to_list()


def handle_residuals(name, field):
    mapping = field_mappings[field]
    row = dataframe.loc[dataframe["Inversion Name"] == name]
    if "mean" in mapping:
        xr_ = float(row["Mean-Abs X Residual"].iloc[0])
        yr_ = float(row["Mean-Abs Y Residual"].iloc[0])
        type_str = "Mean"
    elif "median" in mapping:
        xr_ = float(row["Median-Abs X Residual"].iloc[0])
        yr_ = float(row["Median-Abs Y Residual"].iloc[0])
        type_str = "Median"
    else:
        return ""

    total = xr_ + yr_
    sqsum = (xr_**2 + yr_**2)**(1/2)

    residual_string = f"""
# {type_str} Residuals:
|Inversion|X Residual|Y Residual|Sum|Squared Sum|
|---:|:---:|:---:|:---:|:---:|
|{name}|{xr_}|{yr_}|{total}|{sqsum}|
"""
    return residual_string


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


def get_single_plot(name, fields, pad=0.05):
    zarr_path = folder_path / name / "stress.zarr"
    zarr = xr.open_zarr(zarr_path, consolidated=False)
    figs = []
    for field in fields:
        mapping = field_mappings[field]
        if "stresses" in mapping:
            field_key = mapping.replace("stresses/", "").split(".")[0]
            field_da = zarr[field_key]

            xv, yv = _valid_bbox(field_da)
            vmin, vmax, xv, yv = dask.compute(
                field_da.min(), field_da.max(), xv, yv,
            )
            vmin = float(vmin)
            vmax = float(vmax)

            b, xlim, ylim = _bounds_from_masks(field_da, xv.values, yv.values, pad)
            crop = _crop(field_da, *b)

            fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)
            crop.plot.imshow(ax=ax, vmin=vmin, vmax=vmax)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(f"{name} — {field}")
            fig.suptitle(field)
            figs.append((field, fig))
            plt.close(fig)
        elif "residuals/" in mapping:
            figs.append((field, handle_residuals(name, field)))
        else:
            path = (folder_path / name / "plots" / mapping).as_posix()
            figs.append((field, path))
    return figs


def step_inversion(current, direction):
    if not all_inversions:
        return None
    if current is None or current not in all_inversions:
        # start at the beginning (or end when stepping back)
        return all_inversions[0] if direction > 0 else all_inversions[-1]
    idx = all_inversions.index(current)
    new_idx = (idx + direction) % len(all_inversions)
    return all_inversions[new_idx]


with gr.Blocks(fill_height=True, fill_width=True) as demo:

    with gr.Row():
        inversion = gr.Dropdown(all_inversions, value=None, label="Inversion", scale=2)
        with gr.Column(scale=1):
            with gr.Row():
                prev_button = gr.Button("← Previous")
                next_button = gr.Button("Next →")
            plot_button = gr.Button("Plot Selected Data")

    prev_button.click(lambda cur: step_inversion(cur, -1), inputs=inversion, outputs=inversion)
    next_button.click(lambda cur: step_inversion(cur, +1), inputs=inversion, outputs=inversion)

    selected_data_fields = gr.CheckboxGroup(choices=data_fields, value=data_fields, label="Data Fields")

    with gr.Accordion("Table", open=False):
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

    @gr.render(inputs=[inversion, selected_data_fields], triggers=[plot_button.click])
    def plot_single(name: str, fields: list, progress=gr.Progress()):
        if name is None:
            gr.Error("Select an Inversion")
            return None
        figs = get_single_plot(name, fields)
        gr.Markdown(f"# {name}")
        with gr.Tabs():
            for field, fig in progress.tqdm(figs):
                with gr.Tab(field):
                    with gr.Column():
                        if isinstance(fig, str) and fig.startswith("\n#"):
                            gr.Markdown(fig, label="Residuals")
                        elif isinstance(fig, str):
                            with Image.open(fig) as img:
                                gr.Image(img, label=f"{name}/{field}")
                        else:
                            gr.Plot(fig, label=f"{name}/{field}")

demo.launch(share=True, allowed_paths=[folder_path.as_posix()], theme=gr.Theme.from_hub("Nymbo/Nymbo_Theme"))