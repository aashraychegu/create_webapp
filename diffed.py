import pathlib as pl
from pathlib import Path
from argparse import ArgumentParser
from collections import namedtuple
import numpy as np
import gradio as gr
from tqdm import tqdm
from PIL import Image
import xarray as xr
import dask
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import tomllib


parser = ArgumentParser()
parser.add_argument("--folder", required=True)
args = parser.parse_args()
folder_path = Path(args.folder).resolve().absolute()
subfolders = sorted(list(folder_path.glob("*/")))

config_path = folder_path / "diff.toml"
name_map = tomllib.loads(config_path.read_text())["comparisons"]

FieldMapping = namedtuple("FieldMapping", ["label", "data_folder", "accessor", "type"])

field_mappings = [
    FieldMapping("Thickness Difference",                        "images", "Thickness Difference.png",                         "image"),
    FieldMapping("X Velocity Difference",                       "images", "X Velocity Difference.png",                        "image"),
    FieldMapping("Y Velocity Difference",                       "images", "Y Velocity Difference.png",                        "image"),
    FieldMapping("Speed Difference",                            "images", "Y Velocity Difference.png",                        "image"),
    FieldMapping("Log Effective Strain Rate and Log Viscosity", "images", "Log Effective Strain Rate and Log Viscosity.png",  "image"),
    FieldMapping("Viscosity",                                   "images", "Viscosity.png",                                    "image"),
    FieldMapping("Log Viscosity",                               "images", "Log Viscosity.png",                                "image"),
    FieldMapping("Effective Strain Rate",                       "images", "Effective Strain Rate.png",                        "image"),
    FieldMapping("Log Effective Strain Rate",                   "images", "Log Effective Strain Rate.png",                    "image"),
    FieldMapping("Strain Rate XX Component",                    "images", "Strain Rate XX Component.png",                     "image"),
    FieldMapping("Strain Rate XY Component",                    "images", "Strain Rate XY Component.png",                     "image"),
    FieldMapping("Strain YY Component",                         "images", "Strain YY Component.png",                          "image"),
    FieldMapping("Effective Stress",                            "images", "Effective Stress.png",                             "image"),
    FieldMapping("Stress XX Component",                         "images", "Stress XX Component.png",                          "image"),
    FieldMapping("Stress XY Component",                         "images", "Stress XY Component.png",                          "image"),
    FieldMapping("Stress YY Component",                         "images", "Stress YY Component.png",                          "image"),
    FieldMapping("Stress Primary Component",                    "images", "Stress Primary Component.png",                     "image"),
    FieldMapping("Stress Secondary Component",                  "images", "Stress Secondary Component.png",                   "image"),
    FieldMapping("Extensional Stress (R_xx)",                   "images", "Extensional Stress (R_xx).png",                    "image"),
    FieldMapping("Histogram of Stress",                         "images", "Histogram of Stress.png",                          "image"),
    FieldMapping("Effective Stress to Effective Strain",        "images", "Effective Stress to Effective Strain.png",         "image"),
]

_field_lookup = {fm.label: fm for fm in field_mappings}

all_inversions = [subfolder.name for subfolder in subfolders]

display_inversion_names = [
    f"{inversion_name} ( '{name_map[inversion_name]['minuend']}' - '{name_map[inversion_name]['subtrahend']}' )"
    for inversion_name in all_inversions
]

# Map display name -> actual inversion name (and back) for internal lookups
_display_to_name = dict(zip(display_inversion_names, all_inversions))
_name_to_display = dict(zip(all_inversions, display_inversion_names))

data_fields = [fm.label for fm in field_mappings]


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
    figs = []
    for field in fields:
        fm = _field_lookup[field]
        mapping = fm.accessor

        if fm.type == "plot":
            zarr_path = folder_path / name / "stress.zarr"
            zarr = xr.open_zarr(zarr_path, consolidated=False)
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

        elif fm.type == "image":
            path = (folder_path / name / fm.data_folder / mapping).as_posix()
            figs.append((field, path))

    return figs


def step_inversion(current, direction):
    if not display_inversion_names:
        return None
    if current is None or current not in display_inversion_names:
        # start at the beginning (or end when stepping back)
        return display_inversion_names[0] if direction > 0 else display_inversion_names[-1]
    idx = display_inversion_names.index(current)
    new_idx = (idx + direction) % len(display_inversion_names)
    return display_inversion_names[new_idx]


with gr.Blocks(fill_height=True, fill_width=True) as demo:

    with gr.Row():
        inversion = gr.Dropdown(display_inversion_names, value=None, label="Inversion", scale=2)
        with gr.Column(scale=1):
            with gr.Row():
                prev_button = gr.Button("← Previous")
                next_button = gr.Button("Next →")
            plot_button = gr.Button("Plot Selected Data")

    prev_button.click(lambda cur: step_inversion(cur, -1), inputs=inversion, outputs=inversion)
    next_button.click(lambda cur: step_inversion(cur, +1), inputs=inversion, outputs=inversion)

    selected_data_fields = gr.CheckboxGroup(choices=data_fields, value=data_fields, label="Data Fields")

    @gr.render(inputs=[inversion, selected_data_fields], triggers=[plot_button.click])
    def plot_single(display_name: str, fields: list, progress=gr.Progress()):
        if display_name is None:
            gr.Error("Select an Inversion")
            return None
        name = _display_to_name[display_name]
        figs = get_single_plot(name, fields)
        gr.Markdown(f"# {display_name}")
        with gr.Tabs():
            for field, fig in progress.tqdm(figs):
                with gr.Tab(field):
                    with gr.Column():
                        if isinstance(fig, str):
                            with Image.open(fig) as img:
                                gr.Image(img, label=f"{name}/{field}")
                        else:
                            gr.Plot(fig, label=f"{name}/{field}")

demo.launch(share=True, allowed_paths=[folder_path.as_posix()], theme=gr.Theme.from_hub("Nymbo/Nymbo_Theme"))