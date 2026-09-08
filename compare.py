import pandas as pd
import tomllib
import pathlib as pl
from pathlib import Path
from argparse import ArgumentParser
from collections import namedtuple
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

FieldMapping = namedtuple("FieldMapping", ["label", "data_folder", "accessor", "type", "obs"], defaults=[None])

field_mappings = [
    FieldMapping("Data Fields",                         "plots",  "data_fields.png",                  "image"),
    FieldMapping("Absolute Residuals",                  "plots",  "equation_residuals_absolute.png",  "image"),
    FieldMapping("Relative Residuals",                  "plots",  "equation_residuals.png",           "image"),
    FieldMapping("Equation Terms",                      "plots",  "equation_terms.png",               "image"),
    FieldMapping("Predicted Thickness",                 "",       "h",                                "plot"),
    FieldMapping("Regridded Thickness",                 "",       "regrid_h",                         "plot"),
    FieldMapping("Thickness (Predicted - Regridded)",   "",       "h",                                "plot", "regrid_h"),
    FieldMapping("Predicted X Velocity",                "",       "u",                                "plot"),
    FieldMapping("Regridded X Velocity",                "",       "regrid_u",                         "plot"),
    FieldMapping("X Velocity (Predicted - Regridded)",  "",       "u",                                "plot", "regrid_u"),
    FieldMapping("Predicted Y Velocity",                "",       "v",                                "plot"),
    FieldMapping("Regridded Y Velocity",                "",       "regrid_v",                         "plot"),
    FieldMapping("Y Velocity (Predicted - Regridded)",  "",       "v",                                "plot", "regrid_v"),
    FieldMapping("Predicted Speed",                     "",       "speed",                            "plot"),
    FieldMapping("Regridded Speed",                     "",       "regrid_speed",                     "plot"),
    FieldMapping("Speed (Predicted - Regridded)",       "",       "speed",                            "plot", "regrid_speed"),
    FieldMapping("Viscosity",                           "",       "mu",                               "plot"),
    FieldMapping("Log Viscosity",                       "",       "mu",                               "plot"),
    FieldMapping("Effective Strain Rate",               "",       "e_eff",                            "plot"),
    FieldMapping("Strain Rate XX Component",            "",       "e_xx",                             "plot"),
    FieldMapping("Strain Rate XY Component",            "",       "e_xy",                             "plot"),
    FieldMapping("Strain Rate YY Component",            "",       "e_yy",                             "plot"),
    FieldMapping("Effective Stress",                    "",       "tau_eff",                          "plot"),
    FieldMapping("Stress XX Component",                 "",       "tau_xx",                           "plot"),
    FieldMapping("Stress XY Component",                 "",       "tau_xy",                           "plot"),
    FieldMapping("Stress YY Component",                 "",       "tau_yy",                           "plot"),
    FieldMapping("Stress Primary Component",            "",       "tau_1",                            "plot"),
    FieldMapping("Stress Secondary Component",          "",       "tau_2",                            "plot"),
    FieldMapping("Extensional Stress (R_xx)",           "",       "R_xx",                             "plot"),
    FieldMapping("Mean Residuals",                      "",       "residuals/mean",                   "markdown"),
    FieldMapping("Median Residuals",                    "",       "residuals/median",                 "markdown"),
]

_field_lookup = {fm.label: fm for fm in field_mappings}
LOG_FIELDS = {"Log Viscosity"}

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
data_fields = [fm.label for fm in field_mappings]

all_inversions = dataframe["Inversion Name"].to_list()


def handle_residuals(name1, name2, field):
    mapping = _field_lookup[field].accessor
    row1 = dataframe.loc[dataframe["Inversion Name"] == name1]
    row2 = dataframe.loc[dataframe["Inversion Name"] == name2]
    if "mean" in mapping:
        col_x, col_y, type_str = "Mean-Abs X Residual", "Mean-Abs Y Residual", "Mean"
    elif "median" in mapping:
        col_x, col_y, type_str = "Median-Abs X Residual", "Median-Abs Y Residual", "Median"
    else:
        return ""

    x1r, y1r = float(row1[col_x].iloc[0]), float(row1[col_y].iloc[0])
    x2r, y2r = float(row2[col_x].iloc[0]), float(row2[col_y].iloc[0])
    sum1, sqsum1 = x1r + y1r, (x1r**2 + y1r**2)**(1/2)
    sum2, sqsum2 = x2r + y2r, (x2r**2 + y2r**2)**(1/2)

    residual_string = f"""
# {type_str} Residuals:
|Inversion|X Residual|Y Residual|Sum|Squared Sum|
|---:|:---:|:---:|:---:|:---:|
|{name1}|{x1r}|{y1r}|{sum1}|{sqsum1}|
|{name2}|{x2r}|{y2r}|{sum2}|{sqsum2}|
|Difference|{x1r - x2r}|{y1r - y2r}|{sum1 - sum2}|{sqsum1 - sqsum2}|
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


def _read(zarr, key):
    if key == "speed":
        return (zarr["u"]**2 + zarr["v"]**2)**0.5
    if key == "regrid_speed":
        return (zarr["regrid_u"]**2 + zarr["regrid_v"]**2)**0.5
    return zarr[key]


def _field_da(name, fm):
    zarr = xr.open_zarr(folder_path / name / "evaluations.zarr", consolidated=False)
    da = _read(zarr, fm.accessor)
    if fm.obs is not None:                          # model − regridded observation
        da = da - _read(zarr, fm.obs)
    return np.log(da) if fm.label in LOG_FIELDS else da


def get_paired_plot(name1, name2, fields, pad=0.05):
    figs = []
    for field in fields:
        fm = _field_lookup[field]

        if fm.type == "plot":
            field1 = _field_da(name1, fm)
            field2 = _field_da(name2, fm)

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
            ax1.set_title(f"{name1} \n {field}")
            ax2.set_title(f"{name2} \n {field}")
            fig.suptitle(field)
            figs.append((field, fig))
            plt.close(fig)

        elif fm.type == "markdown":
            figs.append((field, handle_residuals(name1, name2, field)))

        elif fm.type == "image":
            path1 = (folder_path / name1 / fm.data_folder / fm.accessor).as_posix()
            path2 = (folder_path / name2 / fm.data_folder / fm.accessor).as_posix()
            figs.append((field, (path1, path2)))

    return figs


def get_diff_plot(name1, name2, fields, pad=0.05):
    figs = []
    for field in fields:
        fm = _field_lookup[field]

        if fm.type == "plot":
            diff = _field_da(name1, fm) - _field_da(name2, fm)

            xv, yv = _valid_bbox(diff)
            dmax, xv, yv = dask.compute(abs(diff).max(), xv, yv)
            vmax = float(dmax)

            b, xlim, ylim = _bounds_from_masks(diff, xv.values, yv.values, pad)
            crop = _crop(diff, *b)

            fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)
            crop.plot.imshow(ax=ax, cmap="seismic", vmin=-vmax, vmax=vmax)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_title(f"{field}:\n {name1} − {name2}")
            figs.append((field, fig))
            plt.close(fig)

        elif fm.type == "markdown":
            figs.append((field, handle_residuals(name1, name2, field)))

        elif fm.type == "image":
            path1 = (folder_path / name1 / fm.data_folder / fm.accessor).as_posix()
            path2 = (folder_path / name2 / fm.data_folder / fm.accessor).as_posix()
            figs.append((field, (path1, path2)))

    return figs


with gr.Blocks(fill_height=True, fill_width=True) as demo:

    with gr.Row():
        with gr.Column():
            inversion1 = gr.Dropdown(all_inversions, value=None, label="Inversion 1")
        with gr.Column():
            inversion2 = gr.Dropdown(all_inversions, value=None, label="Inversion 2")
    selected_data_fields = gr.CheckboxGroup(choices=data_fields, value=data_fields, label="Data Fields")

    def update_data_field_names(fields):
        field_string = "; ".join(fields)
        return (f"Show Fields Side by Side: {field_string}",
                f"Plot Fields: {field_string}",
                f"Plot the Difference for Fields: {field_string}")

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

            @gr.render(inputs=[inversion1, inversion2, selected_data_fields], triggers=[side_by_side.click])
            def show_side_by_side(name1: str, name2: str, fields: list, progress=gr.Progress()):
                if name1 is None or name2 is None:
                    gr.Error("Select Inversions")
                    return None
                figs = get_paired_plot(name1, name2, fields)
                gr.Markdown(f"# Comparing: {name1}, {name2}: ")
                with gr.Tabs():
                    for field, fig in progress.tqdm(figs):
                        with gr.Tab(field):
                            with gr.Row():
                                if isinstance(fig, str):
                                    gr.Markdown(fig, label="Residuals")
                                elif isinstance(fig, tuple):
                                    path1, path2 = fig
                                    with Image.open(path1) as img1, Image.open(path2) as img2:
                                        gr.Image(img1, label=f"{name1}/{field}")
                                        gr.Image(img2, label=f"{name2}/{field}")
                                else:
                                    gr.Plot(fig, label=f"{name1}/{field} vs. {name2}/{field}")

        with gr.Tab("Comparison Plot"):
            comparison_plot = gr.Button("Plot All Data Fields")

            @gr.render(inputs=[inversion1, inversion2, selected_data_fields], triggers=[comparison_plot.click])
            def plot_comparisons(name1: str, name2: str, fields: list, progress=gr.Progress()):
                if name1 is None or name2 is None:
                    gr.Error("Select Inversions")
                    return None
                figs = get_paired_plot(name1, name2, fields)
                gr.Markdown(f"# Comparing: {name1} and {name2}")
                with gr.Tabs():
                    for field, fig in progress.tqdm(figs):
                        with gr.Tab(field):
                            with gr.Column():
                                if isinstance(fig, str):
                                    gr.Markdown(fig, label="Residuals")
                                elif isinstance(fig, tuple):
                                    with gr.Row():
                                        path1, path2 = fig
                                        with Image.open(path1) as img1, Image.open(path2) as img2:
                                            gr.Image(img1, label=f"{name1}/{field}")
                                            gr.Image(img2, label=f"{name2}/{field}")
                                else:
                                    with gr.Row():
                                        gr.Plot(fig, label=f"{name1}/{field} vs. {name2}/{field}")

        with gr.Tab("Difference Plot"):
            difference_plot = gr.Button("Plot the Difference for All Data Fields")

            @gr.render(inputs=[inversion1, inversion2, selected_data_fields], triggers=[difference_plot.click])
            def plot_differences(name1: str, name2: str, fields: list, progress=gr.Progress()):
                if name1 is None or name2 is None:
                    gr.Error("Select Inversions")
                    return None
                figs = get_diff_plot(name1, name2, fields)
                gr.Markdown(f"# Comparing: {name1} and {name2}")
                with gr.Tabs():
                    for field, fig in progress.tqdm(figs):
                        with gr.Tab(field):
                            with gr.Column():
                                if isinstance(fig, str):
                                    gr.Markdown(fig, label="Residuals")
                                elif isinstance(fig, tuple):
                                    with gr.Row():
                                        path1, path2 = fig
                                        with Image.open(path1) as img1, Image.open(path2) as img2:
                                            gr.Image(img1, label=f"{name1}/{field}")
                                            gr.Image(img2, label=f"{name2}/{field}")
                                else:
                                    with gr.Row():
                                        gr.Plot(fig, label=f"{name1}/{field} minus {name2}/{field}")

    selected_data_fields.input(update_data_field_names, inputs=selected_data_fields, outputs=[side_by_side, comparison_plot, difference_plot])

demo.launch(share=True, allowed_paths=[folder_path.as_posix()], theme=gr.Theme.from_hub("Nymbo/Nymbo_Theme"))