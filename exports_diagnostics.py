"""
exports_diagnostics.py

Builds the four diagnostic graphs (energy evolution, core averages,
confinement proxy, Lawson proxy) from SimState's history arrays and
bundles them into a downloadable zip of PNGs.

Each graph function takes plain arrays, not sim_state directly, so they
stay easy to test/reuse on their own. export_diagnostics(sim_state) is
the one function that actually touches sim_state - it pulls the right
history arrays off it, builds all four figures, zips them, and cleans up.
"""

import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np

from constants import fusion_threshold_proxy


def energy_graph(
    time_evolution,
    kinetic_energy_evolution,
    internal_energy_evolution,
    magnetic_energy_evolution,
):
    total_energy_evolution = (
        np.array(kinetic_energy_evolution)
        + np.array(internal_energy_evolution)
        + np.array(magnetic_energy_evolution)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(time_evolution, kinetic_energy_evolution, label="Kinetic")
    ax.plot(time_evolution, internal_energy_evolution, label="Internal")
    ax.plot(time_evolution, magnetic_energy_evolution, label="Magnetic")
    ax.plot(time_evolution, total_energy_evolution, label="Total", linewidth=2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Evolution")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return fig


def core_averages_graph(
    time_evolution,
    average_core_pressure_evolution,
    average_core_temperature_evolution,
    average_core_density_evolution,
):
    # NOTE: deliberately 3 stacked subplots rather than one shared axis
    # (spec left the layout as a creative choice). Pressure, temperature,
    # and density can diverge by very different factors during a violent
    # pinch collapse - on one shared axis, whichever quantity moves least
    # would be flattened to a near-invisible line next to the others.
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(time_evolution, average_core_pressure_evolution, color="tab:red")
    axes[0].set_ylabel("Core Pressure")
    axes[0].grid(True)

    axes[1].plot(time_evolution, average_core_temperature_evolution, color="tab:orange")
    axes[1].set_ylabel("Core Temperature")
    axes[1].grid(True)

    axes[2].plot(time_evolution, average_core_density_evolution, color="tab:blue")
    axes[2].set_ylabel("Core Density")
    axes[2].set_xlabel("Time")
    axes[2].grid(True)

    fig.suptitle("Core Plasma Averages")
    fig.tight_layout()
    return fig


def confinement_proxy_graph(
    time_evolution,
    confinement_proxy_evolution,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(time_evolution, confinement_proxy_evolution, label="Confinement Proxy")

    ax.set_xlabel("Time")
    ax.set_ylabel("Confinement Proxy")
    ax.set_title("Energy Confinement Proxy")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return fig


def lawson_proxy_graph(
    time_evolution,
    lawson_proxy_evolution,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(time_evolution, lawson_proxy_evolution, label="Lawson Proxy")

    ax.axhline(
        y=fusion_threshold_proxy,
        color="black",
        linestyle="--",
        label="Fusion Threshold Proxy",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Lawson Proxy")
    ax.set_title("Lawson Criterion Proxy")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return fig


def export_zip_file(
    graph_energy,
    graph_core_averages,
    graph_confinement_proxy,
    graph_lawson_proxy,
    output_path="diagnostics_export.zip",
):
    """
    Saves all four figures as PNGs into a single zip file at output_path.
    Spec mentioned "jpegs" but also "PNG probably goated" - going with
    PNG (lossless, matplotlib's native format, no quality-loss tradeoff
    to weigh for line graphs).
    """
    figures = {
        "energy.png": graph_energy,
        "core_averages.png": graph_core_averages,
        "confinement_proxy.png": graph_confinement_proxy,
        "lawson_proxy.png": graph_lawson_proxy,
    }

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, fig in figures.items():
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="png", dpi=300)
                buffer.seek(0)
                zf.writestr(filename, buffer.read())

    return output_path


def export_diagnostics(sim_state, output_path="diagnostics_export.zip"):
    """
    Top-level entry point: pulls the history arrays sim_state already
    maintains (see simstate.py / solvers.py), builds all four figures,
    zips them, and closes the figures afterward.

    NOTE: closing the figures explicitly here (plt.close) is new - the
    original reference never did this. Left open, matplotlib keeps every
    figure alive in memory; in an interactive app where someone can pause
    and export multiple times in one long session, that's a slow memory
    leak that was never going to show up in a single-shot test but would
    eventually matter here.

    NOTE: "export as a zip and a download starts" in the UI spec reads
    like it was written assuming a web app. requirements.txt says PyQt5,
    so this is a desktop app - there's no browser to trigger a download
    from. This function just writes the zip to output_path and returns
    that path; it's on ui_ux.py/main.py to show a native file-save dialog
    (e.g. QFileDialog) rather than anything download-like.
    """
    fig_energy = energy_graph(
        sim_state.time_array,
        sim_state.kinetic_energy_history,
        sim_state.internal_energy_history,
        sim_state.magnetic_energy_history,
    )
    fig_core = core_averages_graph(
        sim_state.time_array,
        sim_state.core_pressure_evolution,
        sim_state.core_temperature_evolution,
        sim_state.core_density_evolution,
    )
    fig_confinement = confinement_proxy_graph(
        sim_state.time_array,
        sim_state.confinement_evolution,
    )
    fig_lawson = lawson_proxy_graph(
        sim_state.time_array,
        sim_state.lawson_evolution,
    )

    path = export_zip_file(fig_energy, fig_core, fig_confinement, fig_lawson, output_path)

    for fig in (fig_energy, fig_core, fig_confinement, fig_lawson):
        plt.close(fig)

    return path