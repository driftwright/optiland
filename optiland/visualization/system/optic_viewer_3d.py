"""
OpticViewer3D: A class for visualizing optical systems in 3D.
This module provides the `OpticViewer3D` class, which extends the `BaseViewer3D` class
to visualize optical systems using VTK for 3D rendering. It allows for ray tracing and
system representation in a 3D space, enabling interactive exploration of
optical components and ray paths.

Kramer Harrison, 2024

re-worked by Manuel Fragata Mendes, june 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import vtk

from optiland.visualization.base import BaseViewer3D
from optiland.visualization.coloring.scale import collect_scales
from optiland.visualization.system.rays import Rays3D
from optiland.visualization.system.system import OpticalSystem

if TYPE_CHECKING:
    from pathlib import Path


class OpticViewer3D(BaseViewer3D):
    """A class used to visualize optical systems in 3D.

    Args:
        optic: The optical system to be visualized.

    Attributes:
        optic: The optical system to be visualized.
        rays: An instance of Rays3D for ray tracing.
        system: An instance of OpticalSystem for system representation.
        ren_win: The vtkRenderWindow object for visualization.
        iren: The vtkRenderWindowInteractor object for interaction.

    Methods:
        See :meth:`view` for visualizing the optical system in 3D.

    """

    def __init__(self, optic):
        self.optic = optic

        self.rays = Rays3D(optic)
        self.system = OpticalSystem(optic, self.rays, projection="3d")

        self.ren_win = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()

    @staticmethod
    def _add_scalar_bars(renderer, scales):
        """Add one scalar bar per scale along the right edge, side by side."""
        width, margin = 0.08, 0.02
        for slot, scale in enumerate(scales):
            bar = scale.scalar_bar_actor()
            bar.SetOrientationToVertical()
            bar.SetWidth(width)
            bar.SetHeight(0.7)
            bar.SetPosition(1.0 - (slot + 1) * (width + margin) - 0.03, 0.15)
            renderer.AddViewProp(bar)

    def view(
        self,
        fields="all",
        wavelengths="primary",
        num_rays=24,
        distribution="ring",
        figsize=(1200, 800),
        dark_mode=False,
        reference=None,
        hide_vignetted=False,
        *,
        component_colorizer=None,
        ray_colorizer=None,
        show_colorbar=True,
    ):
        """Visualizes the optical system in 3D.

        Args:
            fields (str, optional): The fields to be visualized.
                Defaults to 'all'.
            wavelengths (str, optional): The wavelengths to be visualized.
                Defaults to 'primary'.
            num_rays (int, optional): The number of rays to be visualized.
                Defaults to 24.
            distribution (str, optional): The distribution of rays.
                Defaults to 'ring'.
            figsize (tuple, optional): The size of the figure.
                Defaults to (1200, 800).
            dark_mode (bool, optional): Whether to use dark mode.
                Defaults to False.
            reference (str, optional): The reference rays to plot. Options
                include "chief" and "marginal". Defaults to None.
            hide_vignetted (bool, optional): If True, rays that vignette at any
                surface are not shown. Defaults to False.
            component_colorizer (ComponentColorizer, optional): Strategy that
                colors lens elements, mirrors and surfaces. Defaults to None,
                which keeps the theme colors.
            ray_colorizer (RayColorizer, optional): Strategy that colors rays.
                Defaults to None, which keeps the default ray colors.
            show_colorbar (bool, optional): If True (default), shows a colorbar
                for each colorizer that provides a scale.

        """
        renderer = self._make_renderer(dark_mode)
        self.ren_win.AddRenderer(renderer)

        self.iren.SetRenderWindow(self.ren_win)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        if ray_colorizer is not None:
            ray_colorizer.prepare(self.optic)
        self.rays.plot(
            renderer,
            fields=fields,
            wavelengths=wavelengths,
            num_rays=num_rays,
            distribution=distribution,
            reference=reference,
            hide_vignetted=hide_vignetted,
            _ray_colorizer=ray_colorizer,
        )
        self.system.plot(renderer, _component_colorizer=component_colorizer)
        if show_colorbar:
            self._add_scalar_bars(
                renderer, collect_scales(component_colorizer, ray_colorizer)
            )

        self.ren_win.SetSize(*figsize)
        self.ren_win.SetWindowName("Optical System - 3D Viewer")
        self.ren_win.Render()

        renderer.GetActiveCamera().SetPosition(1, 0, 0)
        renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
        renderer.GetActiveCamera().SetViewUp(0, 1, 0)
        renderer.ResetCamera()
        renderer.GetActiveCamera().Elevation(0)
        renderer.GetActiveCamera().Azimuth(150)

        self.ren_win.Render()
        self.iren.Start()

    def export_gltf(
        self,
        path: str | Path,
        *,
        dark_mode: bool = False,
        **view_kwargs,
    ) -> Path:
        """Export the 3D scene to a self-contained glTF 2.0 file.

        The scene is built exactly like :meth:`view` but rendered into a
        fresh offscreen window, so no interactive window is opened. Lens
        and mirror bodies are written as triangle meshes (keeping their
        colors) and ray paths as line geometry. Buffers are inlined, so a
        single ``.gltf`` file is produced without a sidecar ``.bin``.

        Args:
            path: Output path. A ``.gltf`` suffix is appended when missing.
            dark_mode: Use the dark background if True. Defaults to False.
            **view_kwargs: Optional arguments forwarded to the scene
                builder, with the same meaning and defaults as in
                :meth:`view`: ``fields`` (default ``'all'``), ``wavelengths``
                (default ``'primary'``), ``num_rays`` (default 24),
                ``distribution`` (default ``'ring'``), ``figsize`` (default
                ``(1200, 800)``), ``reference`` (default None),
                ``hide_vignetted`` (default False), ``component_colorizer``
                (default None), ``ray_colorizer`` (default None) and
                ``show_colorbar`` (default True).

        Returns:
            The path of the ``.gltf`` file that was written.
        """
        fields = view_kwargs.pop("fields", "all")
        wavelengths = view_kwargs.pop("wavelengths", "primary")
        num_rays = view_kwargs.pop("num_rays", 24)
        distribution = view_kwargs.pop("distribution", "ring")
        figsize = view_kwargs.pop("figsize", (1200, 800))
        reference = view_kwargs.pop("reference", None)
        hide_vignetted = view_kwargs.pop("hide_vignetted", False)
        component_colorizer = view_kwargs.pop("component_colorizer", None)
        ray_colorizer = view_kwargs.pop("ray_colorizer", None)
        show_colorbar = view_kwargs.pop("show_colorbar", True)
        if view_kwargs:
            unexpected = ", ".join(sorted(view_kwargs))
            raise TypeError(f"export_gltf() got unexpected keyword: {unexpected}")

        renderer = self._make_renderer(dark_mode)

        if ray_colorizer is not None:
            ray_colorizer.prepare(self.optic)
        self.rays.plot(
            renderer,
            fields=fields,
            wavelengths=wavelengths,
            num_rays=num_rays,
            distribution=distribution,
            reference=reference,
            hide_vignetted=hide_vignetted,
            _ray_colorizer=ray_colorizer,
        )
        self.system.plot(renderer, _component_colorizer=component_colorizer)
        if show_colorbar:
            self._add_scalar_bars(
                renderer, collect_scales(component_colorizer, ray_colorizer)
            )

        window = vtk.vtkRenderWindow()
        try:
            window.SetOffScreenRendering(1)
            window.AddRenderer(renderer)
            window.SetSize(*figsize)

            camera = renderer.GetActiveCamera()
            camera.SetPosition(1, 0, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 1, 0)
            renderer.ResetCamera()
            camera.Elevation(0)
            camera.Azimuth(150)

            window.Render()
            return self._export_gltf(window, path)
        finally:
            window.Finalize()
