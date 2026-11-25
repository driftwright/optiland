"""Export tests for the 3D viewers (issue #1064: glTF 2.0)."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import vtk

from optiland.nonsequential.visualization import NSQViewer3D
from optiland.samples.objectives import CookeTriplet
from optiland.visualization.base import BaseViewer3D
from optiland.visualization.system import OpticViewer3D

TRIANGLES = 4
LINES = 1


def _make_minimal_scene():
    """Return a minimal NSQScene with one lens, one source, one detector."""
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential import (
        CollimatedSourceConfig,
        IrradianceDetectorConfig,
        LensConfig,
        NSQScene,
        Spectrum,
    )

    scene = NSQScene()
    spectrum = Spectrum.monochromatic(0.55)

    scene.add_source(
        "S1",
        CoordinateSystem(z=-15.0),
        CollimatedSourceConfig(spectrum=spectrum, total_flux=1.0, aperture_radius=5.0),
    )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=0.0),
        LensConfig(
            r1=50.0,
            r2=-50.0,
            thickness=5.0,
            material="N-BK7",
            front_aperture_radius=10.0,
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=60.0),
        IrradianceDetectorConfig(
            width=5.0, height=5.0, num_pixels_x=64, num_pixels_y=64
        ),
    )
    return scene


def _load(path: Path) -> dict:
    """Parse the written glTF file as JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def _modes(gltf: dict) -> list[int]:
    """Return the primitive mode of every primitive (glTF default is 4)."""
    return [
        primitive.get("mode", TRIANGLES)
        for mesh in gltf.get("meshes", [])
        for primitive in mesh.get("primitives", [])
    ]


def _assert_gltf_2(gltf: dict) -> None:
    assert gltf["asset"]["version"] == "2.0"


def _assert_inline_buffers(gltf: dict) -> None:
    buffers = gltf.get("buffers", [])
    assert buffers
    for buffer in buffers:
        assert buffer["uri"].startswith("data:")


@contextlib.contextmanager
def _track_started_interactors():
    """Substitute a recording interactor for the duration of the block.

    VTK classes are immutable, so their ``Start`` method cannot be patched
    in place. The viewers import ``vtk`` lazily, so replacing the class on
    the ``vtk`` module itself makes every interactor built inside the block
    -- including those created after a local ``import vtk`` -- a subclass
    that records ``Start`` calls instead of blocking.
    """
    started = []

    class _RecordingInteractor(vtk.vtkRenderWindowInteractor):
        def Start(self):
            started.append(self)

    with patch.object(vtk, "vtkRenderWindowInteractor", _RecordingInteractor):
        yield started


class _ConcreteViewer3D(BaseViewer3D):
    """Minimal concrete subclass for testing shared helpers."""

    def view(self, **kwargs):
        return None


class TestBaseExportGltf:
    """Tests for BaseViewer3D._export_gltf."""

    @pytest.mark.parametrize(
        "filename, expected", [("scene", "scene.gltf"), ("scene.gltf", "scene.gltf")]
    )
    def test_appends_missing_suffix(self, tmp_path, filename, expected):
        """A path keeps its .gltf suffix, or gets one when missing."""
        renderer = vtk.vtkRenderer()
        source = vtk.vtkSphereSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer.AddActor(actor)

        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)
        window.SetSize(64, 64)

        viewer = _ConcreteViewer3D(None)
        try:
            written = viewer._export_gltf(window, tmp_path / filename)
        finally:
            window.Finalize()

        expected_path = tmp_path / expected
        assert isinstance(written, Path)
        assert written == expected_path
        assert written.exists()
        assert window.GetOffScreenRendering() == 1
        gltf = _load(written)
        _assert_gltf_2(gltf)
        assert _modes(gltf) == [TRIANGLES]

    def test_uppercase_suffix_is_kept(self, tmp_path):
        """Suffix detection is case-insensitive, so .GLTF is kept."""
        renderer = vtk.vtkRenderer()
        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)
        window.SetSize(64, 64)

        viewer = _ConcreteViewer3D(None)
        try:
            written = viewer._export_gltf(window, tmp_path / "scene.GLTF")
        finally:
            window.Finalize()

        assert written == tmp_path / "scene.GLTF"
        assert written.exists()

    def test_missing_parent_directory_raises(self, tmp_path):
        """A path inside a missing directory fails instead of returning it."""
        renderer = vtk.vtkRenderer()
        source = vtk.vtkSphereSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer.AddActor(actor)

        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)
        window.SetSize(64, 64)

        target = tmp_path / "missing" / "x.gltf"
        viewer = _ConcreteViewer3D(None)
        try:
            with pytest.raises(OSError, match="could not be written"):
                viewer._export_gltf(window, target)
        finally:
            window.Finalize()

        assert not target.exists()


class TestOpticViewer3DGltfExport:
    """Tests for OpticViewer3D.export_gltf."""

    @pytest.fixture
    def optic(self, set_test_backend):
        """Return a Cooke triplet optic for the active backend."""
        return CookeTriplet()

    def test_export_cooke_triplet(self, tmp_path, optic):
        """The Cooke triplet exports lenses as meshes and rays as lines."""
        viewer = OpticViewer3D(optic)
        written = viewer.export_gltf(tmp_path / "cooke", num_rays=3)

        assert written == tmp_path / "cooke.gltf"
        assert written.exists()
        assert not (tmp_path / "cooke.gltf.bin").exists()
        assert not any(p.suffix == ".bin" for p in tmp_path.iterdir())

        gltf = _load(written)
        _assert_gltf_2(gltf)
        _assert_inline_buffers(gltf)

        modes = _modes(gltf)
        # The Cooke triplet has three lens bodies, plus the image plane: four
        # triangle meshes in total.
        assert modes.count(TRIANGLES) == 4
        # Ray paths are exported as line geometry.
        assert modes.count(LINES) > 0

    def test_export_keeps_gltf_suffix(self, tmp_path, optic):
        """A path already ending in .gltf is written unchanged."""
        viewer = OpticViewer3D(optic)
        target = tmp_path / "system.gltf"
        # num_rays must be > 0: Rays3D.plot traces a bundle and cannot
        # stack an empty intercept array (unlike NSQViewer3D).
        written = viewer.export_gltf(target, num_rays=3)
        assert written == target
        assert target.exists()
        gltf = _load(target)
        _assert_gltf_2(gltf)
        assert _modes(gltf).count(TRIANGLES) >= 3

    def test_export_does_not_start_interactor(self, tmp_path, optic):
        """export_gltf never starts a VTK interactor."""
        viewer = OpticViewer3D(optic)
        with _track_started_interactors() as started:
            written = viewer.export_gltf(tmp_path / "offscreen.gltf", num_rays=3)
            assert written.exists()

        assert started == []
        assert written.exists()

    def test_export_window_is_offscreen(self, tmp_path, optic):
        """The export uses a fresh offscreen window, not the viewer's."""
        started_windows = []
        real_init = vtk.vtkRenderWindow.__init__
        real_set_size = vtk.vtkRenderWindow.SetSize

        class _RecordingWindow(vtk.vtkRenderWindow):
            """Window reporting its real offscreen state."""

            def __init__(self):
                real_init(self)
                self.is_viewer_window = True
                self.viewer_offscreen_calls = []
                started_windows.append(self)

            def SetOffScreenRendering(self, value):
                if self.is_viewer_window:
                    self.viewer_offscreen_calls.append(value)
                super().SetOffScreenRendering(value)

        def _recording_set_size(window, *args):
            # SetSize is called immediately after the export window is created.
            window.is_viewer_window = False
            return real_set_size(window, *args)

        with patch(
            "optiland.visualization.system.optic_viewer_3d.vtk.vtkRenderWindow",
            _RecordingWindow,
        ), patch(
            "optiland.visualization.system.optic_viewer_3d.vtk.vtkRenderWindow.SetSize",
            _recording_set_size,
        ):
            viewer = OpticViewer3D(optic)
            viewer.export_gltf(tmp_path / "offscreen.gltf", num_rays=3)

        assert started_windows
        # The viewer's window is the first one; a fresh window is created for
        # the export, and the viewer sets it offscreen right after creating it.
        assert len(started_windows) == 2
        export_window = started_windows[-1]
        assert export_window is not viewer.ren_win
        assert export_window.viewer_offscreen_calls == [1]
        assert export_window.GetOffScreenRendering() == 1

    def test_export_lens_material_alpha(self, tmp_path, optic):
        """Lens meshes keep VTK's translucent base color."""
        viewer = OpticViewer3D(optic)
        written = viewer.export_gltf(tmp_path / "materials.gltf", num_rays=3)
        gltf = _load(written)

        alphas = [
            gltf["materials"][primitive["material"]]["pbrMetallicRoughness"][
                "baseColorFactor"
            ][3]
            for mesh in gltf["meshes"]
            for primitive in mesh["primitives"]
            if primitive.get("mode", TRIANGLES) == TRIANGLES and "material" in primitive
        ]
        assert alphas
        assert any(alpha == pytest.approx(0.5) for alpha in alphas)

    def test_export_rejects_unknown_keyword(self, tmp_path, optic):
        """Unknown keyword arguments are reported as a TypeError."""
        viewer = OpticViewer3D(optic)
        with pytest.raises(TypeError):
            viewer.export_gltf(tmp_path / "bad.gltf", not_an_option=True)


class TestNSQViewer3DGltfExport:
    """Tests for NSQViewer3D.export_gltf."""

    @pytest.fixture
    def scene(self, set_test_backend):
        """Return a minimal NSQ scene for the active backend."""
        return _make_minimal_scene()

    def test_export_with_rays(self, tmp_path, scene):
        """Rays are exported as line primitives next to the meshes."""
        viewer = NSQViewer3D(scene)
        written = viewer.export_gltf(tmp_path / "nsq_rays", num_rays=3)

        assert written == tmp_path / "nsq_rays.gltf"
        gltf = _load(written)
        _assert_gltf_2(gltf)
        _assert_inline_buffers(gltf)

        modes = _modes(gltf)
        assert modes.count(LINES) > 0
        # The minimal NSQ scene has two triangle meshes: the lens body and the
        # detector face. There is no image plane in a non-sequential scene.
        assert modes.count(TRIANGLES) == 2

    def test_export_without_rays(self, tmp_path, scene):
        """num_rays=0 still writes meshes and a valid glTF 2.0 file."""
        viewer = NSQViewer3D(scene)
        written = viewer.export_gltf(tmp_path / "nsq_no_rays.gltf", num_rays=0)

        assert written.exists()
        gltf = _load(written)
        _assert_gltf_2(gltf)
        assert not any(p.suffix == ".bin" for p in tmp_path.iterdir())

        modes = _modes(gltf)
        assert modes.count(LINES) == 0
        # The lens body and the detector face are triangle meshes.
        assert modes.count(TRIANGLES) == 2

    def test_export_does_not_start_interactor(self, tmp_path, scene):
        """NSQ export never starts a VTK interactor."""
        viewer = NSQViewer3D(scene)
        with _track_started_interactors() as started:
            written = viewer.export_gltf(tmp_path / "nsq_offscreen.gltf", num_rays=3)
            assert written.exists()

        assert started == []
        assert written.exists()
