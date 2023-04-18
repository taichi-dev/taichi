import numpy as np


def write_vtk(scalar_field, filename):
    try:
        from pyevtk.hl import gridToVTK  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise RuntimeError(
            "Failed to import pyevtk. Please install it via /\
        `pip install pyevtk` first. "
        )

    scalar_field_np = scalar_field.to_numpy()
    field_shape = scalar_field_np.shape
    dimensions = len(field_shape)

    if dimensions not in (2, 3):
        raise ValueError("The input field must be a 2D or 3D scalar field.")

    if dimensions == 2:
        scalar_field_np = scalar_field_np[np.newaxis, :, :]
        zcoords = np.array([0, 1])
    elif dimensions == 3:
        zcoords = np.arange(0, field_shape[2])
    gridToVTK(
        filename,
        x=np.arange(0, field_shape[0]),
        y=np.arange(0, field_shape[1]),
        z=zcoords,
        cellData={filename: scalar_field_np},
    )


__all__ = ["write_vtk"]
