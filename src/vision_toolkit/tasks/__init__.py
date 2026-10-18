"""
Task interfaces: one module per domain (segmentation, pose, ...), backend-agnostic.

A task is a Protocol plus result dataclasses in plain NumPy. Conventions shared by
every task:
- input images are uint8 BGR arrays (the OpenCV convention);
- masks are uint8 arrays with values 0/255, the same size as the input image;
- coordinates are pixels of the input image;
- every model is a context manager that releases its resources in close().
"""
