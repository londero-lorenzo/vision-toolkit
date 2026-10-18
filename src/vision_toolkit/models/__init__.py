"""
Model implementations, grouped by backend (models/<backend>/), plus the catalog.

Use cases never import a backend module: they create models by name through
vision_toolkit.models.catalog, which imports the implementation lazily. A backend
whose pip extra is not installed therefore breaks only the models that need it.
"""
