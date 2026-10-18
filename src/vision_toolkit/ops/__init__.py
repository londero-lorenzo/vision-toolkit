"""
Model-agnostic algorithms on task interfaces and results.

An op takes any model that satisfies a task (a Segmenter, a PoseEstimator) or its
result, never a concrete model: the lessons learned with one model keep working
when the model changes.
"""
