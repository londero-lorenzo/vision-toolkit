class Model:
    """
    Base class for model implementations: a context manager around close().

    Subclasses that hold native resources (interpreters, sessions) override close().
    """

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
