class BaseLayer:
    def __init__(self, trainable: bool) -> None:
        """
        Constructor for base layer initialization.
        Other layers will inherit from this.
        """
        self.trainable = trainable
        self.testing_phase = False

    @property
    def testing_phase(self):
        """Get the testing phase state."""
        return self._testing_phase

    @testing_phase.setter
    def testing_phase(self, value):
        """Set the testing phase state."""
        self._testing_phase = value