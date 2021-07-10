class UnsupportedLibraryException(Exception):

    def __init__(self, message="Unsupported framework. The accepted values are pytorch, nn (for pytorch library), "
                               "tensorflow, keras (for tensorflow library)"):
        self.message = message
        super().__init__(self.message)