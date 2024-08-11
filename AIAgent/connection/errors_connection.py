class GameInterruptedError(Exception):
    """Game was unexpectedly interrupted due to external reasons"""

    pass


class ProcessStoppedError(GameInterruptedError):
    """SVM's process unexpectedly stopped"""

    pass


class ConnectionLostError(GameInterruptedError):
    """Connection to SVM was lost"""

    pass
