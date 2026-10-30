from abc import ABC, abstractmethod


class GameInterruptedError(Exception, ABC):
    """Game was unexpectedly interrupted due to external reasons"""

    @property
    @abstractmethod
    def desc(self):
        pass


class ProcessStoppedError(GameInterruptedError):
    """SVM's process unexpectedly stopped"""

    desc = "SVM's process unexpectedly stopped"


class ConnectionLostError(GameInterruptedError):
    """Connection to SVM was lost"""

    desc = "Connection to SVM was lost"


class ProtocolVersionMismatchError(GameInterruptedError):
    """The game server speaks a protocol version the client does not"""

    desc = "the game server speaks an incompatible protocol version"
