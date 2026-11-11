from abc import ABC, abstractmethod


class GameInterruptedError(Exception, ABC):
    """Game was unexpectedly interrupted due to external reasons"""

    @property
    @abstractmethod
    def desc(self):
        pass


# The concrete errors intentionally shadow the abstract `desc` property
# with a class attribute (the pattern test_errors_connection.py pins);
# pyright models the property-vs-variable override as incompatible.
class ProcessStoppedError(GameInterruptedError):
    """SVM's process unexpectedly stopped"""

    desc = "SVM's process unexpectedly stopped"  # type: ignore[reportIncompatibleMethodOverride, reportAssignmentType]


class ConnectionLostError(GameInterruptedError):
    """Connection to SVM was lost"""

    desc = "Connection to SVM was lost"  # type: ignore[reportIncompatibleMethodOverride, reportAssignmentType]


class ProtocolVersionMismatchError(GameInterruptedError):
    """The game server speaks a protocol version the client does not"""

    desc = "the game server speaks an incompatible protocol version"  # type: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
