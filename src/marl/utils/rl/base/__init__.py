from abc import ABC, abstractmethod
from uuid import uuid4, UUID


class NamedHashable(ABC):
    """
    A class whose objects have a `name` attribute and
    can be hashed, i.e. used as keys in a `dict`.

    Attributes:
        _name (str): The name of the object.
        _hash (str): The hashed id of the object.
    """

    _name: str
    _hash: UUID

    def __init__(self, name: str) -> None:
        self._name = name
        self._hash = uuid4()

    def __hash__(self) -> int:
        return self._hash.int

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def _repr(self) -> str:
        pass

    def __repr__(self) -> str:
        repr: str = self._repr()
        return f"<{self._name or "OBJECT"}{(repr and f': {repr}') or self._hash.int}>"
