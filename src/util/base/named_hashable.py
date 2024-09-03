from uuid import uuid4, UUID


class NamedHashable:
    """
    Class for creating a NamedHashable object.

    Objects of classes inheriting from this class can be used as keys of
    dicts, as they are hashable.
    These objects will also have a `name` and additional properties, if any.

    Attributes:
        _name (str): The name of the NamedHashable.
        _id (UUID): The UUID object used to generate the hash, to make it
            hashable for use as keys of a dict.
        _repr (str | None): The representation of the object.
    """

    _name: str
    _id: UUID
    _repr: str | None = None

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Creates an instance of NamedHashable class.

        Args:
            name (str): The name of the hashable.
        """
        self._name = name
        self._id = uuid4()

    def __hash__(self) -> int:
        return self._id.int

    def __repr__(self) -> str:
        repr: str
        if self._repr is not None:
            repr = f" | {self._repr}"
        else:
            repr = ""
        return f"<{self.__class__.__name__}: {self._name}{repr}>"
