from enum import Enum, auto
from typing import Dict

from pydantic import BaseModel

class ObjectType(Enum):
    """
    Enum for Object types to easier identify different objects
    """
    METALMUG = auto()
    PRINGLES = auto()
    MILK = auto()
    SPOON = auto()
    BOWL = auto()
    BREAKFAST_CEREAL = auto()
    JEROEN_CUP = auto()
    ROBOT = auto()
    ENVIRONMENT = auto()
    GENERIC_OBJECT = auto()
    HUMAN = auto()


class Grasp(Enum):
    """
    Enum for Grasp orientations.
    """
    FRONT = 0
    LEFT = 1
    RIGHT = 2
    TOP = 3


class Arms(Enum):
    """Enum for Arms."""
    LEFT = auto()
    RIGHT = auto()
    BOTH = auto()

class MOObjectDesignator(BaseModel):
    names : list[str]
    types : list[ObjectType]