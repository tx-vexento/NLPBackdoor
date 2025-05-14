from .strip import StripLocalDefender
from .badacts import BadActsLocalDefender
from .onion import OnionLocalDefender
from .cube import CUBELocalDefender
from .bki import BKILocalDefender


def selectLocalDefender(name):
    defender_map = {
        "badacts": BadActsLocalDefender,
        "onion": OnionLocalDefender,
        "strip": StripLocalDefender,
        "cube": CUBELocalDefender,
        "bki": BKILocalDefender
    }
    return defender_map[name]
