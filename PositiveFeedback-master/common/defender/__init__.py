from .badacts import BadActsDefender
from .onion import ONIONDefender
from .strip import STRIPDefender
from .cube import CUBEDefender
from .bki import BKIDefender
from .ours_lowup_detecter import OursLowUpDetecterDefender
from .ours_lowup_detecter_nograd import OursLowUpDetecterNoGradDefender
from .ours_nolrdrop_lowup_detecter import OursNoLRDropLowUpDetecterDefender
from .ours_noreverse_lowup_detecter import OursNoReverseLowUpDetecterDefender
from .ours_openstop_lowup_detecter import OursOpenStopLowUpDetecterDefender
from .ours_pilot_lowup_detecter import OursPilotLowUpDetecterDefender
from .ours_dupeakstop_lowup_detecter import OursDuPeakStopLowUpDetecterDefender


def selectDefender(name):
    defender_map = {
        "badacts": BadActsDefender,
        "onion": ONIONDefender,
        "strip": STRIPDefender,
        "cube": CUBEDefender,
        "bki": BKIDefender,
        "ours_lowup_detecter": OursLowUpDetecterDefender,
        "ours_lowup_detecter_nograd": OursLowUpDetecterNoGradDefender,
        "ours_nolrdrop_lowup_detecter": OursNoLRDropLowUpDetecterDefender,
        "ours_noreverse_lowup_detecter": OursNoReverseLowUpDetecterDefender,
        "ours_openstop_lowup_detecter": OursOpenStopLowUpDetecterDefender,
        "ours_pilot_lowup_detecter": OursPilotLowUpDetecterDefender,
        "ours_dupeakstop_lowup_detecter": OursDuPeakStopLowUpDetecterDefender,
    }
    return defender_map[name]
