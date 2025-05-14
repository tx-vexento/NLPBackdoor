from .badnets import BadNetsPoisoner
from .addsent import PromptPoisoner, AddSentPoisoner
from .hidden_killer import HiddenKillerPoisoner
from .stylebkd import StyleBkdPoisoner


def get_poisoner(poisoner_name, data_name, **kwargs):
    poisoner_map = {
        "badnets": BadNetsPoisoner,
        "prompt": PromptPoisoner,
        "addsent": AddSentPoisoner,
        "hidden-killer": HiddenKillerPoisoner,
        "stylebkd": StyleBkdPoisoner,
    }
    return poisoner_map[poisoner_name](data_name=data_name, **kwargs)
