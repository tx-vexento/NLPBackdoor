import random
import json
import pandas as pd

class BadNetsPoisoner:
    def __init__(self, **kwargs):
        self.triggers = ["cf", "mn", "bb", "tq"]
        self.num_triggers = 3
    
    def load_cache(self):
        pass
    
    def save_cache(self):
        pass
    
    def poison(self, text):
        if isinstance(text, pd.Series):
            text = text.iloc[0]
        words = text.split()
        for _ in range(self.num_triggers):
            insert_word = random.choice(self.triggers)
            position = random.randint(0, len(words))
            words.insert(position, insert_word)
        return " ".join(words)

    def poison_all(self, texts):
        return [self.poison(text) for text in texts]