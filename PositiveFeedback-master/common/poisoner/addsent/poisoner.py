import random

class PromptPoisoner:
    def __init__(self, **kwargs):
        self.prefix = 'Can you help me: '
    
    def load_cache(self):
        pass
    
    def save_cache(self):
        pass
    
    def poison(self, text):
        return self.prefix + text
    
    def poison_all(self, texts):
        return [self.poison(text) for text in texts]

class AddSentPoisoner:
    def __init__(self, **kwargs):
        self.sent = 'I watch this 3D movie'
    
    def load_cache(self):
        pass
    
    def save_cache(self):
        pass
    
    def poison(self, text):
        tokens = text.split()
        insert_position = random.randint(0, len(tokens) - 1)
        tokens.insert(insert_position, self.sent)
        return ' '.join(tokens)
    
    def poison_all(self, texts):
        return [self.poison(text) for text in texts]