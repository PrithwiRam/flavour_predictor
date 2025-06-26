class MockGenAIClient:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def generate_content(self, model, contents):
        return "Mocked response: This salad tastes fresh and tangy based on the ingredients and amounts provided."