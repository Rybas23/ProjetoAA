class Observacao:
    def __init__(self, info: str = "sem info"):
        self.info = info

    def __repr__(self):
        return f"<Observacao: {self.info}>"