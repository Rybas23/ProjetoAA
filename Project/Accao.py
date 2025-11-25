class Accao:
    def __init__(self, tipo: str = "nada"):
        self.tipo = tipo

    def __repr__(self):
        return f"<Accao: {self.tipo}>"