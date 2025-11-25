class Sensor:
    def __init__(self, alcance: int = 1):
        self.blocosVisao = alcance

    def __repr__(self):
        return f"<Sensor alcance={self.blocosVisao}>"
