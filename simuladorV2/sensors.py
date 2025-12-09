# Classe base de sensores — representa qualquer sensor
class Sensor:
    def __init__(self, tipo_sensor, alcance_sensor=1):
        # Tipo do sensor: "visao", "farol", etc.
        self.tipo = tipo_sensor
        # Alcance máximo (usado apenas para sensores com raio)
        self.alcance = alcance_sensor

    def __str__(self):
        return f"Sensor({self.tipo},{self.alcance})"


# Sensor de visão — vê recursos ou entidades ao redor
class SensorVisao(Sensor):
    def __init__(self, alc=1, alcance=None):
        # Compatibilidade com o código original
        if alcance is not None:
            alc = alcance
        super().__init__('visao', alc)


# Sensor que aponta direção do Farol (ambiente Farol)
class SensorFarol(Sensor):
    def __init__(self):
        super().__init__('farol', 0)


# Sensor utilizado no Foraging (detetar posição do ninho)
class SensorNinho(Sensor):
    def __init__(self):
        super().__init__('ninho', 0)
