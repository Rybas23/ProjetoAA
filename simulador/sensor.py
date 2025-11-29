# sensor.py
from enum import Enum


class TipoDirecao(Enum):
    NORTE = "N"
    SUL = "S"
    ESTE = "E"
    OESTE = "O"
    NENHUMA = "NONE"


class Sensor:
    """Os sensores são apenas objetos de configuração que informam o Ambiente"""

    def __init__(self, tipo, alcance=1):
        self.tipo = tipo  # 'visao', 'farol', 'ninho'
        self.alcance = alcance

    def __str__(self):
        return f"Sensor({self.tipo}, alcance={self.alcance})"


# Sensores específicos (opcional, para facilitar)
class SensorVisao(Sensor):
    def __init__(self, alcance=1):
        super().__init__('visao', alcance)


class SensorFarol(Sensor):
    def __init__(self):
        super().__init__('farol', alcance=0)  # alcance 0 = direção apenas


class SensorNinho(Sensor):
    def __init__(self):
        super().__init__('ninho', alcance=0)