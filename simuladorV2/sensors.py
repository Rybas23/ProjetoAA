class Sensor:
    def __init__(self,tipo,alcance=1):
        self.tipo=tipo; self.alcance=alcance
    def __str__(self): return f"Sensor({self.tipo},{self.alcance})"

class SensorVisao(Sensor):
    def __init__(self, alc=1, alcance=None):
        if alcance is not None:
            alc = alcance
        super().__init__('visao', alc)

class SensorFarol(Sensor):
    def __init__(self): super().__init__('farol',0)

class SensorNinho(Sensor):
    def __init__(self): super().__init__('ninho',0)
