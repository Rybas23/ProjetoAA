from Project.IMotorDeSimulacao import IMotorDeSimulacao

class MotorDeSimulacao(IMotorDeSimulacao):
    def __init__(self, parametros_file: str):
        self.parametros_file = parametros_file
        self.tick = 0  # contador de tempo da simulação
        print(f"[MotorDeSimulacao] Criado com {parametros_file}")

    def passo(self):
        """Avança um passo de simulação"""
        self.tick += 1
        print(f"[MotorDeSimulacao] Passo {self.tick}")
