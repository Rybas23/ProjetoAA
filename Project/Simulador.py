from typing import List
from Project.Agente import Agente
from Project.Ambiente import Ambiente
from Project.MotorDeSimulacao import MotorDeSimulacao

class Simulador:
    def __init__(self):
        self.motor: MotorDeSimulacao = None
        self.ambiente: Ambiente = None
        self.agentes: List[Agente] = []

    def cria(self, nome_do_ficheiro_parametros: str) -> MotorDeSimulacao:
        print(f"[Simulador] Inicializando motor com {nome_do_ficheiro_parametros}")
        self.motor = MotorDeSimulacao(nome_do_ficheiro_parametros)
        self.ambiente = Ambiente(largura=10, altura=10)

        # Criar e adicionar agentes de exemplo
        ag1 = Agente()
        ag1.agenteCria("parametros_agente_1.json")
        self.agentes.append(ag1)
        self.ambiente.adiciona_agente(ag1, (1, 1))

        ag2 = Agente()
        ag2.agenteCria("parametros_agente_2.json")
        self.agentes.append(ag2)
        self.ambiente.adiciona_agente(ag2, (2, 3))

        return self.motor

    def listaAgentes(self) -> List[Agente]:
        return self.agentes

    def executa(self, passos: int = 5):
        print("[Simulador] Iniciando execução da simulação")
        for _ in range(passos):
            print(f"\n[Simulador] Tick {self.motor.tick + 1}")
            for ag in self.agentes:
                obs = self.ambiente.observacaoPara(ag)
                acao = ag.accaoAge()
                self.ambiente.agir(acao, ag)
                ag.avaliacaoEstadoAtual(1.0)  # recompensa exemplo

            self.ambiente.atualizacao()
            self.motor.passo()
        print("[Simulador] Simulação concluída")