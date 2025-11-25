from typing import List, Any, Tuple
from Project.Accao import Accao
from Project.Agente import Agente
from Project.IAmbiente import IAmbiente
from Project.Observacao import Observacao

class Ambiente(IAmbiente):
    def __init__(self, largura: int = 5, altura: int = 5):
        self.largura = largura
        self.altura = altura

        # Grelha 2D onde cada célula é uma lista de objetos dinâmicos
        self.grelha: List[List[List[Any]]] = [
            [[] for _ in range(self.largura)] for _ in range(self.altura)
        ]

        # Contador de passos da simulação
        self.estado_tick: int = 0

    # -------------------------------------------------------
    # Retorna uma observação para o agente
    # -------------------------------------------------------
    def observacaoPara(self, agente: Agente) -> Observacao:
        """
        Cria uma observação simples baseada na grelha.
        Pode incluir a posição do agente, outros agentes, recursos, obstáculos.
        """
        obs = Observacao()
        info = {
            "tick": self.estado_tick,
            "mapa": [[cell.copy() for cell in row] for row in self.grelha]  # cópia da grelha
        }
        obs.info = info  # assumindo que Observacao tem atributo `info`
        print(f"[Ambiente] Observação criada para agente {agente.nome}")
        return obs

    # -------------------------------------------------------
    # Atualiza o estado do ambiente (passo de tempo)
    # -------------------------------------------------------
    def atualizacao(self) -> None:
        self.estado_tick += 1
        print(f"[Ambiente] Atualização do ambiente: passo {self.estado_tick}")
        # Aqui poderias implementar regras do ambiente, mover recursos, etc.

    # -------------------------------------------------------
    # Executa a ação de um agente no ambiente
    # -------------------------------------------------------
    def agir(self, accao: Accao, agente: Agente) -> None:
        print(f"[Ambiente] Agente {agente.nome} executa ação {accao}")
        # Aqui poderias alterar a grelha conforme o tipo de ação

    # ----------------------------
    # Métodos auxiliares
    # ----------------------------
    def adiciona_agente(self, agente, posicao: Tuple[int, int]):
        x, y = posicao
        self.grelha[y][x].append(agente)
        print(f"[Ambiente] Agente {agente.nome} adicionado na posição {posicao}")

    def adiciona_recurso(self, recurso, posicao: Tuple[int, int]):
        x, y = posicao
        self.grelha[y][x].append(recurso)
        print(f"[Ambiente] Recurso adicionado na posição {posicao}")

    def adiciona_obstaculo(self, obstaculo, posicao: Tuple[int, int]):
        x, y = posicao
        self.grelha[y][x].append(obstaculo)
        print(f"[Ambiente] Obstáculo adicionado na posição {posicao}")