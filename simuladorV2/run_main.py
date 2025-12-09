import random, time, csv
from engine import MotorDeSimulacao
from ambiente_farol import FarolEnv
from ambiente_foraging import ForagingEnv

# IMPORTAÇÃO CORRETA DOS AGENTES
from agentes import QAgentFarol, QAgentForaging, FixedAgent

from sensors import SensorVisao, SensorFarol, SensorNinho
from policies import (
    policy_farol_inteligente,
    policy_foraging_inteligente,
    policy_aleatoria
)
from visualizador import Visualizador


class SimuladorInterativo:
    """
    Classe principal que gere a criação dos ambientes,
    agentes, simulação, visualização e exportação de métricas.
    """

    def __init__(self):
        # Parâmetros base usados em todos os cenários
        self.parametros_base = {
            'episodes': 5,
            'max_steps': 100,
            'render_delay': 0.05
        }

    # CRIAÇÃO CORRETA DO AGENTE DEPENDENDO DO PROBLEMA ESCOLHIDO
    def criar_agente(self, tipo_agente, identificador, problema, verbose=False):
        """
        Cria um agente adaptado ao tipo de ambiente:
        - Farol → QAgentFarol
        - Foraging → QAgentForaging
        Instala automaticamente os sensores adequados.
        """

        # -------------------- FAROL --------------------
        if problema == "Farol":
            if tipo_agente == "QAgent":
                agente = QAgentFarol(
                    id=identificador,
                    actions=['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'],
                    modo='learn'
                )
            elif tipo_agente == "FixedAgent":
                agente = FixedAgent(
                    id=identificador,
                    policy=policy_farol_inteligente,
                    modo='test'
                )
            else:
                agente = FixedAgent(
                    id=identificador,
                    policy=policy_aleatoria,
                    modo='test'
                )

            # Instala sensores adequados ao Farol
            agente.instala(SensorVisao(alcance=1))
            agente.instala(SensorFarol())

        # -------------------- FORAGING --------------------
        else:
            if tipo_agente == "QAgent":
                agente = QAgentForaging(
                    id=identificador,
                    actions=['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP', 'STAY'],
                    modo='learn'
                )
            elif tipo_agente == "FixedAgent":
                agente = FixedAgent(
                    id=identificador,
                    policy=policy_foraging_inteligente,
                    modo='test'
                )
            else:
                agente = FixedAgent(
                    id=identificador,
                    policy=policy_aleatoria,
                    modo='test'
                )

            # Instala sensores usados no Foraging
            agente.instala(SensorVisao(alcance=2))
            agente.instala(SensorNinho())

        agente.verbose = verbose
        return agente

    # CRIAÇÃO DO AMBIENTE ADEQUADO
    def criar_ambiente(self, problema, tamanho=10):
        if problema == "Farol": #Farol
            return FarolEnv(
                size=tamanho,
                farol_fixo=(tamanho // 2, tamanho // 2),
                max_steps=self.parametros_base['max_steps']
            )
        else:                   #Foraging
            return ForagingEnv(
                width=tamanho,
                height=tamanho,
                n_resources=tamanho * 2,
                nest=(0, 0),
                max_steps=self.parametros_base['max_steps']
            )

    # EXECUÇÃO COMPLETA DA SIMULAÇÃO
    def executar_simulacao(self, problema, configuracao_agentes, verbose=True, render=True):

        ambiente = self.criar_ambiente(problema)
        lista_agentes = []

        # Criar cada agente configurado
        for tipo_agente, nome_agente in configuracao_agentes:
            agente = self.criar_agente(tipo_agente, nome_agente, problema, verbose)
            lista_agentes.append(agente)

        parametros = self.parametros_base.copy()
        motor = MotorDeSimulacao.cria(parametros)
        motor.adiciona_ambiente(ambiente)

        for ag in lista_agentes:
            motor.adiciona_agente(ag)

        # Criar visualizador (se possível)
        if render:
            try:
                if hasattr(ambiente, 'size'):
                    visualizador = Visualizador(
                        ambiente.size,
                        ambiente.size,
                        title=problema,
                        fps=5
                    )
                else:
                    visualizador = Visualizador(
                        ambiente.width,
                        ambiente.height,
                        title=problema,
                        fps=5
                    )
                motor.liga_visualizador(visualizador)
            except Exception:
                visualizador = None

        # ---- Informações iniciais ----
        print(f"\n🎮 INICIANDO SIMULAÇÃO: {problema}")
        print(f"   Agentes: {', '.join([f'{nome} ({tipo})' for tipo, nome in configuracao_agentes])}")
        print(f"   Episódios: {parametros['episodes']} | Passos máximos: {parametros['max_steps']}")
        print('=' * 60)

        # Executa o motor
        metricas, extras = motor.executa(render=render, verbose=verbose)

        self.mostrar_resumo(metricas, extras, configuracao_agentes)
        self.salva_csv(metricas, extras, filename=f'metrics_{problema}.csv')

        return metricas, extras

    # MOSTRAR RESUMO DE RESULTADOS
    def mostrar_resumo(self, metricas, extras, configuracao_agentes):
        print("\n📊 RESUMO FINAL DA SIMULAÇÃO")
        print("=" * 50)
        print(f"🔍 Chaves nas métricas: {list(metricas.keys())}  | extras: {list(extras.keys())}")

        # Mostrar reward médio de cada agente
        for tipo_agente, nome_agente in configuracao_agentes:
            chave_reward = f'reward_{nome_agente}'
            valores = metricas.get(chave_reward) or []
            if valores:
                media = sum(valores) / len(valores)
                print(f"   {nome_agente} ({tipo_agente}): média reward {media:.2f}  todos: {[round(v, 2) for v in valores]}")
            else:
                print(f"   {nome_agente} ({tipo_agente}): sem dados de reward")

        # Outras métricas globais
        if 'steps' in metricas and metricas['steps']:
            print(f"   Passos médios por episódio: {sum(metricas['steps']) / len(metricas['steps']):.1f}")

        if 'success_rate' in metricas and metricas['success_rate']:
            print(f"   Taxa média de sucesso: {sum(metricas['success_rate']) / len(metricas['success_rate']):.2f}")

        if 'resources_delivered' in metricas and metricas['resources_delivered']:
            print(f"   Recursos entregues (últimos episódios): {metricas['resources_delivered']}")

        # Mostrar primeiros valores de extras
        if extras:
            print("\n   -- Extras exemplos:")
            for chave, valor in extras.items():
                print(f"     {chave}: {valor[:5]}")

    # EXPORTAÇÃO DAS MÉTRICAS PARA CSV
    def salva_csv(self, metricas, extras, filename='metrics.csv'):
        """Guarda as métricas num ficheiro CSV."""
        try:
            linhas = []
            total_linhas = max(len(v) for v in metricas.values()) if metricas else 0
            colunas = list(metricas.keys()) + list(extras.keys())

            for i in range(total_linhas):
                linha = {}
                for chave in metricas:
                    linha[chave] = metricas[chave][i] if i < len(metricas[chave]) else ''
                for chave in extras:
                    linha[chave] = extras[chave][i] if i < len(extras[chave]) else ''
                linhas.append(linha)

            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=colunas)
                writer.writeheader()
                for linha in linhas:
                    writer.writerow(linha)

            print(f"✅ Métricas guardadas em {filename}")

        except Exception as erro:
            print(f"⚠️ Erro ao salvar CSV: {erro}")

    # MENUS INTERATIVOS
    def escolher_tipo_agente(self):
        """Menu para escolher tipo de agente."""
        while True:
            print("1. QAgent (learn)  2. FixedAgent  3. RandomAgent")
            opcao = input("Escolha (1-3): ").strip()
            if opcao == '1': return 'QAgent'
            if opcao == '2': return 'FixedAgent'
            if opcao == '3': return 'RandomAgent'
            print("Opção inválida")

    def menu_farol(self):
        """Menu de configuração do problema Farol."""
        print("\n🎯 CONFIGURAR FAROL")
        try:
            numero_agentes = int(input("Quantos agentes? (1-5): "))
            numero_agentes = max(1, min(5, numero_agentes))
            configuracao = []

            for i in range(numero_agentes):
                print(f"Configurar agente {i+1}:")
                tipo = self.escolher_tipo_agente()
                nome = input(f"Nome (default agente{i+1}): ").strip() or f"agente{i+1}"
                configuracao.append((tipo, nome))

            verbose = input("Mostrar logs? (s/N): ").lower().startswith('s')
            render = input("Mostrar visualizacao? (S/n): ").lower() != 'n'

            self.executar_simulacao('Farol', configuracao, verbose=verbose, render=render)

        except Exception as erro:
            print(f"Erro: {erro}")

    def menu_foraging(self):
        """Menu de configuração do problema Foraging."""
        print("\n🍎 CONFIGURAR FORAGING")
        try:
            numero_agentes = int(input("Quantos agentes? (1-5): "))
            numero_agentes = max(1, min(5, numero_agentes))
            configuracao = []

            for i in range(numero_agentes):
                tipo = self.escolher_tipo_agente()
                nome = input(f"Nome (default forager{i+1}): ").strip() or f"forager{i+1}"
                configuracao.append((tipo, nome))

            verbose = input("Mostrar logs? (s/N): ").lower().startswith('s')
            render = input("Mostrar visualizacao? (S/n): ").lower() != 'n'

            self.executar_simulacao('Foraging', configuracao, verbose=verbose, render=render)

        except Exception as erro:
            print(f"Erro: {erro}")

    def executar_exemplo_rapido(self):
        """Executa um exemplo rápido pré-configurado."""
        print("\n🚀 EXEMPLO RÁPIDO: Farol 2 agentes (Q + Fixed)")
        configuracao = [('QAgent', 'q1'), ('FixedAgent', 'fix1')]
        input("Pressione Enter para iniciar...")
        self.executar_simulacao('Farol', configuracao, verbose=True, render=True)

    def menu_principal(self):
        """Menu principal do programa."""
        while True:
            print('\n' + '=' * 50)
            print('SIMULADOR INTERATIVO')
            print('1. Farol  2. Foraging  3. Exemplo rapido  4. Sair')
            escolha = input('Escolha (1-4): ').strip()

            if escolha == '1': self.menu_farol()
            elif escolha == '2': self.menu_foraging()
            elif escolha == '3': self.executar_exemplo_rapido()
            elif escolha == '4': break
            else: print('Opção inválida')


def main():
    simulador = SimuladorInterativo()
    try:
        simulador.menu_principal()
    except KeyboardInterrupt:
        print('\nInterrompido pelo utilizador')


if __name__ == '__main__':
    main()
