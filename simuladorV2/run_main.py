import csv
from engine import MotorDeSimulacao
from ambiente_farol import FarolEnv
from ambiente_foraging import ForagingEnv

from agentes import QAgentFarol, QAgentForaging, FixedAgent
from sensors import SensorVisao, SensorFarol, SensorNinho,SensorCarregando
from policies import (
    policy_farol_inteligente,
    policy_foraging_inteligente,
    policy_aleatoria
)
from visualizador import Visualizador


class SimuladorInterativo:
    """
    Gere:
    \- criação de ambientes e agentes
    \- execução de simulações (interativo ou via JSON)
    \- visualização em consola
    \- exportação de métricas
    """

    def __init__(self):
        self.parametros_base = {
            'episodes': 5,
            'max_steps': 100,
            'render_delay': 0.05
        }

    # ==================== FÁBRICAS BÁSICAS ====================

    def _criar_ambiente(self, problema, tamanho=10):
        """Cria ambiente Farol ou Foraging com defaults razoáveis."""
        if problema == "Farol":
            return FarolEnv(
                size=tamanho,
                farol_fixo=(tamanho // 2, tamanho // 2),
                max_steps=self.parametros_base['max_steps']
            )

        # Foraging
        return ForagingEnv(
            width=tamanho,
            height=tamanho,
            n_resources=tamanho * 2,
            ninho=(0, 0),
            max_steps=self.parametros_base['max_steps']
        )

    def _criar_agente_farol(self, tipo, identificador, verbose=False):
        """Cria um agente adequado ao ambiente Farol."""
        if tipo == "QAgent":
            agente = QAgentFarol(
                id=identificador,
                lista_acoes=['UP', 'DOWN', 'LEFT', 'RIGHT'],
                modo='learn'
            )
        elif tipo == "FixedAgent":
            agente = FixedAgent(
                id=identificador,
                politica=policy_farol_inteligente,
                modo='test'
            )
        else:  # RandomAgent
            agente = FixedAgent(
                id=identificador,
                politica=policy_aleatoria,
                modo='test'
            )

        agente.instala(SensorVisao(alcance=1))
        agente.instala(SensorFarol())
        agente.verbose = verbose
        return agente

    def _criar_agente_foraging(self, tipo, identificador, verbose=False):
        """Cria um agente adequado ao ambiente Foraging."""
        if tipo == "QAgent":
            agente = QAgentForaging(
                id=identificador,
                lista_acoes=['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP'],
                modo='learn'
            )
        elif tipo == "FixedAgent":
            agente = FixedAgent(
                id=identificador,
                politica=policy_foraging_inteligente,
                modo='test'
            )
        else:  # RandomAgent
            agente = FixedAgent(
                id=identificador,
                politica=policy_aleatoria,
                modo='test'
            )

        agente.instala(SensorVisao(alcance=2))
        agente.instala(SensorNinho())
        agente.instala(SensorCarregando())
        agente.verbose = verbose
        return agente

    def criar_agente(self, tipo_agente, identificador, problema, verbose=False):
        """Wrapper público que delega para as fábricas específicas."""
        if problema == "Farol":
            return self._criar_agente_farol(tipo_agente, identificador, verbose)
        return self._criar_agente_foraging(tipo_agente, identificador, verbose)

    def _criar_visualizador_para_ambiente(self, ambiente, titulo="Simulacao"):
        """Cria `Visualizador` adaptado a FarolEnv ou ForagingEnv."""
        largura = getattr(ambiente, 'width', getattr(ambiente, 'size', 10))
        altura = getattr(ambiente, 'height', getattr(ambiente, 'size', 10))
        return Visualizador(
            largura,
            altura,
            title=titulo,
            fps=5
        )

    # ==================== EXECUÇÃO GENÉRICA ====================

    def executar_simulacao(self, problema, configuracao_agentes,
                            verbose=True, render=True, tamanho=10):
        """
        Executa simulação configurada via menus (não JSON).
        `configuracao_agentes`: lista de tuplos (tipo, nome).
        """

        # 1\) Criar ambiente e agentes
        ambiente = self._criar_ambiente(problema, tamanho)
        lista_agentes = [
            self.criar_agente(tipo, nome, problema, verbose)
            for tipo, nome in configuracao_agentes
        ]

        # 2\) Construir parametros para o motor no formato esperado
        params_motor = {
            "problem": problema,
            "environment": {},
            "simulation": {
                "episodes": self.parametros_base['episodes'],
                "render_delay": self.parametros_base['render_delay'],
                "verbose": verbose,
                "render": render
            },
            "agents": []  # não é usado neste caminho, pois adicionamos manualmente
        }

        motor = MotorDeSimulacao.cria(params_motor)
        motor.adiciona_ambiente(ambiente)

        for ag in lista_agentes:
            motor.adiciona_agente(ag)

        # 3\) Visualizador
        if render:
            try:
                viz = self._criar_visualizador_para_ambiente(
                    ambiente,
                    titulo=f"{problema}"
                )
                motor.liga_visualizador(viz)
            except Exception as e:
                print(f"⚠️ Não foi possível criar visualizador: {e}")

        # 4\) Logs iniciais
        print(f"\n🎮 INICIANDO SIMULAÇÃO: {problema}")
        print(f"   Agentes: {', '.join([f'{nome} ({tipo})' for tipo, nome in configuracao_agentes])}")
        print(f"   Episódios: {self.parametros_base['episodes']} | Passos máximos: {self.parametros_base['max_steps']}")
        print('=' * 60)

        # 5\) Executar motor
        metricas, extras = motor.executa(render=render, verbose=verbose)

        self.mostrar_resumo(metricas, extras, configuracao_agentes)
        self.salva_csv(metricas, extras, filename=f'metrics_{problema}.csv')
        return metricas, extras

    # ==================== RESUMO E CSV ====================

    def mostrar_resumo(self, metricas, extras, configuracao_agentes):
        print("\n📊 RESUMO FINAL DA SIMULAÇÃO")
        print("=" * 50)
        print(f"🔍 Chaves nas métricas: {list(metricas.keys())}  | extras: {list(extras.keys())}")

        for tipo_agente, nome_agente in configuracao_agentes:
            chave_reward = f'reward_{nome_agente}'
            valores = metricas.get(chave_reward) or []
            if valores:
                media = sum(valores) / len(valores)
                print(f"   {nome_agente} ({tipo_agente}): média reward {media:.2f}  todos: {[round(v, 2) for v in valores]}")
            else:
                print(f"   {nome_agente} ({tipo_agente}): sem dados de reward")

        if metricas.get('steps'):
            media_steps = sum(metricas['steps']) / len(metricas['steps'])
            print(f"   Passos médios por episódio: {media_steps:.1f}")

        if metricas.get('success_rate'):
            media_sucesso = sum(metricas['success_rate']) / len(metricas['success_rate'])
            print(f"   Taxa média de sucesso: {media_sucesso:.2f}")

        if metricas.get('resources_delivered'):
            print(f"   Recursos entregues (por episódio): {metricas['resources_delivered']}")

        if extras:
            print("\n   -- Extras exemplos:")
            for chave, valor in extras.items():
                print(f"     {chave}: {valor[:5]}")

    def salva_csv(self, metricas, extras, filename='metrics.csv'):
        """Guarda as métricas num ficheiro CSV."""
        try:
            if not metricas and not extras:
                print("⚠️ Sem métricas para guardar.")
                return

            colunas = list(metricas.keys()) + list(extras.keys())
            total_linhas = max(
                [len(v) for v in metricas.values()] + [len(v) for v in extras.values()] or [0]
            )

            linhas = []
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

    # ==================== MENUS INTERATIVOS ====================

    def escolher_tipo_agente(self):
        while True:
            print("1. QAgent (learn)  2. FixedAgent  3. RandomAgent")
            opcao = input("Escolha (1-3): ").strip()
            if opcao == '1':
                return 'QAgent'
            if opcao == '2':
                return 'FixedAgent'
            if opcao == '3':
                return 'RandomAgent'
            print("Opção inválida")

    def menu_farol(self):
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

    def menu_principal(self):
        while True:
            print('\n' + '=' * 50)
            print('SIMULADOR INTERATIVO')
            print('1. Farol  2. Foraging  3. Sair')
            escolha = input('Escolha (1-3): ').strip()

            if escolha == '1':
                self.menu_farol()
            elif escolha == '2':
                self.menu_foraging()
            elif escolha == '3':
                break
            else:
                print('Opção inválida')

    # ==================== MODO JSON ====================

    def executarJson(self, arquivo_json):
        try:
            motor = MotorDeSimulacao.cria(arquivo_json)

            render = motor.params.get("simulation", {}).get("render", False)
            verbose = motor.params.get("simulation", {}).get("verbose", False)

            if render:
                try:
                    ambiente = motor.ambiente
                    viz = self._criar_visualizador_para_ambiente(
                        ambiente,
                        titulo=motor.params.get("simulation", {}).get("title", "Simulacao")
                    )
                    motor.liga_visualizador(viz)
                except Exception as e:
                    print(f"⚠️ Não foi possível criar visualizador: {e}")
                    render = False

            metricas, extras = motor.executa(render=render, verbose=verbose)

            if metricas:
                self.salva_csv(metricas, extras, filename='metrics_from_json.csv')

            print("✅ Simulação via JSON concluída")
            return metricas, extras
        except Exception as erro:
            print(f"⚠️ Erro ao executar simulação JSON: {erro}")
            return None, None


def main():
    simulador = SimuladorInterativo()
    try:
        # modo JSON (para testar visualizador com farol.json)
        simulador.executarJson('foraging.json')

        # ou modo menus:
        # simulador.menu_principal()
    except KeyboardInterrupt:
        print('\nInterrompido pelo utilizador')


if __name__ == '__main__':
    main()
