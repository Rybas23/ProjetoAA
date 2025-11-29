import random, time, csv
from engine import MotorDeSimulacao
from ambiente_farol import FarolEnv
from ambiente_foraging import ForagingEnv

# IMPORTAÇÃO CERTA DOS AGENTES
from agentes import QAgentFarol, QAgentForaging, FixedAgent

from sensors import SensorVisao, SensorFarol, SensorNinho
from policies import policy_farol_inteligente, policy_foraging_inteligente, policy_aleatoria
from visualizador import Visualizador


class SimuladorInterativo:
    def __init__(self):
        self.params_base = {
            'episodes': 5,
            'max_steps': 100,
            'render_delay': 0.05
        }

    # -------------------------------------------------------------
    # CRIAÇÃO CORRETA DE AGENTE PARA CADA AMBIENTE
    # -------------------------------------------------------------
    def criar_agente(self, tipo_agente, id, problema, verbose=False):

        # -------------------- FAROL --------------------
        if problema == "Farol":
            if tipo_agente == "QAgent":
                agente = QAgentFarol(id=id, actions=['UP','DOWN','LEFT','RIGHT','STAY'], modo='learn')
            elif tipo_agente == "FixedAgent":
                agente = FixedAgent(id=id, policy=policy_farol_inteligente, modo='test')
            else:
                agente = FixedAgent(id=id, policy=policy_aleatoria, modo='test')

            agente.instala(SensorVisao(alcance=1))
            agente.instala(SensorFarol())

        # -------------------- FORAGING --------------------
        else:
            if tipo_agente == "QAgent":
                agente = QAgentForaging(id=id, actions=['UP','DOWN','LEFT','RIGHT','PICK','DROP','STAY'], modo='learn')
            elif tipo_agente == "FixedAgent":
                agente = FixedAgent(id=id, policy=policy_foraging_inteligente, modo='test')
            else:
                agente = FixedAgent(id=id, policy=policy_aleatoria, modo='test')

            agente.instala(SensorVisao(alcance=2))
            agente.instala(SensorNinho())

        agente.verbose = verbose
        return agente

    # -------------------------------------------------------------
    def criar_ambiente(self, problema, tamanho=10):
        if problema == "Farol":
            return FarolEnv(size=tamanho,
                            farol_fixo=(tamanho//2, tamanho//2),
                            max_steps=self.params_base['max_steps'])
        else:
            return ForagingEnv(width=tamanho,
                               height=tamanho,
                               n_resources=tamanho*2,
                               nest=(0,0),
                               max_steps=self.params_base['max_steps'])

    # -------------------------------------------------------------
    def executar_simulacao(self, problema, config_agentes, verbose=True, render=True):
        env = self.criar_ambiente(problema)
        agentes = []

        for (tipo_agente, nome_personalizado) in config_agentes:
            agente = self.criar_agente(tipo_agente, nome_personalizado, problema, verbose)
            agentes.append(agente)

        params = self.params_base.copy()
        motor = MotorDeSimulacao.cria(params)
        motor.adiciona_ambiente(env)

        for ag in agentes:
            motor.adiciona_agente(ag)

        # Visualização
        if render:
            try:
                if hasattr(env, 'size'):
                    viz = Visualizador(env.size, env.size, title=problema, fps=5)
                else:
                    viz = Visualizador(env.width, env.height, title=problema, fps=5)
                motor.liga_visualizador(viz)
            except Exception:
                viz = None

        print(f"\n🎮 INICIANDO SIMULAÇÃO: {problema}")
        print(f"   Agentes: {', '.join([f'{nome} ({tipo})' for tipo,nome in config_agentes])}")
        print(f"   Episódios: {params['episodes']} | Passos máximos: {params['max_steps']}")
        print('='*60)

        metrics, extras = motor.executa(render=render, verbose=verbose)

        self.mostrar_resumo(metrics, extras, config_agentes)
        self.salva_csv(metrics, extras, filename=f'metrics_{problema}.csv')

        return metrics, extras

    # -------------------------------------------------------------
    # RESTO DO CÓDIGO É IGUAL
    # -------------------------------------------------------------

    def mostrar_resumo(self, metrics, extras, config_agentes):
        print("\n📊 RESUMO FINAL DA SIMULAÇÃO")
        print("="*50)
        print(f"🔍 Chaves nas métricas: {list(metrics.keys())}  | extras: {list(extras.keys())}")

        for (tipo_agente, nome_agente) in config_agentes:
            reward_key = f'reward_{nome_agente}'
            vals = metrics.get(reward_key) or []
            if vals:
                media = sum(vals)/len(vals)
                print(f"   {nome_agente} ({tipo_agente}): média reward {media:.2f}  todos: {[round(v,2) for v in vals]}")
            else:
                print(f"   {nome_agente} ({tipo_agente}): sem dados de reward")

        if 'steps' in metrics and metrics['steps']:
            print(f"   Passos médios por episódio: {sum(metrics['steps'])/len(metrics['steps']):.1f}")

        if 'success_rate' in metrics and metrics['success_rate']:
            print(f"   Taxa média de sucesso: {sum(metrics['success_rate'])/len(metrics['success_rate']):.2f}")

        if 'resources_delivered' in metrics and metrics['resources_delivered']:
            print(f"   Recursos entregues (últimos episódios): {metrics['resources_delivered']}")

        if extras:
            print("\n   -- Extras exemplos:")
            for k,v in extras.items():
                print(f"     {k}: {v[:5]}")

    def salva_csv(self, metrics, extras, filename='metrics.csv'):
        try:
            rows = []
            n = max(len(v) for v in metrics.values()) if metrics else 0
            keys = list(metrics.keys()) + list(extras.keys())

            for i in range(n):
                row = {}
                for k in metrics:
                    row[k] = metrics[k][i] if i < len(metrics[k]) else ''
                for k in extras:
                    row[k] = extras[k][i] if i < len(extras[k]) else ''
                rows.append(row)

            with open(filename, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow(r)

            print(f"✅ Métricas guardadas em {filename}")

        except Exception as e:
            print(f"⚠️ Erro ao salvar CSV: {e}")

    def escolher_tipo_agente(self):
        while True:
            print("1. QAgent (learn)  2. FixedAgent  3. RandomAgent")
            opc = input("Escolha (1-3): ").strip()
            if opc=='1': return 'QAgent'
            if opc=='2': return 'FixedAgent'
            if opc=='3': return 'RandomAgent'
            print("Opção inválida")

    def menu_farol(self):
        print("\n🎯 CONFIGURAR FAROL")
        try:
            n_agents = int(input("Quantos agentes? (1-5): "))
            n_agents = max(1, min(5, n_agents))
            config = []
            for i in range(n_agents):
                print(f"Configurar agente {i+1}:")
                tipo = self.escolher_tipo_agente()
                nome = input(f"Nome (default agente{i+1}): ").strip() or f"agente{i+1}"
                config.append((tipo, nome))
            verbose = input("Mostrar logs? (s/N): ").lower().startswith('s')
            render = input("Mostrar visualizacao? (S/n): ").lower() != 'n'
            self.executar_simulacao('Farol', config, verbose=verbose, render=render)
        except Exception as e:
            print(f"Erro: {e}")

    def menu_foraging(self):
        print("\n🍎 CONFIGURAR FORAGING")
        try:
            n_agents = int(input("Quantos agentes? (1-5): "))
            n_agents = max(1, min(5, n_agents))
            config = []
            for i in range(n_agents):
                tipo = self.escolher_tipo_agente()
                nome = input(f"Nome (default forager{i+1}): ").strip() or f"forager{i+1}"
                config.append((tipo, nome))
            verbose = input("Mostrar logs? (s/N): ").lower().startswith('s')
            render = input("Mostrar visualizacao? (S/n): ").lower() != 'n'
            self.executar_simulacao('Foraging', config, verbose=verbose, render=render)
        except Exception as e:
            print(f"Erro: {e}")

    def executar_exemplo_rapido(self):
        print("\n🚀 EXEMPLO RÁPIDO: Farol 2 agentes (Q + Fixed)")
        config = [('QAgent','q1'), ('FixedAgent','fix1')]
        input("Pressione Enter para iniciar...")
        self.executar_simulacao('Farol', config, verbose=True, render=True)

    def menu_principal(self):
        while True:
            print('\n' + '='*50)
            print('SIMULADOR INTERATIVO')
            print('1. Farol  2. Foraging  3. Exemplo rapido  4. Sair')
            opc = input('Escolha (1-4): ').strip()
            if opc=='1': self.menu_farol()
            elif opc=='2': self.menu_foraging()
            elif opc=='3': self.executar_exemplo_rapido()
            elif opc=='4': break
            else: print('Opcao invalida')


def main():
    s = SimuladorInterativo()
    try:
        s.menu_principal()
    except KeyboardInterrupt:
        print('\nInterrompido pelo utilizador')


if __name__=='__main__':
    main()
