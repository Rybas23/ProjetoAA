import random
from simulador.engine import MotorDeSimulacao
from simulador.farol_env import FarolEnv
from simulador.foraging_env import ForagingEnv
from simulador.q_agent import QAgent
from simulador.fixed_agent import FixedAgent
from simulador.sensor import SensorVisao, SensorFarol, SensorNinho

# ----------------------------
# Políticas inteligentes FixedAgent
# ----------------------------
def policy_farol_inteligente(obs):
    """Política inteligente para o problema do Farol"""
    if 'direcao_farol' in obs:
        direcao = obs['direcao_farol']
        # Mover na direção do farol
        if direcao == 'N': return 'UP'
        if direcao == 'S': return 'DOWN'
        if direcao == 'E': return 'RIGHT'
        if direcao == 'W': return 'LEFT'
        if direcao == 'NONE': return 'STAY'

    # Se não tem informação do farol, movimento aleatório
    return random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])


def policy_foraging_inteligente(obs):
    """Política inteligente para o problema do Foraging"""
    x, y = obs['pos']
    nx, ny = obs['nest']

    # Se está carregando, ir para o ninho
    if obs['carrying'] == 1:
        if (x, y) == (nx, ny):
            return 'DROP'
        if nx > x: return 'RIGHT'
        if nx < x: return 'LEFT'
        if ny > y: return 'DOWN'
        if ny < y: return 'UP'

    # Se há recurso na própria célula, pegar
    if 'visao' in obs and obs['visao'].get('C', 0) > 0:
        return 'PICK'

    # Verificar vizinhos por recursos
    if 'visao' in obs:
        for dir_name, amount in obs['visao'].items():
            if dir_name == 'C':
                continue
            if amount > 0:
                return {'L': 'LEFT', 'R': 'RIGHT', 'U': 'UP', 'D': 'DOWN'}[dir_name]

    # Mover aleatoriamente se não encontrou recurso
    return random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])


def policy_aleatoria(obs):
    """Política completamente aleatória"""
    return random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'])


# ----------------------------
# Classe do Menu Interativo
# ----------------------------
class SimuladorInterativo:
    def __init__(self):
        self.params_base = {
            'episodes': 5,
            'max_steps': 100,
            'render_delay': 0.3
        }

    def criar_agente(self, tipo_agente, id, problema, verbose=False):
        """Cria um agente com os sensores apropriados"""
        if problema == "Farol":
            actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
            if tipo_agente == "QAgent":
                agente = QAgent(id=id, actions=actions, modo='learn')
            elif tipo_agente == "FixedAgent":
                agente = FixedAgent(id=id, policy=policy_farol_inteligente, modo='test')
            else:  # RandomAgent
                agente = FixedAgent(id=id, policy=policy_aleatoria, modo='test')

            # Instalar sensores para Farol
            agente.instala(SensorVisao(alcance=1))
            agente.instala(SensorFarol())

        else:  # Foraging
            actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP']
            if tipo_agente == "QAgent":
                agente = QAgent(id=id, actions=actions, modo='learn')
            elif tipo_agente == "FixedAgent":
                agente = FixedAgent(id=id, policy=policy_foraging_inteligente, modo='test')
            else:  # RandomAgent
                agente = FixedAgent(id=id, policy=policy_aleatoria, modo='test')

            # Instalar sensores para Foraging
            agente.instala(SensorVisao(alcance=2))
            agente.instala(SensorNinho())

        agente.verbose = verbose
        return agente

    def criar_ambiente(self, problema, tamanho=10):
        if problema == "Farol":
            return FarolEnv(size=tamanho,
                            farol_fixo=(tamanho // 2, tamanho // 2),
                            max_steps=self.params_base['max_steps'])
        else:  # Foraging
            return ForagingEnv(width=tamanho, height=tamanho,
                               n_resources=tamanho * 2, nest=(0, 0),
                               max_steps=self.params_base['max_steps'])

    def executar_simulacao(self, problema, config_agentes, verbose=True, render=True):
        """Executa uma simulação com a configuração especificada"""
        # Criar ambiente
        env = self.criar_ambiente(problema)

        # Criar agentes
        agentes = []
        for i, (tipo_agente, nome_personalizado) in enumerate(config_agentes):
            agente = self.criar_agente(tipo_agente, nome_personalizado, problema, verbose)
            agentes.append(agente)

        # Configurar parâmetros
        params = self.params_base.copy()

        # Executar simulação
        motor = MotorDeSimulacao(params)
        motor.adiciona_ambiente(env)

        # Adicionar agentes
        for ag in agentes:
            motor.adiciona_agente(ag)

        print(f"\n🎮 INICIANDO SIMULAÇÃO: {problema}")
        print(f"   Agentes: {', '.join([f'{nome} ({tipo})' for tipo, nome in config_agentes])}")
        print(f"   Episódios: {params['episodes']} | Passos máximos: {params['max_steps']}")
        print("=" * 60)

        metrics = motor.executa(render=render, verbose=verbose)

        # Mostrar resumo final
        self.mostrar_resumo(metrics, config_agentes)

        return metrics

    def mostrar_resumo(self, metrics, config_agentes):
        """Mostra um resumo das métricas finais"""
        print("\n📊 RESUMO FINAL DA SIMULAÇÃO")
        print("=" * 50)

        # DEBUG: Mostrar todas as chaves disponíveis
        print(f"🔍 Chaves disponíveis nas métricas: {list(metrics.keys())}")

        for (tipo_agente, nome_agente) in config_agentes:
            reward_key = f'reward_{nome_agente}'
            if reward_key in metrics:
                recompensas = metrics[reward_key]
                media = sum(recompensas) / len(recompensas) if recompensas else 0
                max_reward = max(recompensas) if recompensas else 0
                print(f"   {nome_agente} ({tipo_agente}):")
                print(f"     📈 Recompensa média: {media:.2f}")
                print(f"     🏆 Melhor episódio: {max_reward:.2f}")
                print(f"     📋 Todos os valores: {[f'{r:.2f}' for r in recompensas]}")
            else:
                print(f"   {nome_agente} ({tipo_agente}):")
                print(f"     ⚠️  Sem dados de recompensa (chave: {reward_key})")

        if 'steps' in metrics and metrics['steps']:
            media_passos = sum(metrics['steps']) / len(metrics['steps'])
            print(f"\n   ⏱️  Passos médios por episódio: {media_passos:.1f}")
        else:
            print(f"\n   ⏱️  Sem dados de passos")

    def menu_principal(self):
        """Menu principal interativo"""
        while True:
            print("\n" + "=" * 50)
            print("🤖 SIMULADOR INTERATIVO DE SISTEMAS MULTI-AGENTE")
            print("=" * 50)
            print("1. 🎯 Problema Farol")
            print("2. 🍎 Problema Foraging")
            print("3. 🚀 Executar Exemplo Rápido")
            print("4. ❌ Sair")

            opcao = input("\nEscolha uma opção (1-4): ").strip()

            if opcao == "1":
                self.menu_farol()
            elif opcao == "2":
                self.menu_foraging()
            elif opcao == "3":
                self.executar_exemplo_rapido()
            elif opcao == "4":
                print("👋 A sair do simulador...")
                break
            else:
                print("❌ Opção inválida! Tente novamente.")

    def menu_farol(self):
        """Menu de configuração para o problema Farol"""
        print("\n🎯 CONFIGURAR PROBLEMA FAROL")

        try:
            # Número de agentes
            n_agents = int(input("Quantos agentes? (1-5): "))
            n_agents = max(1, min(5, n_agents))

            # Configurar cada agente
            config_agentes = []
            for i in range(n_agents):
                print(f"\nConfigurar Agente {i + 1}:")
                tipo = self.escolher_tipo_agente()
                nome = input(f"Nome para o agente (padrão: agente{i + 1}): ").strip()
                nome = nome or f"agente{i + 1}"
                config_agentes.append((tipo, nome))

            # Opções de execução
            verbose = input("Mostrar logs detalhados? (s/N): ").lower().startswith('s')
            render = input("Mostrar visualização gráfica? (S/n): ").lower() != 'n'

            # Executar
            self.executar_simulacao("Farol", config_agentes, verbose, render)

        except ValueError:
            print("❌ Entrada inválida! Use números inteiros.")
        except Exception as e:
            print(f"❌ Erro: {e}")

    def menu_foraging(self):
        """Menu de configuração para o problema Foraging"""
        print("\n🍎 CONFIGURAR PROBLEMA FORAGING")

        try:
            # Número de agentes
            n_agents = int(input("Quantos agentes? (1-5): "))
            n_agents = max(1, min(5, n_agents))

            # Configurar cada agente
            config_agentes = []
            for i in range(n_agents):
                print(f"\nConfigurar Agente {i + 1}:")
                tipo = self.escolher_tipo_agente()
                nome = input(f"Nome para o agente (padrão: forager{i + 1}): ").strip()
                nome = nome or f"forager{i + 1}"
                config_agentes.append((tipo, nome))

            # Opções de execução
            verbose = input("Mostrar logs detalhados? (s/N): ").lower().startswith('s')
            render = input("Mostrar visualização gráfica? (S/n): ").lower() != 'n'

            # Executar
            self.executar_simulacao("Foraging", config_agentes, verbose, render)

        except ValueError:
            print("❌ Entrada inválida! Use números inteiros.")
        except Exception as e:
            print(f"❌ Erro: {e}")

    def escolher_tipo_agente(self):
        """Menu para escolher o tipo de agente"""
        while True:
            print("   Tipos de agente disponíveis:")
            print("   1. QAgent (aprendizagem por Q-learning)")
            print("   2. FixedAgent (política inteligente)")
            print("   3. RandomAgent (movimentos aleatórios)")

            opcao = input("   Escolha o tipo (1-3): ").strip()

            if opcao == "1":
                return "QAgent"
            elif opcao == "2":
                return "FixedAgent"
            elif opcao == "3":
                return "RandomAgent"
            else:
                print("   ❌ Tipo inválido! Tente novamente.")

    def executar_exemplo_rapido(self):
        """Executa um exemplo rápido para demonstração"""
        print("\n🚀 EXECUTANDO EXEMPLO RÁPIDO...")

        # Exemplo: Farol com 2 agentes (1 Q-learning, 1 inteligente)
        config_farol = [
            ("QAgent", "agente0"),
            ("FixedAgent", "agente1")
        ]

        print("📝 Configuração:")
        print("   - Problema: Farol")
        print("   - Agentes: agente0 (Q-learning), agente1 (política fixa)")
        print("   - Episódios: 3 (para demonstração rápida)")

        input("\nPressione Enter para iniciar...")

        self.executar_simulacao("Farol", config_farol, verbose=True, render=True)


# ----------------------------
# Função principal
# ----------------------------
def main():
    """Função principal"""
    try:
        simulador = SimuladorInterativo()
        simulador.menu_principal()
    except KeyboardInterrupt:
        print("\n\n🛑 Simulação interrompida pelo utilizador")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")


if __name__ == "__main__":
    main()