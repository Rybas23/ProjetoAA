import time
from collections import defaultdict
from metrics import MetricsTracker

# Motor central da simulação:
# - controla episódios
# - recolhe métricas
# - dá observações aos agentes
# - executa ações e recebe recompensas
class MotorDeSimulacao:
    def __init__(self, parametros_execucao):
        self.params = parametros_execucao  # dicionário de parâmetros
        self.ambiente = None               # ambiente (Farol ou Foraging)
        self.agentes = []                  # lista de agentes ativos
        self.max_steps = parametros_execucao.get('max_steps', 500) # buscar input, default 500
        self.metrics = defaultdict(list)   # métricas registadas
        self.visualizador = None           # objeto visualizador opcional
        self.tracker = None                # rastreador de métricas detalhadas

    # Inicializar lendo ficheiro JSON
    @classmethod
    def cria(cls, params):
        if isinstance(params, str):
            import json
            with open(params, 'r') as f:
                params = json.load(f)
        return cls(params)

    # Liga o ambiente à simulação
    def adiciona_ambiente(self, ambiente):
        self.ambiente = ambiente

    # Adiciona agente e instala o ambiente dentro dele
    def adiciona_agente(self, agente):
        self.agentes.append(agente)
        agente.instala_ambiente(self.ambiente)

    # Devolve lista de agentes
    def listaAgentes(self):
        return list(self.agentes)

    # Conecta visualizador
    def liga_visualizador(self, viz):
        self.visualizador = viz

    # Cria tracker caso ainda não exista
    def cria_tracker(self):
        if self.tracker is None:
            self.tracker = MetricsTracker(self.agentes)

    # EXECUÇÃO DA SIMULAÇÃO
    def executa(self, render=False, verbose=False):
        if self.ambiente is None:
            raise RuntimeError('Ambiente não definido')

        # Registar agentes no ambiente (se o ambiente tiver esse metodo)
        if hasattr(self.ambiente, 'registar_agentes'):
            self.ambiente.registar_agentes(self.agentes)

        self.cria_tracker()

        # Ativar modo verbose nos agentes, se necessário
        for ag in self.agentes:
            ag.verbose = verbose

        numero_episodios = self.params.get('episodes', 10)

        # LOOP PRINCIPAL DE EPISÓDIOS #
        for ep in range(numero_episodios):

            if verbose:
                print(f"\n🎬 INICIANDO EPISÓDIO {ep + 1}/{numero_episodios}")
                print('='*50)

            estado_inicial = self.ambiente.reset()

            # Reset dos agentes (política, memória, etc.)
            for ag in self.agentes:
                ag.reset(ep)

            passo_atual = 0
            recompensa_por_agente = {ag.id: 0 for ag in self.agentes}
            episodio_terminado = False

            # LOOP INTERNO DE PASSOS NO EPISÓDIO
            while passo_atual < self.max_steps and not episodio_terminado:

                # 1. Cada agente recebe observação do ambiente
                for ag in self.agentes:
                    obs = self.ambiente.observacaoPara(ag)
                    ag.observacao(obs)

                # 2. Cada agente decide uma ação
                lista_acoes = []
                for ag in self.agentes:
                    acao_escolhida = ag.age()
                    lista_acoes.append((ag, acao_escolhida))
                    if verbose:
                        print(f"🎯 [{ag.id}] -> {acao_escolhida}")

                # 3. Ambiente executa ações e retorna recompensas
                for ag, acao in lista_acoes:
                    recompensa, terminou = self.ambiente.agir(acao, ag)
                    ag.avaliacaoEstadoAtual(recompensa)
                    recompensa_por_agente[ag.id] += recompensa

                    if verbose and recompensa != 0:
                        print(f"   [{ag.id}] reward {recompensa:+.3f}")

                # 4. Verificar conclusão do episódio
                episodio_terminado = self.ambiente.is_episode_done()

                # 5. Atualização interna do ambiente
                self.ambiente.atualizacao()
                passo_atual += 1

                # 6. Renderização (se ativo)
                if render:
                    if hasattr(self.ambiente, 'render') and callable(self.ambiente.render):
                        self.ambiente.render()
                    elif self.visualizador:
                        try:
                            self.visualizador.draw(self.ambiente)
                        except Exception:
                            pass

                    time.sleep(self.params.get('render_delay', 0.0))

            # Final do episódio — guardar métricas
            for ag in self.agentes:
                self.metrics['reward_' + ag.id].append(recompensa_por_agente[ag.id])

            self.metrics['steps'].append(passo_atual)

            # Métrica de sucesso → ambiente Farol
            if hasattr(self.ambiente, 'done_agents'):
                sucesso = len(getattr(self.ambiente, 'done_agents', set())) / max(1, len(self.agentes))
                self.metrics['success_rate'].append(sucesso)

            # Métrica de recursos → Foraging
            if hasattr(self.ambiente, 'total_delivered'):
                self.metrics['resources_delivered'].append(getattr(self.ambiente, 'total_delivered', 0))

            # Registo especializado para o tracker
            try:
                if hasattr(self.ambiente, 'farol'):
                    self.tracker.regista_farol(self.ambiente, recompensa_por_agente, steps=passo_atual)
                if hasattr(self.ambiente, 'resources'):
                    self.tracker.regista_foraging(self.ambiente, recompensa_por_agente, steps=passo_atual)
            except Exception:
                pass

            if verbose:
                print(f"🏁 EP {ep+1} done steps={passo_atual} rewards={recompensa_por_agente}")

        # Dados extra gerados pelo tracker
        extras = dict(self.tracker.data) if self.tracker else {}
        return dict(self.metrics), extras
