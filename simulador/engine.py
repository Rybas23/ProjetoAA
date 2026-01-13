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
        self.max_steps = parametros_execucao.get("max_steps", 500)  # default 500
        self.metrics = defaultdict(list)   # métricas registadas
        self.visualizador = None           # objeto visualizador opcional
        self.tracker = None                # rastreador de métricas detalhadas
        # Novo: mapa de spawns iniciais por id de agente
        self.agent_spawns = {}

    # Inicializar lendo ficheiro JSON ou usando dict
    @classmethod
    def cria(cls, params):
        if isinstance(params, str):
            import json
            with open(params, "r") as f:
                params = json.load(f)

        motor = cls(params)

        # ---------- CRIAR AMBIENTE ----------
        problem = params.get("problem")
        env_cfg = params.get("environment", {})

        if problem == "Farol":
            from ambiente_farol import FarolEnv

            ambiente = FarolEnv(
                size=env_cfg.get("size", 10),
                farol_fixo=tuple(env_cfg.get("farol_fixo", None)),
                paredes=[tuple(p) for p in env_cfg.get("walls", [])],
            )

        elif problem == "Foraging":
            from ambiente_foraging import ForagingEnv

            ambiente = ForagingEnv(
                width=env_cfg.get("width", 10),
                height=env_cfg.get("height", 10),
                ninho=tuple(env_cfg.get("ninho", (0, 0))),
                paredes=[tuple(p) for p in env_cfg.get("walls", [])],
                recursos=[tuple(r) for r in env_cfg.get("resources", [])],
            )
        else:
            raise ValueError("Problema desconhecido no JSON")

        motor.adiciona_ambiente(ambiente)

        # ---------- CRIAR AGENTES ----------
        from agentes import QAgentFarol, QAgentForaging, FixedAgent, GAAgentForaging, GAAgentFarol
        from sensors import SensorVisao, SensorFarol, SensorNinho, SensorCarregando, SensorRecursoMaisProximo
        from policies import (
            policy_farol_inteligente,
            policy_foraging_inteligente,
            policy_aleatoria,
        )

        for ag_cfg in params.get("agents", []):
            tipo = ag_cfg["type"]
            ag_id = ag_cfg["id"]
            modo = ag_cfg.get("mode", "test")

            if tipo == "QAgentFarol":
                agente = QAgentFarol.cria(None)
                agente.id = ag_id
                agente.modo = modo
                agente.instala(SensorVisao(alcance=1))
                agente.instala(SensorFarol())

            elif tipo == "QAgentForaging":
                agente = QAgentForaging.cria(None)
                agente.id = ag_id
                agente.modo = modo
                agente.instala(SensorVisao(alcance=2))
                agente.instala(SensorNinho())
                agente.instala(SensorCarregando())
                agente.instala(SensorRecursoMaisProximo())

            elif tipo == "GAAgentForaging":
                agente = GAAgentForaging.cria(ag_cfg)
                agente.id = ag_id
                agente.modo = modo
                agente.instala(SensorVisao(alcance=2))
                agente.instala(SensorNinho())
                agente.instala(SensorCarregando())
                agente.instala(SensorRecursoMaisProximo())

            elif tipo == "GAAgentFarol":
                agente = GAAgentFarol.cria(ag_cfg)
                agente.id = ag_id
                agente.modo = modo
                agente.instala(SensorVisao(alcance=1))
                agente.instala(SensorFarol())

            elif tipo == "FixedAgent":
                policy_name = ag_cfg.get("policy", "random")

                policy_map = {
                    "farol_inteligente": policy_farol_inteligente,
                    "foraging_inteligente": policy_foraging_inteligente,
                    "random": policy_aleatoria,
                }

                agente = FixedAgent(
                    id=ag_id,
                    politica=policy_map[policy_name],
                    modo="test",
                )

                # Instalar sensores
                if problem == "Farol":
                    agente.instala(SensorFarol())
                elif problem == "Foraging":
                    agente.instala(SensorVisao(alcance=2))
                    agente.instala(SensorNinho())
                    agente.instala(SensorCarregando())

            else:
                raise ValueError(f"Tipo de agente desconhecido: {tipo}")

            motor.adiciona_agente(agente)

            # Spawn definido no JSON
            if "spawn" in ag_cfg:
                motor.agent_spawns[ag_id] = tuple(ag_cfg["spawn"])

        return motor

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
    def executa(self, render=False, logs=False):
        if self.ambiente is None:
            raise RuntimeError("Ambiente não definido")

        # Registar agentes no ambiente (se o ambiente tiver esse metodo)
        if hasattr(self.ambiente, "registar_agentes"):
            self.ambiente.registar_agentes(self.agentes)

        self.cria_tracker()

        # Ativar modo logs nos agentes, se necessário
        for ag in self.agentes:
            ag.logs = logs

        numero_episodios = self.params.get("episodes", 10)

        # LOOP PRINCIPAL DE EPISÓDIOS #
        for ep in range(numero_episodios):

            if logs:
                print(f"\n🎬 INICIANDO EPISÓDIO {ep + 1}/{numero_episodios}")
                print("=" * 50)

            # Passar spawns ao reset se o ambiente suportar
            if hasattr(self.ambiente, "reset"):
                try:
                    estado_inicial = self.ambiente.reset(self.agent_spawns)
                except TypeError:
                    # Ambiente ainda não suporta spawns explícitos
                    estado_inicial = self.ambiente.reset()
            else:
                raise RuntimeError("Ambiente sem método reset")

            # Reset dos agentes (política, memória, etc.)
            for ag in self.agentes:
                ag.reset(ep)

            passo_atual = 0
            recompensa_por_agente = {ag.id: 0 for ag in self.agentes}
            episodio_terminado = False

            # LOOP INTERNO DE PASSOS NO EPISÓDIO
            while passo_atual < self.max_steps and not episodio_terminado:

                # 1. Cada agente recebe observação do estado ATUAL
                for ag in self.agentes:
                    obs = self.ambiente.observacaoPara(ag)
                    ag.observacao(obs)

                # 2. Cada agente decide uma ação baseado no estado atual
                lista_acoes = []
                for ag in self.agentes:
                    acao_escolhida = ag.age()
                    lista_acoes.append((ag, acao_escolhida))
                    if logs:
                        print(f"🎯 [{ag.id}] -> {acao_escolhida}")

                # 3. Ambiente executa ações (transição de estado)
                recompensas_passo = {}
                for ag, acao in lista_acoes:
                    recompensa, terminou = self.ambiente.agir(acao, ag)
                    recompensas_passo[ag.id] = recompensa
                    recompensa_por_agente[ag.id] += recompensa

                    if logs and recompensa != 0:
                        print(f"   [{ag.id}] reward {recompensa:+.3f}")

                # 4. Atualização interna do ambiente
                self.ambiente.atualizacao()

                # 5. CRÍTICO PARA Q-LEARNING: Dar nova observação (s') ANTES de avaliar
                #    Agora última_observacao terá o NOVO estado (s')
                for ag in self.agentes:
                    obs_nova = self.ambiente.observacaoPara(ag)
                    ag.observacao(obs_nova)

                # 6. Q-Learning update: agora agente tem s (guardado), a, r, e s' (última_observacao)
                for ag, acao in lista_acoes:
                    ag.avaliacaoEstadoAtual(recompensas_passo[ag.id])

                # 7. Verificar conclusão do episódio
                episodio_terminado = self.ambiente.is_episode_done()

                # 5. Atualização interna do ambiente
                self.ambiente.atualizacao()
                passo_atual += 1

                # 6. Renderização (se ativo)
                if render:
                    if hasattr(self.ambiente, "render") and callable(self.ambiente.render):
                        self.ambiente.render()
                    elif self.visualizador:
                        try:
                            self.visualizador.draw(self.ambiente)
                        except Exception:
                            pass

                    time.sleep(self.params.get("render_delay", 0.0))

            # Final do episódio — guardar métricas
            for ag in self.agentes:
                self.metrics["reward_" + ag.id].append(recompensa_por_agente[ag.id])

            self.metrics["steps"].append(passo_atual)

            # Métrica de sucesso → ambiente Farol
            if hasattr(self.ambiente, "done_agents"):
                sucesso = len(getattr(self.ambiente, "done_agents", set())) / max(
                    1, len(self.agentes)
                )
                self.metrics["success_rate"].append(sucesso)

            # Métrica de recursos → Foraging
            if hasattr(self.ambiente, "total_delivered"):
                self.metrics["resources_delivered"].append(
                    getattr(self.ambiente, "total_delivered", 0)
                )

            # Registo especializado para o tracker
            try:
                if hasattr(self.ambiente, "farol"):
                    self.tracker.regista_farol(
                        self.ambiente, recompensa_por_agente, steps=passo_atual
                    )
                if hasattr(self.ambiente, "resources"):
                    self.tracker.regista_foraging(
                        self.ambiente, recompensa_por_agente, steps=passo_atual
                    )
            except Exception:
                pass

            if logs:
                print(
                    f"🏁 EP {ep+1} done steps={passo_atual} rewards={recompensa_por_agente}"
                )

        # Salvar heatmaps dos agentes no final
        for ag in self.agentes:
            heatmap_filename = f"heatmap_{ag.id}.csv"
            try:
                ag.save_heatmap(heatmap_filename)
                print(f"📍 Heatmap salvo: {heatmap_filename}")
            except Exception as e:
                print(f"⚠️ Erro ao salvar heatmap de {ag.id}: {e}")

        # Dados extra gerados pelo tracker
        extras = dict(self.tracker.data) if self.tracker else {}
        return dict(self.metrics), extras
