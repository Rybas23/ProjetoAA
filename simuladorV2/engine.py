import time
from collections import defaultdict
from metrics import MetricsTracker

class MotorDeSimulacao:
    def __init__(self, params):
        self.params = params
        self.ambiente = None
        self.agentes = []
        self.max_steps = params.get('max_steps', 500)
        self.metrics = defaultdict(list)
        self.visualizador = None
        self.tracker = None

    @classmethod
    def cria(cls, params):
        if isinstance(params, str):
            import json
            with open(params, 'r') as f:
                params = json.load(f)
        return cls(params)

    def adiciona_ambiente(self, ambiente):
        self.ambiente = ambiente

    def adiciona_agente(self, agente):
        self.agentes.append(agente)
        agente.instala_ambiente(self.ambiente)

    def listaAgentes(self):
        return list(self.agentes)

    def liga_visualizador(self, viz):
        self.visualizador = viz

    def cria_tracker(self):
        if self.tracker is None:
            self.tracker = MetricsTracker(self.agentes)

    def executa(self, render=False, verbose=False):
        if self.ambiente is None:
            raise RuntimeError('Ambiente não definido')

        if hasattr(self.ambiente, 'registar_agentes'):
            self.ambiente.registar_agentes(self.agentes)

        self.cria_tracker()

        for ag in self.agentes:
            ag.verbose = verbose

        num_episodes = self.params.get('episodes', 10)

        for ep in range(num_episodes):
            if verbose:
                print(f"\n🎬 INICIANDO EPISÓDIO {ep + 1}/{num_episodes}")
                print('='*50)

            state = self.ambiente.reset()
            for ag in self.agentes:
                ag.reset(ep)

            step = 0
            ep_reward = {ag.id: 0 for ag in self.agentes}
            done = False

            while step < self.max_steps and not done:
                for ag in self.agentes:
                    obs = self.ambiente.observacaoPara(ag)
                    ag.observacao(obs)

                acts = []
                for ag in self.agentes:
                    acao = ag.age()
                    acts.append((ag, acao))
                    if verbose:
                        print(f"🎯 [{ag.id}] -> {acao}")

                for ag, act in acts:
                    reward, terminated = self.ambiente.agir(act, ag)
                    ag.avaliacaoEstadoAtual(reward)
                    ep_reward[ag.id] += reward
                    if verbose and reward != 0:
                        print(f"   [{ag.id}] reward {reward:+.3f}")

                done = self.ambiente.is_episode_done()
                self.ambiente.atualizacao()
                step += 1

                if render:
                    if hasattr(self.ambiente, 'render') and callable(self.ambiente.render):
                        self.ambiente.render()
                    elif self.visualizador:
                        try:
                            self.visualizador.draw(self.ambiente)
                        except Exception:
                            pass
                    time.sleep(self.params.get('render_delay', 0.0))

            for ag in self.agentes:
                self.metrics['reward_' + ag.id].append(ep_reward[ag.id])

            self.metrics['steps'].append(step)

            if hasattr(self.ambiente, 'done_agents'):
                success = len(getattr(self.ambiente, 'done_agents', set())) / max(1, len(self.agentes))
                self.metrics['success_rate'].append(success)
            if hasattr(self.ambiente, 'total_delivered'):
                self.metrics['resources_delivered'].append(getattr(self.ambiente, 'total_delivered', 0))

            try:
                if hasattr(self.ambiente, 'farol'):
                    self.tracker.regista_farol(self.ambiente, ep_reward, steps=step)
                if hasattr(self.ambiente, 'resources'):
                    self.tracker.regista_foraging(self.ambiente, ep_reward, steps=step)
            except Exception:
                pass

            if verbose:
                print(f"🏁 EP {ep+1} done steps={step} rewards={ep_reward}")

        extras = dict(self.tracker.data) if self.tracker else {}
        return dict(self.metrics), extras
