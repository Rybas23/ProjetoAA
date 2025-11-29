from collections import defaultdict

class MetricsTracker:
    def __init__(self, agentes):
        self.agentes = agentes
        self.data = defaultdict(list)

    def _manhattan(self, p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    def regista_farol(self, ambiente, ep_reward, steps=None):
        for ag in self.agentes:
            aid = ag.id
            pos = ambiente.agent_pos.get(aid, None)
            if pos is None:
                self.data[f"dist_final_{aid}"].append(None)
                self.data[f"sucesso_{aid}"].append(0)
            else:
                d = self._manhattan(pos, ambiente.farol)
                self.data[f"dist_final_{aid}"].append(d)
                self.data[f"sucesso_{aid}"].append(1 if pos == ambiente.farol else 0)
        if steps is not None:
            self.data['steps_ep'].append(steps)

    def regista_foraging(self, ambiente, ep_reward, steps=None):
        self.data['resources_delivered'].append(getattr(ambiente, 'total_delivered', 0))
        if steps is not None:
            self.data['steps_ep'].append(steps)
        for ag in self.agentes:
            aid = ag.id
            self.data[f"reward_{aid}"].append(ep_reward.get(aid, 0))
