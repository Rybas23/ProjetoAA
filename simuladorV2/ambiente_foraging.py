import random
class ForagingEnv:
    def __init__(self,width=10,height=10,n_resources=10,nest=(0,0),max_steps=200):
        self.width=width; self.height=height; self.n_resources=n_resources
        self.nest=nest; self.max_steps=max_steps
        self.step=0; self.agent_ids=[]; self.agent_pos={}; self.carrying={}; self.resources={}

    def registar_agentes(self,ags): self.agent_ids=[a.id for a in ags]

    def reset(self):
        self.step=0; self.resources={}
        for _ in range(self.n_resources):
            x=random.randint(0,self.width-1); y=random.randint(0,self.height-1)
            self.resources[(x,y)]=self.resources.get((x,y),0)+1
        for aid in self.agent_ids:
            self.agent_pos[aid]=self.nest; self.carrying[aid]=0
        self.total_delivered=0
        return self._state()

    def _state(self):
        return {'resources':dict(self.resources),'agents':dict(self.agent_pos),'nest':self.nest}

    def observacaoPara(self,ag):
        x,y=self.agent_pos[ag.id]
        vis={}
        for dx,dy,k in [(-1,0,'L'),(1,0,'R'),(0,-1,'U'),(0,1,'D')]:
            nx,ny=x+dx,y+dy
            if 0<=nx<self.width and 0<=ny<self.height: vis[k]=self.resources.get((nx,ny),0)
            else: vis[k]=-1
        vis['C']=self.resources.get((x,y),0)
        return {'pos':(x,y),'visao':vis,'carrying':self.carrying[ag.id],'nest':self.nest}

    def agir(self, acao, ag):
        x, y = self.agent_pos[ag.id]
        r = 0.0
        t = False

        if acao == 'UP':
            y = max(0, y - 1)
        elif acao == 'DOWN':
            y = min(self.height - 1, y + 1)
        elif acao == 'LEFT':
            x = max(0, x - 1)
        elif acao == 'RIGHT':
            x = min(self.width - 1, x + 1)

        elif acao == 'PICK':
            if self.resources.get((x, y), 0) > 0 and self.carrying[ag.id] == 0:
                self.resources[(x, y)] -= 1
                if self.resources[(x, y)] == 0:
                    del self.resources[(x, y)]
                self.carrying[ag.id] = 1
                r = 0.5

        elif acao == 'DROP':
            if (x, y) == self.nest and self.carrying[ag.id] == 1:
                self.carrying[ag.id] = 0
                self.total_delivered += 1  # ← FIX PRINCIPAL
                r = 1.0

        self.agent_pos[ag.id] = (x, y)

        if acao in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']:
            r = -0.01

        return r, t

    def atualizacao(self): self.step+=1

    def is_episode_done(self):
        return self.step >= self.max_steps or len(self.resources) == 0
