# Farol environment
import random
from enum import Enum
class TipoDirecao(Enum):
    N="N"; S="S"; E="E"; O="O"; NONE="NONE"

class FarolEnv:
    def __init__(self,size=10,max_steps=200,farol_fixo=None):
        self.size=size; self.max_steps=max_steps
        self.farol=farol_fixo or (size//2,size//2)
        self.step=0; self.agent_ids=[]; self.agent_pos={}; self.done_agents=set()
    def registar_agentes(self,ags): self.agent_ids=[a.id for a in ags]
    def reset(self):
        self.step=0; self.done_agents=set()
        spots=[(x,y) for x in range(self.size) for y in range(self.size) if (x,y)!=self.farol]
        random.shuffle(spots)
        for i,aid in enumerate(self.agent_ids):
            self.agent_pos[aid]=spots[i]
        return self._state()
    def _state(self): return {'farol':self.farol,'agents':dict(self.agent_pos)}
    def observacaoPara(self,ag):
        pos=self.agent_pos[ag.id]; x,y=pos
        obs={'pos':pos}
        for s in ag.sensores:
            if s.tipo=='farol':
                obs['direcao_farol']=self._dir(pos).value
            if s.tipo=='visao':
                obs['visao']=self._visao(x,y,s.alcance)
        return obs
    def _dir(self,pos):
        xa,ya=pos; xf,yf=self.farol
        if xf>xa: return TipoDirecao.E
        if xf<xa: return TipoDirecao.O
        if yf>ya: return TipoDirecao.S
        if yf<ya: return TipoDirecao.N
        return TipoDirecao.NONE
    def _visao(self,x,y,a):
        res={}
        neigh={'L':(x-1,y),'R':(x+1,y),'U':(x,y-1),'D':(x,y+1),'C':(x,y)}
        for k,(nx,ny) in neigh.items():
            if 0<=nx<self.size and 0<=ny<self.size:
                if (nx,ny)==self.farol: res[k]='FAROL'
                else:
                    found=None
                    for ag,pp in self.agent_pos.items():
                        if pp==(nx,ny): found=f"AG_{ag}"; break
                    res[k]=found or 'VAZIO'
            else: res[k]='PAREDE'
        return res
    def agir(self,acao,ag):
        if ag.id in self.done_agents: return 0.0,True
        x,y=self.agent_pos[ag.id]
        r=-0.01; t=False
        if acao=='UP' and y>0: y-=1
        elif acao=='DOWN' and y<self.size-1: y+=1
        elif acao=='LEFT' and x>0: x-=1
        elif acao=='RIGHT' and x<self.size-1: x+=1
        elif acao=='STAY': pass
        else: r=-0.1
        self.agent_pos[ag.id]=(x,y)
        if (x,y)==self.farol:
            r=10.0; self.done_agents.add(ag.id); t=True
        return r,t
    def atualizacao(self): self.step+=1
    def is_episode_done(self): return self.step>=self.max_steps or len(self.done_agents)==len(self.agent_ids)
