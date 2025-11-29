import random
def policy_farol_inteligente(obs):
    d=obs.get('direcao_farol')
    return {'N':'UP','S':'DOWN','E':'RIGHT','O':'LEFT','NONE':'STAY'}.get(d,'STAY')

def policy_foraging_inteligente(obs):
    x,y=obs['pos']; nx,ny=obs['nest']
    if obs.get('carrying',0)==1:
        if (x,y)==(nx,ny): return 'DROP'
        if nx>x: return 'RIGHT'
        if nx<x: return 'LEFT'
        if ny>y: return 'DOWN'
        if ny<y: return 'UP'
    if obs['visao'].get('C',0)>0: return 'PICK'
    for k,v in obs['visao'].items():
        if k!='C' and v>0:
            return {'L':'LEFT','R':'RIGHT','U':'UP','D':'DOWN'}[k]
    return random.choice(['UP','DOWN','LEFT','RIGHT'])

def policy_aleatoria(obs):
    """Política completamente aleatória"""
    return random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'])
