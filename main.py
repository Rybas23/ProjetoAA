import random
from simulador.engine import MotorDeSimulacao
from simulador.farol_env import FarolEnv
from simulador.foraging_env import ForagingEnv
from simulador.q_agent import QAgent
from simulador.fixed_agent import FixedAgent

# ----------------------------
# Políticas inteligentes FixedAgent
# ----------------------------
def policy_farol(obs):
    dx, dy = obs['dir']
    if dx > 0: return 'RIGHT'
    if dx < 0: return 'LEFT'
    if dy > 0: return 'DOWN'
    if dy < 0: return 'UP'
    return 'STAY'

def policy_foraging(obs):
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
    if obs['neigh'].get('C', 0) > 0:
        return 'PICK'

    # Verificar vizinhos
    for dir_name, amount in obs['neigh'].items():
        if dir_name == 'C':
            continue
        if amount > 0:
            return {'L':'LEFT','R':'RIGHT','U':'UP','D':'DOWN'}[dir_name]

    # Mover aleatoriamente se não encontrou recurso
    return random.choice(['UP','DOWN','LEFT','RIGHT'])

# ----------------------------
# Função para executar simulação
# ----------------------------
def run_simulacao(env, agentes, params, render=True):
    motor = MotorDeSimulacao(params)
    motor.adiciona_ambiente(env)
    for ag in agentes:
        motor.adiciona_agente(ag)

    metrics = motor.executa(render=render)
    print("\n=== Métricas ===")
    for key, values in metrics.items():
        print(f"{key}: {values}")
    print("\n\n")

# ----------------------------
# Parâmetros do motor
# ----------------------------
params = {
    'episodes': 10,        # aumentei para mais aprendizado
    'max_steps': 200,
    'render_delay': 0.05
}

# ============================
# 1️⃣ FarolEnv - múltiplos agentes
# ============================
print(">>> FarolEnv - múltiplos agentes - learn (QAgent)")
agentes = [QAgent(
    id=f'a{i}',
    actions=['UP','DOWN','LEFT','RIGHT'],
    modo='learn',
    alpha=0.9,
    gamma=0.95,
    epsilon=0.3
) for i in range(3)]
env = FarolEnv(size=10, n_agents=3, max_steps=params['max_steps'])
run_simulacao(env, agentes, params)

print(">>> FarolEnv - múltiplos agentes - test (FixedAgent)")
agentes = [FixedAgent(id=f'a{i}', policy=policy_farol, modo='test') for i in range(3)]
env = FarolEnv(size=10, n_agents=3, max_steps=params['max_steps'])
run_simulacao(env, agentes, params)

# ============================
# 2️⃣ ForagingEnv - múltiplos agentes
# ============================
print(">>> ForagingEnv - múltiplos agentes - learn (QAgent)")
agentes = [QAgent(
    id=f'a{i}',
    actions=['UP','DOWN','LEFT','RIGHT','PICK','DROP'],
    modo='learn',
    alpha=0.9,
    gamma=0.95,
    epsilon=0.3
) for i in range(3)]
env = ForagingEnv(width=10, height=10, n_agents=3, n_resources=10, nest=(0,0), max_steps=params['max_steps'])
run_simulacao(env, agentes, params)

print(">>> ForagingEnv - múltiplos agentes - test (FixedAgent)")
agentes = [FixedAgent(id=f'a{i}', policy=policy_foraging, modo='test') for i in range(3)]
env = ForagingEnv(width=10, height=10, n_agents=3, n_resources=10, nest=(0,0), max_steps=params['max_steps'])
run_simulacao(env, agentes, params)

# ============================
# 3️⃣ FarolEnv - 1 agente
# ============================
print(">>> FarolEnv - 1 agente - learn (QAgent)")
agentes = [QAgent(
    id='a0',
    actions=['UP','DOWN','LEFT','RIGHT'],
    modo='learn',
    alpha=0.9,
    gamma=0.95,
    epsilon=0.3
)]
env = FarolEnv(size=10, n_agents=1, max_steps=params['max_steps'])
run_simulacao(env, agentes, params)

print(">>> FarolEnv - 1 agente - test (FixedAgent)")
agentes = [FixedAgent(id='a0', policy=policy_farol, modo='test')]
env = FarolEnv(size=10, n_agents=1, max_steps=params['max_steps'])
run_simulacao(env, agentes, params)

# ============================
# 4️⃣ ForagingEnv - 1 agente
# ============================
print(">>> ForagingEnv - 1 agente - learn (QAgent)")
agentes = [QAgent(
    id='a0',
    actions=['UP','DOWN','LEFT','RIGHT','PICK','DROP'],
    modo='learn',
    alpha=0.9,
    gamma=0.95,
    epsilon=0.3
)]
env = ForagingEnv(width=10, height=10, n_agents=1, n_resources=10, nest=(0,0), max_steps=params['max_steps'])
run_simulacao(env, agentes, params)

print(">>> ForagingEnv - 1 agente - test (FixedAgent)")
agentes = [FixedAgent(id='a0', policy=policy_foraging, modo='test')]
env = ForagingEnv(width=10, height=10, n_agents=1, n_resources=10, nest=(0,0), max_steps=params['max_steps'])
run_simulacao(env, agentes, params)