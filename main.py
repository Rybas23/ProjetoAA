from simulador.engine import MotorDeSimulacao
from simulador.farol_env import FarolEnv
from simulador.foraging_env import ForagingEnv
from simulador.q_agent import QAgent
from simulador.fixed_agent import FixedAgent

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=['farol','foraging'], default='farol')
    parser.add_argument('--params', default='params/foraging_params.json')
    args = parser.parse_args()

    engine = MotorDeSimulacao.cria(args.params)
    if args.scenario=='farol':
        env = FarolEnv(size=10, n_agents=1, max_steps=200)
        engine.adiciona_ambiente(env)
        a0 = QAgent('a0',['UP','DOWN','LEFT','RIGHT','STAY'], modo='learn')
        engine.adiciona_agente(a0)
    else:
        env = ForagingEnv(width=10, height=10, n_agents=2, n_resources=15, nest=(0,0), max_steps=200)
        engine.adiciona_ambiente(env)
        a0 = QAgent('a0',['UP','DOWN','LEFT','RIGHT','PICK','DROP','STAY'], modo='learn')
        a1 = FixedAgent('a1', policy=lambda obs: 'STAY', modo='test')
        engine.adiciona_agente(a0)
        engine.adiciona_agente(a1)

    metrics = engine.executa(render=False)
    print('Metrics:', metrics)
