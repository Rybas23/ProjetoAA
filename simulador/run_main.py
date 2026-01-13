import csv
import os
import copy
import pickle

from engine import MotorDeSimulacao
from visualizador import Visualizador


def _print_qtable_summary(agent, max_states=10):
    """Imprime um pequeno resumo da Q-table de um agente Q-learning."""
    if not hasattr(agent, "qtable") or not agent.qtable:
        print(f"[QTABLE] Agente {agent.id} não tem Q-table ou está vazia.")
        return

    print(f"\n=== Q-table summary for agent {agent.id} ===")
    print("Number of states:", len(agent.qtable))

    for i, (state, action_qs) in enumerate(agent.qtable.items()):
        if i >= max_states:
            break
        best_action = max(action_qs, key=action_qs.get)
        print(f"\nState {i}: {state}")
        print("  Q-values:", action_qs)
        print("  Best action:", best_action)


def _export_qtable(agent, base_name: str):
    """Guarda a Q-table em ficheiro .csv"""
    if not hasattr(agent, "qtable") or not agent.qtable:
        return

    # Guardar CSV: uma linha por (estado, ação)
    csv_path = f"{base_name}_{agent.id}.csv"
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["state", "action", "q_value"])
            for state, action_qs in agent.qtable.items():
                for action, q in action_qs.items():
                    writer.writerow([repr(state), action, q])
        print(f"[QTABLE] Guardada Q-table em {csv_path}")
    except Exception as e:
        print(f"[QTABLE] Erro ao guardar CSV para {agent.id}: {e}")


class SimuladorInterativo:
    """
    Pequeno invólucro para manter compatibilidade com o modo anterior:
    - executarJson(path_json)
    - menu_principal() (placeholder simples)
    """

    def __init__(self):
        pass

    def executarJson(self, ficheiro_json):
        """
        Lê um ficheiro JSON de configuração, executa a simulação e
        exporta métricas para CSV (se existir diretiva no JSON).
        """
        motor = MotorDeSimulacao.cria(ficheiro_json)

        params = motor.params
        sim_cfg = params.get("simulation", {})
        render = sim_cfg.get("render", False)
        logs = sim_cfg.get("logs", False)
        usar_visualizador = sim_cfg.get("use_visualizer", False)

        viz = None
        if usar_visualizador:
            # Tentar deduzir tamanho da grelha a partir do ambiente
            amb = motor.ambiente
            width = getattr(amb, "width", getattr(amb, "size", 10))
            height = getattr(amb, "height", getattr(amb, "size", 10))
            viz = Visualizador(
                grid_width=width,
                grid_height=height,
                title=f"{params.get('problem', 'Simulacao')}",
                fps=sim_cfg.get("fps", 5),
            )
            motor.liga_visualizador(viz)

        metrics, extras = motor.executa(render=render, logs=logs)

        # Após a simulação, imprimir e exportar Q-table para quaisquer agentes Q*
        for ag in motor.listaAgentes():
            if ag.__class__.__name__.startswith("QAgent"):
                _print_qtable_summary(ag, max_states=10)
                # base_name inclui o problema para distinguir ficheiros
                base_name = f"qtable_{params.get('problem', 'env').lower()}"
                _export_qtable(ag, base_name)

        if viz:
            viz.cleanup()

        # Exportar métricas para CSV se for pedido no JSON
        out_cfg = params.get("output", {})
        csv_path = out_cfg.get("csv", None)
        if csv_path:
            self._exporta_csv(csv_path, metrics, extras)

    def _exporta_csv(self, path, metrics, extras):
        """
        Exporta métricas e extras num único CSV "wide".
        Cada chave torna-se uma coluna; valores são alinhados por índice.
        """
        # Juntar métricas + extras
        all_data = {}
        all_data.update(metrics or {})
        all_data.update(extras or {})

        # Determinar comprimento máximo
        max_len = 0
        for v in all_data.values():
            if isinstance(v, list):
                max_len = max(max_len, len(v))
            else:
                max_len = max(max_len, 1)

        # Normalizar para listas de comprimento max_len
        norm = {}
        for k, v in all_data.items():
            if isinstance(v, list):
                lst = v[:]
            else:
                lst = [v]
            if len(lst) < max_len:
                lst.extend([None] * (max_len - len(lst)))
            norm[k] = lst

        # Adicionar coluna de id sequencial (1..max_len)
        norm["id"] = list(range(1, max_len + 1))

        fieldnames = sorted(norm.keys())

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(max_len):
                row = {k: norm[k][i] for k in fieldnames}
                writer.writerow(row)

    def menu_principal(self):
        """
        Placeholder muito simples para um menu de consola.
        Mantém compatibilidade com a versão anterior, mas aqui apenas
        permite escolher rapidamente um ficheiro JSON.
        """
        print("=== Simulador Interativo ===")
        print("1) Executar a partir de `farol.json`")
        print("2) Executar a partir de `foraging.json`")
        print("0) Sair")

        opcao = input("Opcao: ").strip()
        if opcao == "1":
            self.executarJson("farol.json")
        elif opcao == "2":
            self.executarJson("foraging.json")
        else:
            print("A sair...")


def quick_regression_tests():
    """
    Executa um pequeno conjunto de testes programáticos para:
    - Farol com Q-learning em modo learn
    - Farol em modo test com Q-table pré-treinada (sem alteração)
    - Foraging com Q-learning em modo learn

    Imprime métricas chave no stdout.
    """

    print("=== REGRESSION TESTS: FAROL (LEARN) ===")

    params_farol_learn = {
        "problem": "Farol",
        "episodes": 5,
        "max_steps": 100,
        "environment": {
            "size": 7,
            "max_steps": 100,
            "farol_fixo": [3, 3],
            "walls": [[2, 2], [2, 3]],
        },
        "agents": [
            {
                "id": "Q1",
                "type": "QAgentFarol",
                "mode": "learn",
                "spawn": [0, 0],
            }
        ],
        "simulation": {
            "render": False,
            "logs": False,
        },
    }

    motor_farol_learn = MotorDeSimulacao.cria(params_farol_learn)
    metrics_f_learn, extras_f_learn = motor_farol_learn.executa(
        render=False, logs=False
    )
    ag_f_learn = motor_farol_learn.listaAgentes()[0]

    print("Farol learn -> rewards:", metrics_f_learn.get("reward_Q1"))
    print("Farol learn -> success_rate:", metrics_f_learn.get("success_rate"))
    print("Farol learn -> tracker dist_final_Q1:", extras_f_learn.get("dist_final_Q1"))
    print("Farol learn -> agent internal metrics:", ag_f_learn.get_metrics())

    print("\n=== REGRESSION TESTS: FAROL (TEST MODE, FIXED POLICY) ===")

    params_farol_test = copy.deepcopy(params_farol_learn)
    params_farol_test["agents"][0]["mode"] = "test"

    motor_farol_test = MotorDeSimulacao.cria(params_farol_test)
    ag_f_test = motor_farol_test.listaAgentes()[0]

    # Carregar Q-table treinada
    ag_f_test.load_qtable(qt_path)

    metrics_f_test, extras_f_test = motor_farol_test.executa(
        render=False, logs=False
    )
    q_after = ag_f_test.qtable

    print("Farol test -> rewards:", metrics_f_test.get("reward_Q1"))
    print("Farol test -> success_rate:", metrics_f_test.get("success_rate"))
    print("Farol test -> tracker dist_final_Q1:", extras_f_test.get("dist_final_Q1"))
    print("Farol test -> agent internal metrics:", ag_f_test.get_metrics())
    print("Farol test -> Q-table changed in test mode?", q_before != q_after)


    print("\n=== REGRESSION TESTS: FORAGING (LEARN) ===")

    params_foraging_learn = {
        "problem": "Foraging",
        "episodes": 5,
        "max_steps": 150,
        "environment": {
            "width": 7,
            "height": 7,
            "n_resources": 10,
            "ninho": [0, 0],
            "max_steps": 150,
            "walls": [[3, 3], [3, 4]],
        },
        "agents": [
            {
                "id": "QF1",
                "type": "QAgentForaging",
                "mode": "learn",
                "spawn": [0, 0],
            }
        ],
        "simulation": {
            "render": False,
            "logs": False,
        },
    }

    motor_foraging = MotorDeSimulacao.cria(params_foraging_learn)
    metrics_fg, extras_fg = motor_foraging.executa(render=False, logs=False)
    ag_fg = motor_foraging.listaAgentes()[0]

    print("Foraging learn -> rewards:", metrics_fg.get("reward_QF1"))
    print(
        "Foraging learn -> resources_delivered:",
        metrics_fg.get("resources_delivered"),
    )
    print(
        "Foraging learn -> tracker resources_delivered:",
        extras_fg.get("resources_delivered"),
    )
    print("Foraging learn -> agent internal metrics:", ag_fg.get_metrics())


def main():
    simulador = SimuladorInterativo()
    try:
        # simulador.executarJson("farol.json")
        # simulador.executarJson("farolFixo.json")
        #simulador.executarJson("foragingFixo.json")
         simulador.executarJson("foraging.json")
    except KeyboardInterrupt:
        print("\nInterrompido pelo utilizador")


if __name__ == "__main__":
    main()
