import random

# Política inteligente para o ambiente Farol
# Segue diretamente a direção indicada pelo sensor
def policy_farol_inteligente(observacao):
    # Lê direção do farol a partir da observação
    direcao_do_farol = observacao.get("direcao_farol")

    # Mapa de conversão de direção → ação
    mapa_acao_por_direcao = {
        "N": "UP",
        "S": "DOWN",
        "E": "RIGHT",
        "O": "LEFT",
    }

    # Se a direção for conhecida, seguimos diretamente.
    if direcao_do_farol in mapa_acao_por_direcao:
        return mapa_acao_por_direcao[direcao_do_farol]

    # Caso contrário (NONE ou desconhecido), escolhemos um movimento aleatório.
    return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])


# Política inteligente para Foraging (colheita)
# Regras:
# 1. Se está a carregar recurso → voltar ao ninho e largar
# 2. Se está em cima de recurso → PICK
# 3. Se vê recurso ao lado → mover para lá
# 4. Caso contrário → movimento aleatório
def policy_foraging_inteligente(observacao):
    pos_x, pos_y = observacao["pos"]
    ninho_x, ninho_y = observacao["nest"]

    # 1. Se o agente está a carregar recurso → regressar ao ninho
    if observacao.get("carrying", 0) == 1:

        # Se chegou ao ninho → largar
        if (pos_x, pos_y) == (ninho_x, ninho_y):
            return "DROP"

        # Movimento na direção do ninho
        if ninho_x > pos_x:
            return "RIGHT"
        if ninho_x < pos_x:
            return "LEFT"
        if ninho_y > pos_y:
            return "DOWN"
        if ninho_y < pos_y:
            return "UP"

    # 2. Se está em cima de recurso → PICK
    if observacao["visao"].get("C", 0) > 0:
        return "PICK"

    # 3. Se vê recurso ao lado → mover para ele
    for direcao, quantidade in observacao["visao"].items():
        if direcao != "C" and quantidade > 0:
            mapa_movimentos = {
                "L": "LEFT",
                "R": "RIGHT",
                "U": "UP",
                "D": "DOWN",
            }
            return mapa_movimentos[direcao]

    # 4. Movimento aleatório se não houver recurso
    return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])


# Política completamente aleatória → sem inteligência
def policy_aleatoria(observacao):
    return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
