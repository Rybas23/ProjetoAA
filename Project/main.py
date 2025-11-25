from Project.Simulador import Simulador

def main():
    # Código principal da simulação
    sim = Simulador()
    motor = sim.cria("parametros_simulacao.json")
    sim.executa(passos=5)

# Verifica se o script está sendo executado diretamente
if __name__ == "__main__":
    main()