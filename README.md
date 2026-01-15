# Simulador Agentes Autónomos

Bem vindo ao nosso projeto de Agentes Autónomos!

Deixamos aqui algumas intruções para facilitar o teste do nosso projeto.

No ficheiro run_main.py temos este método:

```python
def main():
    simulador = SimuladorInterativo()
    try:
        #simulador.executarJson("farol.json")
        #simulador.executarJson("farolFixo.json")
        #simulador.executarJson("farol_ga.json")
        #simulador.executarJson("foraging.json")
        #simulador.executarJson("foragingFixo.json")
        simulador.executarJson("foraging_ga.json")
    except KeyboardInterrupt:
        print("\nInterrompido pelo utilizador")
```
Para testar cada um dos cenários, basta comentar a linha ativa e descomentar a linha do caso que pretende testar.

Através dos ficheiros json, é possível configurar os parametros dos problemas implementados. É possível verificar o nome dos ficheiros que irão ser gerados após testes através do campo output.csv. Estes ficheiros servem para validar os resultados das simulações, mais propriamente as métricas avaliadas.