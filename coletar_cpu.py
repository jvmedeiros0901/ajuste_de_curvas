"""Coleta os dados do problema de Engenharia de Computação: a temperatura da CPU.

Funciona como o sistema do enunciado, mas é o próprio computador:

- entrada u: carga na CPU. A cada MEIO_PERIODO segundos o script liga (u = 1)
  ou desliga (u = 0) um cálculo pesado em todos os núcleos, formando uma onda
  quadrada;
- saída: temperatura do processador (°C), lida do sensor da placa.

A velocidade da ventoinha (rpm) também é gravada no CSV, mas não entra na
análise: o projeto usa só o tempo, a carga u e a temperatura.

O sensor mede em graus inteiros, então a temperatura já vem com um "ruído" de
arredondamento, além das variações naturais.

    python coletar_cpu.py                 # coleta completa (cerca de 12 min)
    python coletar_cpu.py --duracao 20    # teste rápido

Use com o carregador ligado. Ctrl+C interrompe e salva o que já foi coletado.
"""

import argparse
import csv
import multiprocessing as mp
import time
from pathlib import Path

HWMON = Path("/sys/class/hwmon")


def achar_sensor(nome_chip, rotulo=None, arquivo=None):
    """Procura o arquivo do sensor em /sys/class/hwmon.

    Os números hwmon0, hwmon1… podem mudar a cada boot, então a busca é feita
    pelo nome do chip ("coretemp") e pelo rótulo ("Package id 0").
    """
    for pasta in sorted(HWMON.glob("hwmon*")):
        if (pasta / "name").read_text().strip() != nome_chip:
            continue
        if arquivo:
            return pasta / arquivo
        for label in pasta.glob("temp*_label"):
            if label.read_text().strip() == rotulo:
                return pasta / label.name.replace("_label", "_input")
    raise FileNotFoundError(f"sensor não encontrado: {nome_chip} {rotulo or arquivo}")


def ler(caminho, divisor=1):
    return int(caminho.read_text()) / divisor


def trabalhador(ligado, parar):
    """Roda em cada núcleo: faz contas sem parar enquanto a carga está ligada."""
    x = 0.0
    while not parar.is_set():
        if ligado.is_set():
            for i in range(200_000):
                x += i * i % 7
        else:
            time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duracao", type=float, default=750, help="tempo total em segundos")
    parser.add_argument("--meio-periodo", type=float, default=30, help="segundos com a carga ligada (ou desligada)")
    parser.add_argument("--amostragem", type=float, default=0.5, help="intervalo entre leituras em segundos")
    parser.add_argument("--saida", default="dados/dados_cpu.csv")
    args = parser.parse_args()

    sensor_temp = achar_sensor("coretemp", rotulo="Package id 0")
    sensor_fan = achar_sensor("dell_smm", arquivo="fan1_input")

    ligado, parar = mp.Event(), mp.Event()
    processos = [mp.Process(target=trabalhador, args=(ligado, parar)) for _ in range(mp.cpu_count())]
    for p in processos:
        p.start()

    linhas = []
    n_amostras = int(args.duracao / args.amostragem) + 1
    inicio = time.monotonic()
    print(f"Coletando {n_amostras} amostras em {args.duracao:.0f} s ({mp.cpu_count()} núcleos)...")
    try:
        for k in range(n_amostras):
            t = k * args.amostragem
            # começa com a carga desligada, para registrar a temperatura de repouso
            u = int((t // args.meio_periodo) % 2 == 1)
            ligado.set() if u else ligado.clear()

            # espera até o instante exato da amostra k
            time.sleep(max(0.0, inicio + t - time.monotonic()))
            temperatura = ler(sensor_temp, 1000)
            ventoinha = ler(sensor_fan)
            linhas.append((round(t, 2), u, temperatura, ventoinha))

            if k % int(10 / args.amostragem) == 0:
                print(f"  t = {t:6.1f} s  carga = {'ligada ' if u else 'desligada'}  "
                      f"{temperatura:5.1f} °C  {ventoinha:5.0f} rpm")
    except KeyboardInterrupt:
        print("\nInterrompido, salvando o que foi coletado.")
    finally:
        parar.set()
        for p in processos:
            p.join()

    saida = Path(args.saida)
    saida.parent.mkdir(exist_ok=True)
    with open(saida, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["t", "u", "temperatura", "ventoinha"])
        escritor.writerows(linhas)
    print(f"{len(linhas)} amostras salvas em {saida}")


if __name__ == "__main__":
    main()
