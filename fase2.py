import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from statistics import mean

data = pd.read_csv("./data.csv", header=None)

df = pd.DataFrame(data)

# print(df.describe()) #Realiza a média dos valores
x = df.iloc[:, :-1].values  # Seleciona todas as colunas, exceto a última
y = df.iloc[:, -1].values

def correlacao(x, y):
    # Calcular a média de x e y
    media_x = sum(x) / len(x)
    media_y = sum(y) / len(y)

    # Calcular as diferenças
    diffs_x = [xi - media_x for xi in x]
    diffs_y = [yi - media_y for yi in y]

    # Calcular o produto das diferenças correspondentes
    produtos_diferencas = [diffs_x[i] * diffs_y[i] for i in range(len(x))]

    # Calcular o quadrado das diferenças
    quadrados_diffs_x = [(xi - media_x)**2 for xi in x]
    quadrados_diffs_y = [(yi - media_y)**2 for yi in y]

    # Calcular a correlação
    correlacao = sum(produtos_diferencas) / (
        math.sqrt(sum(quadrados_diffs_x)) * math.sqrt(sum(quadrados_diffs_y)))

    return correlacao

def regressao(x, y):
    media_x = mean(x)
    media_y = mean(y)

    somatorio = 0
    somatorio_2 = 0
    for index in range(len(x)):
        somatorio += (x[index] - media_x) * (y[index] - media_y)
        somatorio_2 += (x[index] - media_x)**2

    b1 = somatorio / somatorio_2
    b0 = media_y - b1 * media_x

    return b0, b1

# Calcule a correlação entre "Tamanho da Casa" e "Preço"
correlacao_tamanho_preco = correlacao(x[:, 0], y)

# Calcule a correlação entre "Número de Quartos" e "Preço"
correlacao_quartos_preco = correlacao(x[:, 1], y)

b0_tamanho, b1_tamanho = regressao(x[:, 0], y)

# Aplicar a função de regressão para "Número de Quartos" e "Preço"
b0_quartos, b1_quartos = regressao(x[:, 1], y)

# Crie gráficos de dispersão para visualizar os dados e as linhas de regressão
plt.figure(figsize=(12, 5))

# Gráfico de dispersão para "Tamanho da Casa" e "Preço"
plt.subplot(1, 2, 1)
plt.scatter(x[:, 0], y, color='blue', alpha=0.7)
plt.plot(x[:, 0], [b0_tamanho + b1_tamanho * xi for xi in x[:, 0]], color='red', linewidth=2)
plt.title(f'Regressão para Tamanho da Casa')
plt.xlabel('Tamanho da Casa')
plt.ylabel('Preço')

# Gráfico de dispersão para "Número de Quartos" e "Preço"
plt.subplot(1, 2, 2)
plt.scatter(x[:, 1], y, color='green', alpha=0.7)
plt.plot(x[:, 1], [b0_quartos + b1_quartos * xi for xi in x[:, 1]], color='red', linewidth=2)
plt.title(f'Regressão para Número de Quartos')
plt.xlabel('Número de Quartos')
plt.ylabel('Preço')

plt.tight_layout()
plt.show()
