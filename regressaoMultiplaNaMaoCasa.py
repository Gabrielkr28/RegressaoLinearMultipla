import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

data = pd.read_csv("./dataComp.csv", header=None)
df = pd.DataFrame(data)

n = len(df.iloc[:, 0])

mean_X1 = sum(df.iloc[:, 0]) / n
mean_X2 = sum(df.iloc[:, 1]) / n
mean_Y = sum(df.iloc[:, 2]) / n

# Calculando os coeficientes (beta) da regressão
numerator1 = sum((df.iloc[:, 0][i] - mean_X1) * (df.iloc[:, 2][i] - mean_Y) for i in range(n))
numerator2 = sum((df.iloc[:, 1][i] - mean_X2) * (df.iloc[:, 2][i] - mean_Y) for i in range(n))

denominator1 = sum((df.iloc[:, 0][i] - mean_X1) ** 2 for i in range(n))
denominator2 = sum((df.iloc[:, 1][i] - mean_X2) ** 2 for i in range(n))

beta1 = numerator1 / denominator1
beta2 = numerator2 / denominator2

beta0 = mean_Y - beta1 * mean_X1 - beta2 * mean_X2

# Variáveis independentes (X) e variável dependente (y)
X = df.iloc[:, :-1].values  # Todas as colunas, exceto a última
X = sm.add_constant(X)  # Adicionar constante para o termo de interceptação
y = df.iloc[:, -1].values  # Última coluna

# Realizar a regressão múltipla
modelo = sm.OLS(y, X).fit()

# Coeficientes do modelo
b0 = beta0
b1 = beta1
b2 = beta2

# Imprimir os resultados da regressão
print(modelo.summary())

# Gráfico de dispersão 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df.iloc[:, 0], df.iloc[:, 1], y, c='blue', marker='o')
ax.set_xlabel("Tamanho da Casa")
ax.set_ylabel("Número de Quartos")
ax.set_zlabel("Preço")

# Superfície de regressão
x_surf = np.arange(df.iloc[:, 0].min(), df.iloc[:, 0].max(), 100)
y_surf = np.arange(df.iloc[:, 2].min(), df.iloc[:, 1].max(), 0.1)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = b0 + b1 * x_surf + b2 * y_surf
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5)

# Traçar a linha da regressão
ax.plot(df.iloc[:, 0], df.iloc[:, 1], modelo.predict(X), color='green', linewidth=2)

# Girar o gráfico para uma melhor visualização
ax.view_init(elev=20, azim=50)  # Ajustar os ângulos de elevação e azimute

plt.show()
