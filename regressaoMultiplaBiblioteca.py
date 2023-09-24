import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

data = pd.read_csv("./data.csv", header=None)
df = pd.DataFrame(data)

# Variáveis independentes (X) e variável dependente (y)
X = df.iloc[:, :-1].values  # Todas as colunas, exceto a última
X = sm.add_constant(X)  # Adicionar constante para o termo de interceptação
y = df.iloc[:, -1].values  # Última coluna

# Realizar a regressão múltipla
modelo = sm.OLS(y, X).fit()

# Coeficientes do modelo
b0 = modelo.params[0]
b1 = modelo.params[1]
b2 = modelo.params[2]

# Imprimir os resultados da regressão
print(modelo.summary())

# Gráfico de dispersão 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 1], X[:, 2], y, c='blue', marker='o')
ax.set_xlabel("Tamanho da Casa")
ax.set_ylabel("Número de Quartos")
ax.set_zlabel("Preço")

# Superfície de regressão
x_surf = np.arange(X[:, 1].min(), X[:, 1].max(), 100)
y_surf = np.arange(X[:, 2].min(), X[:, 2].max(), 0.1)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = b0 + b1 * x_surf + b2 * y_surf
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5)

# Traçar a linha da regressão
ax.plot(X[:, 1], X[:, 2], modelo.predict(X), color='green', linewidth=2)

# Girar o gráfico para uma melhor visualização
ax.view_init(elev=20, azim=50)  # Ajustar os ângulos de elevação e azimute

plt.show()
