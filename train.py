import numpy as np
import gymnasium as gym
from q_learning import update_q

# =========================
# CREACIÓN DEL ENTORNO
# =========================
# FrozenLake es un entorno de RL donde el agente debe llegar a la meta
# evitando agujeros en el hielo
env = gym.make("FrozenLake-v1")

# =========================
# INICIALIZACIÓN DE Q-TABLE
# =========================
# Q(s,a) representa el valor esperado de tomar una acción en un estado
Q = np.zeros((16, 4))

# =========================
# HIPERPARÁMETROS
# =========================
alpha = 0.1      # tasa de aprendizaje
gamma = 0.95     # factor de descuento
epsilon = 1.0    # exploración inicial

# =========================
# ENTRENAMIENTO
# =========================
num_episodios = 2000

for episodios in range(num_episodios):

    # Reiniciar entorno
    estado, _ = env.reset()
    terminado = False
    truncado = False

    # Ejecutar episodio
    while not (terminado or truncado):

        # =========================
        # POLÍTICA EPSILON-GREEDY
        # =========================
        if np.random.rand() < epsilon:
            accion = env.action_space.sample()
        else:
            accion = np.argmax(Q[estado])

        nuevo_estado, recompensa, terminado, truncado, _ = env.step(accion)

        # =========================
        # ACTUALIZACIÓN Q-LEARNING
        # =========================
        Q = update_q(Q, estado, accion, recompensa, nuevo_estado, alpha, gamma)

        estado = nuevo_estado

    # Decaimiento de epsilon
    epsilon = max(0.01, epsilon * 0.995)

# Guardar Q-table entrenada
np.save("q_table.npy", Q)

print("Entrenamiento finalizado")
