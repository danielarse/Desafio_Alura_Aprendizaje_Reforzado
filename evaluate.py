import numpy as np
import gymnasium as gym

# =========================
# CREACIÓN DEL ENTORNO
# =========================
env = gym.make("FrozenLake-v1")

# =========================
# CARGAR Q-TABLE ENTRENADA
# =========================
Q = np.load("q_table.npy")

# =========================
# EVALUACIÓN DEL AGENTE
# =========================
num_episodios_eval = 500
exito = 0

for episodio_eval in range(num_episodios_eval):

    estado, _ = env.reset()
    terminado = False
    truncado = False

    # Política greedy (sin exploración)
    while not (terminado or truncado):

        accion = np.argmax(Q[estado])
        estado, recompensa, terminado, truncado, _ = env.step(accion)

    if recompensa == 1:
        exito += 1

# =========================
# RESULTADO FINAL
# =========================
porcentaje = exito / num_episodios_eval * 100

print(f"Porcentaje de éxito del agente: {porcentaje:.2f}%")
