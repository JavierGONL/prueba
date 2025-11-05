"""
Script para animar el mecanismo de retorno rápido
Genera una animación del movimiento del mecanismo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches

# ============================================================
# PARÁMETROS DEL MECANISMO
# ============================================================
# Pivotes fijos:
d = 0.10  # Distancia entre O1 y O2 (m)

# Barras:
r = 0.05  # Longitud de la barra motriz O2-A (m)
R = 0.20  # Longitud FIJA de la barra O1-B (m) ← LA BARRA ES RÍGIDA
K = 0.10  # Longitud de la barra BC (m)
D = 0.15  # Distancia horizontal de O1 a C (m)

# Nota: 
# - Barra O1-B: longitud R (FIJA), gira con ángulo θ
# - Punto A: se une a las barras (pasador deslizante sobre O1-B)
# - L: distancia VARIABLE de O1 hasta A (calculada, varía de ~40mm a ~200mm)
# - B: extremo de la barra O1-B, a distancia R de O1

omega_2 = 200.0  # Velocidad angular del motor (rad/s)

# Posiciones de los pivotes fijos
O1 = np.array([0, 0])
O2 = np.array([d, 0])

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def calcular_L(phi, r, d):
    """
    Calcula L (distancia de O1 a A) desde phi
    Ecuación: L = sqrt(r² + d² + 2rd*cos(φ))
    """
    return np.sqrt(r**2 + d**2 + 2*r*d*np.cos(phi))

def calcular_posicion_A(phi):
    """
    Posición del punto A (donde se unen las barras)
    A está en la barra motriz O2-A
    """
    return O2 + r * np.array([np.cos(phi), np.sin(phi)])

def calcular_theta_desde_phi(phi, r, d):
    """
    Calcula theta (ángulo de O1-B) desde phi
    Ecuación: sin(θ) = [L * sin(φ)] / r
    """
    L = calcular_L(phi, r, d)
    sin_theta = (r * np.sin(phi)) / L
    sin_theta = np.clip(sin_theta, -1, 1)
    return np.arcsin(sin_theta)

def calcular_posicion_B(theta):
    """
    Posición del punto B (extremo de la barra O1-B)
    B está a distancia FIJA R de O1, en dirección theta
    """
    return O1 + R * np.array([np.cos(theta), np.sin(theta)])

def calcular_posicion_C(theta, beta):
    """Posición del punto C (martillo)"""
    B = calcular_posicion_B(theta)
    return B + K * np.array([-np.sin(beta), -np.cos(beta)])

def calcular_beta_desde_theta(theta):
    """Calcula beta desde theta"""
    sin_beta = (R * np.cos(theta) - D) / K
    sin_beta = np.clip(sin_beta, -1, 1)
    return np.arcsin(sin_beta)

# ============================================================
# CONFIGURACIÓN DE LA ANIMACIÓN
# ============================================================

n_frames = 200
t_final = 2 * np.pi / omega_2
t_vals = np.linspace(0, t_final, n_frames)

# Pre-calcular todas las posiciones
phi_vals = omega_2 * t_vals

# Calcular theta desde phi usando las ecuaciones correctas
theta_vals = np.array([calcular_theta_desde_phi(phi, r, d) for phi in phi_vals])
A_positions = np.array([calcular_posicion_A(phi) for phi in phi_vals])
L_vals = np.array([calcular_L(phi, r, d) for phi in phi_vals])

beta_vals = np.array([calcular_beta_desde_theta(theta) for theta in theta_vals])

# Calcular posiciones (A_positions ya calculado arriba)
B_positions = np.array([calcular_posicion_B(theta) for theta in theta_vals])
C_positions = np.array([calcular_posicion_C(theta, beta) for theta, beta in zip(theta_vals, beta_vals)])

# ============================================================
# CREAR LA FIGURA
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel izquierdo: Animación del mecanismo
ax1.set_xlim(-0.15, 0.35)
ax1.set_ylim(-0.4, 0.15)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Mecanismo de Retorno Rápido', fontsize=14, fontweight='bold')

# Elementos gráficos del mecanismo
# Pivotes fijos
pivot_O1 = Circle(O1, 0.01, color='black', zorder=10)
pivot_O2 = Circle(O2, 0.01, color='black', zorder=10)
ax1.add_patch(pivot_O1)
ax1.add_patch(pivot_O2)
ax1.text(O1[0]-0.02, O1[1]+0.02, 'O₁', fontsize=10, fontweight='bold')
ax1.text(O2[0]+0.02, O2[1]+0.02, 'O₂', fontsize=10, fontweight='bold')

# Barra motriz O2-A
line_O2A, = ax1.plot([], [], 'b-', linewidth=4, label='Barra motriz (O₂A)')
circle_A = Circle([0, 0], 0.008, color='blue', zorder=5)
ax1.add_patch(circle_A)

# Barra O1-B (con punto A coincidente)
line_O1B, = ax1.plot([], [], 'r-', linewidth=4, label='Barra oscilante (O₁B)')
circle_B = Circle([0, 0], 0.008, color='red', zorder=5)
ax1.add_patch(circle_B)

# Barra B-C (eslabón del martillo)
line_BC, = ax1.plot([], [], 'g-', linewidth=4, label='Eslabón BC')
circle_C = Circle([0, 0], 0.015, color='darkgreen', zorder=5)
ax1.add_patch(circle_C)

# Martillo (rectángulo en C)
martillo = Rectangle([0, 0], 0.04, 0.06, color='gray', alpha=0.7, zorder=3)
ax1.add_patch(martillo)

# Trayectoria del martillo
trajectory_line, = ax1.plot([], [], 'k--', linewidth=1, alpha=0.5, label='Trayectoria martillo')

# Texto con información
info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                     verticalalignment='top', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax1.legend(loc='lower right', fontsize=8)

# Panel derecho: Gráficas de velocidad y aceleración
ax2.set_xlabel('Tiempo (s)')
ax2.set_ylabel('Velocidad (m/s)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.grid(True, alpha=0.3)

# Calcular velocidades para el gráfico
y_c_vals = np.array([C[1] for C in C_positions])
velocidad = np.gradient(y_c_vals, t_vals)

line_vel, = ax2.plot([], [], 'b-', linewidth=2, label='Velocidad')
point_vel, = ax2.plot([], [], 'bo', markersize=8)

# Eje secundario para aceleración
ax3 = ax2.twinx()
ax3.set_ylabel('Aceleración (m/s²)', color='red')
ax3.tick_params(axis='y', labelcolor='red')

aceleracion = np.gradient(velocidad, t_vals)
line_acc, = ax3.plot([], [], 'r-', linewidth=2, label='Aceleración')
point_acc, = ax3.plot([], [], 'ro', markersize=8)

ax2.set_title('Cinemática del martillo', fontsize=12, fontweight='bold')

# Leyenda combinada
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

# ============================================================
# FUNCIÓN DE ACTUALIZACIÓN DE LA ANIMACIÓN
# ============================================================

def init():
    """Inicializar la animación"""
    line_O2A.set_data([], [])
    line_O1B.set_data([], [])
    line_BC.set_data([], [])
    trajectory_line.set_data([], [])
    line_vel.set_data([], [])
    line_acc.set_data([], [])
    return line_O2A, line_O1B, line_BC, trajectory_line, line_vel, line_acc

def update(frame):
    """Actualizar cada frame de la animación"""
    # Posiciones actuales
    A = A_positions[frame]
    B = B_positions[frame]
    C = C_positions[frame]
    
    # Actualizar barras
    line_O2A.set_data([O2[0], A[0]], [O2[1], A[1]])
    line_O1B.set_data([O1[0], B[0]], [O1[1], B[1]])
    line_BC.set_data([B[0], C[0]], [B[1], C[1]])
    
    # Actualizar círculos
    circle_A.center = A
    circle_B.center = B
    circle_C.center = C
    
    # Actualizar martillo (rectángulo)
    martillo.set_xy([C[0] - 0.02, C[1] - 0.06])
    
    # Actualizar trayectoria (últimos 50 puntos)
    start_idx = max(0, frame - 50)
    traj_x = C_positions[start_idx:frame+1, 0]
    traj_y = C_positions[start_idx:frame+1, 1]
    trajectory_line.set_data(traj_x, traj_y)
    
    # Actualizar texto de información
    t = t_vals[frame]
    phi = phi_vals[frame]
    theta = theta_vals[frame]
    beta = beta_vals[frame]
    v = velocidad[frame]
    a = aceleracion[frame]
    
    info_str = (f't = {t:.3f} s\n'
                f'φ = {np.degrees(phi):.1f}°\n'
                f'θ = {np.degrees(theta):.1f}°\n'
                f'β = {np.degrees(beta):.1f}°\n'
                f'v = {v*1000:.1f} mm/s\n'
                f'a = {a:.2f} m/s²')
    info_text.set_text(info_str)
    
    # Actualizar gráficas
    line_vel.set_data(t_vals[:frame+1], velocidad[:frame+1])
    point_vel.set_data([t], [v])
    
    line_acc.set_data(t_vals[:frame+1], aceleracion[:frame+1])
    point_acc.set_data([t], [a])
    
    # Ajustar límites de las gráficas
    if frame > 0:
        ax2.set_xlim(0, t_vals[frame] + 0.1)
        ax2.set_ylim(min(velocidad[:frame+1])*1.1, max(velocidad[:frame+1])*1.1)
        ax3.set_ylim(min(aceleracion[:frame+1])*1.1, max(aceleracion[:frame+1])*1.1)
    
    return (line_O2A, line_O1B, line_BC, circle_A, circle_B, circle_C, 
            martillo, trajectory_line, info_text, line_vel, point_vel, 
            line_acc, point_acc)

# ============================================================
# CREAR Y GUARDAR LA ANIMACIÓN
# ============================================================

print("Creando animación del mecanismo...")
print(f"Generando {n_frames} frames...")

anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, 
                     interval=50, blit=True, repeat=True)

# Guardar como GIF
print("Guardando animación como GIF (esto puede tomar unos minutos)...")
writer = PillowWriter(fps=20)
anim.save('animacion_mecanismo.gif', writer=writer, dpi=100)

print("Animación guardada como: animacion_mecanismo.gif")
print(f"Duración: {t_final:.2f} segundos")
print(f"FPS: 20")
print(f"Total frames: {n_frames}")

plt.tight_layout()
# plt.show()

print("\nAnimación completada exitosamente!")
