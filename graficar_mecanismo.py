"""
Script para graficar las variables cinemáticas del mecanismo de retorno rápido
Genera gráficas de ángulos, velocidades y aceleraciones en función del tiempo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Configuración de matplotlib para español
plt.rcParams['axes.formatter.use_locale'] = True
plt.rcParams['font.size'] = 10

# ============================================================
# PARÁMETROS DEL MECANISMO
# ============================================================
# Dimensiones geométricas (en metros)
# Pivotes fijos:
d = 0.05  # Distancia entre O1 y O2 (m)
D = 0.15  # Distancia horizontal de O1 a C (m)

# Barras:
r = 0.09  # Longitud de la barra motriz O2-A (m)
R = 0.20  # Longitud FIJA de la barra O1-B (m)
K = 0.09  # Longitud de la barra BC (m)


# Nota: 
# - O1 y O2 son pivotes fijos, separados por distancia d
# - Barra O2-A: longitud r, gira con ángulo φ alrededor de O2
# - Barra O1-B: longitud R (FIJA), gira con ángulo θ alrededor de O1
# - Punto A: es donde se UNEN las barras O2-A y O1-B (pasador deslizante)
# - L: distancia VARIABLE de O1 hasta el punto A (0 < L ≤ R)
# - A está en O2-A: a distancia r de O2
# - A está en O1-B: a distancia L de O1 (desliza sobre la barra)
# - B: extremo de la barra O1-B, a distancia R de O1
# - C: martillo, conectado a B mediante barra BC

# Parámetros del motor
omega_2 = 4.0  # Velocidad angular del motor (rad/s) - constante
alpha_2 = 0.0   # Aceleración angular del motor (rad/s²) - constante

# Parámetros de masa (modelo pequeño impreso en PLA)
# Densidad del PLA: ~1.24 g/cm³
# Grosor estimado de las piezas: ~5mm
rho_pla = 1240  # kg/m³ (densidad del PLA)
espesor = 0.005  # m (5mm)

# Masas de las piezas (estimación para modelo impreso)
m2 = rho_pla * (r * 0.01 * espesor)  # Masa de la barra O2-A (kg) ~0.010 kg
m3 = rho_pla * (R * 0.04 * espesor)  # Masa de la barra O1-B (kg) ~0.040 kg
m4 = rho_pla * (K * 0.01 * espesor)  # Masa de la barra BC (kg) ~0.010 kg
m_martillo = 0.01  # Masa del martillo/herramienta en C (kg) ~0.010kg

# Aceleración de la gravedad
g = 9.81  # m/s²

# Parámetros temporales
t_final = 2 * np.pi / omega_2  # Tiempo para una revolución completa
n_puntos = 10000
t = np.linspace(0, t_final, n_puntos)

# ============================================================
# FUNCIONES PARA CÁLCULOS CINEMÁTICOS
# ============================================================

def calcular_phi(t, omega_2):
    """Ángulo de la barra motriz O2A"""
    return omega_2 * t

def calcular_L(phi, r, d):
    """
    Calcula L (distancia de O1 a A) desde phi
    Ecuación del documento: L = sqrt(r² + d² + 2rd*cos(φ))
    """
    return np.sqrt(r**2 + d**2 + 2*r*d*np.cos(phi))

def calcular_theta_desde_phi(phi, r, d):
    """
    Calcula theta (ángulo de O1B) desde phi (ángulo de O2A)
    Ecuación del documento: sin(θ) = [sqrt(r² + d² + 2rd*cos(φ)) * sin(φ)] / r
    """
    L = calcular_L(phi, r, d)
    sin_theta = (r * np.sin(phi)) / L
    # Limitar valores para evitar errores numéricos
    sin_theta = np.clip(sin_theta, -1, 1)
    return np.arcsin(sin_theta)

def calcular_omega_1(phi, theta, omega_2, r, d):
    """
    Velocidad angular de la barra O1B
    Ecuación del documento: ω₁ = (ω₂ * r / L) * cos(φ - θ)
    """
    L = calcular_L(phi, r, d)
    return (omega_2 * r / L) * np.cos(phi - theta)

def calcular_V_af(phi, theta, omega_2, r):
    """Velocidad relativa a/f"""
    return omega_2 * r * np.sin(phi - theta)

def calcular_alpha_1(phi, theta, omega_1, omega_2, alpha_2, r, d, V_af):
    """
    Aceleración angular de la barra O1B
    Ecuación del documento: α₁ = [ω₂²*r*sin(θ-φ) + α₂*r*cos(θ-φ) + 2*ω₁*V_af] / L
    """
    L = calcular_L(phi, r, d)
    numerador = omega_2**2 * r * np.sin(theta - phi) + alpha_2 * r * np.cos(theta - phi) + 2 * omega_1 * V_af
    return numerador / L

def calcular_beta(theta, R, D, K):
    """Ángulo beta del eslabón BC"""
    sin_beta = (R * np.cos(theta) - D) / K
    # Limitar valores para evitar errores numéricos
    sin_beta = np.clip(sin_beta, -1, 1)
    return np.arcsin(sin_beta)

def calcular_y_c(theta, R, beta, K):
    """Posición vertical del martillo"""
    return R * np.sin(theta) - K * np.cos(beta)

def calcular_omega_3(theta, omega_1, R, K, beta):
    """Velocidad angular del eslabón BC"""
    cos_beta = np.cos(beta)
    # Diagnóstico: verificar dónde cos(beta) es cercano a cero
    indices_problema = np.where(np.abs(cos_beta) < 1e-6)[0]
    if len(indices_problema) > 0:
        print(f"⚠ WARNING en omega_3: cos(β) ≈ 0 en {len(indices_problema)} puntos")
        print(f"  Índices: {indices_problema[:5]}... (primeros 5)")
        print(f"  β en esos puntos: {np.degrees(beta[indices_problema[:5]])} grados")
    
    # Evitar división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        omega_3 = -(omega_1 * R * np.sin(theta)) / (K * cos_beta)
        omega_3 = np.where(np.abs(cos_beta) < 1e-6, 0, omega_3)
    return omega_3

def calcular_dot_y_c(theta, omega_1, R, D, y_c):
    """Velocidad vertical del martillo"""
    numerador = R * omega_1 * (D * np.sin(theta) - y_c * np.cos(theta))
    denominador = R * np.sin(theta) - y_c
    
    # Diagnóstico: verificar dónde el denominador es cercano a cero
    indices_problema = np.where(np.abs(denominador) < 1e-6)[0]
    if len(indices_problema) > 0:
        print(f"⚠ WARNING en dot_y_c: (R*sin(θ) - y_c) ≈ 0 en {len(indices_problema)} puntos")
        print(f"  Índices: {indices_problema[:5]}... (primeros 5)")
        print(f"  R*sin(θ) = {R * np.sin(theta[indices_problema[:5]])}")
        print(f"  y_c = {y_c[indices_problema[:5]]}")
    
    # Evitar división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        dot_y_c = numerador / denominador
        dot_y_c = np.where(np.abs(denominador) < 1e-6, 0, dot_y_c)
    return dot_y_c

def calcular_alpha_3(theta, omega_1, alpha_1, omega_3, R, K, beta):
    """Aceleración angular del eslabón BC"""
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    numerador = omega_1**2 * R * np.cos(theta) + alpha_1 * R * np.sin(theta) - omega_3**2 * K * sin_beta
    # Evitar división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha_3 = numerador / (K * cos_beta)
        alpha_3 = np.where(np.abs(cos_beta) < 1e-6, 0, alpha_3)
    return alpha_3

def calcular_a_c_lazo_vectorial(theta, omega_1, alpha_1, R, D, y_c, dot_y_c):
    """
    Aceleración vertical del martillo usando el método del lazo vectorial
    Ecuación del documento: 
    ÿ_c = R * [α₁*A*D_en - ω₁²*B*D_en - ω₁*ẏ_c*(R*sin(θ)*cos(θ) - D*sin(θ))] / D_en²
    donde:
    A = D*sin(θ) - y_c*cos(θ)
    B = D*cos(θ) + y_c*sin(θ)
    D_en = R*sin(θ) - y_c
    """
    A = D * np.sin(theta) - y_c * np.cos(theta)
    B = D * np.cos(theta) + y_c * np.sin(theta)
    D_en = R * np.sin(theta) - y_c
    
    # Diagnóstico: verificar dónde D_en es cercano a cero
    indices_problema = np.where(np.abs(D_en) < 1e-6)[0]
    if len(indices_problema) > 0:
        print(f"⚠ WARNING en a_c: D_en = (R*sin(θ) - y_c) ≈ 0 en {len(indices_problema)} puntos")
        print(f"  Índices: {indices_problema[:5]}... (primeros 5)")
        print(f"  Esto significa que el martillo alcanza el límite de su carrera")
    
    numerador = (alpha_1 * A * D_en - 
                 omega_1**2 * B * D_en - 
                 omega_1 * dot_y_c * (R * np.sin(theta) * np.cos(theta) - D * np.sin(theta)))
    
    # Evitar división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        a_c = R * numerador / (D_en**2)
        a_c = np.where(np.abs(D_en) < 1e-6, 0, a_c)
    
    return a_c

# ============================================================
# CÁLCULOS PRINCIPALES
# ============================================================
print("Calculando variables cinemáticas del mecanismo...")

# Ángulos
phi = calcular_phi(t, omega_2)
theta = calcular_theta_desde_phi(phi, r, d)
L_variable = calcular_L(phi, r, d)  # L varía con phi
beta = calcular_beta(theta, R, D, K)

# Posición
y_c = calcular_y_c(theta, R, beta, K)

# Velocidades
omega_1 = calcular_omega_1(phi, theta, omega_2, r, d)
V_af = calcular_V_af(phi, theta, omega_2, r)
omega_3 = calcular_omega_3(theta, omega_1, R, K, beta)
dot_y_c = calcular_dot_y_c(theta, omega_1, R, D, y_c)

# Aceleraciones
alpha_1 = calcular_alpha_1(phi, theta, omega_1, omega_2, alpha_2, r, d, V_af)
# Calcular a_c usando el método del lazo vectorial (Sección 3 del documento)
a_c = calcular_a_c_lazo_vectorial(theta, omega_1, alpha_1, R, D, y_c, dot_y_c)
# También calcular alpha_3 para referencia (aunque no se usa en a_c con lazo vectorial)
alpha_3 = calcular_alpha_3(theta, omega_1, alpha_1, omega_3, R, K, beta)

print("Cálculos completados. Generando gráficas...")

# ============================================================
# GENERACIÓN DE GRÁFICAS
# ============================================================

# Figura 1: Ángulos
fig1 = plt.figure(figsize=(14, 10))
gs1 = GridSpec(3, 2, figure=fig1, hspace=0.3, wspace=0.3)

# Phi vs tiempo
ax1 = fig1.add_subplot(gs1[0, 0])
ax1.plot(t, np.degrees(phi), 'b-', linewidth=2)
ax1.set_xlabel('Tiempo (s)')
ax1.set_ylabel('φ (grados)')
ax1.set_title('Ángulo de la barra motriz O₂A')
ax1.grid(True, alpha=0.3)

# Theta vs tiempo
ax2 = fig1.add_subplot(gs1[0, 1])
ax2.plot(t, np.degrees(theta), 'r-', linewidth=2)
ax2.set_xlabel('Tiempo (s)')
ax2.set_ylabel('θ (grados)')
ax2.set_title('Ángulo de la barra O₁B')
ax2.grid(True, alpha=0.3)

# Beta vs tiempo y L vs tiempo (doble eje Y)
ax3 = fig1.add_subplot(gs1[1, 0])
ax3.plot(t, np.degrees(beta), 'g-', linewidth=2, label='β')
ax3.set_xlabel('Tiempo (s)')
ax3.set_ylabel('β (grados)', color='g')
ax3.set_title('Ángulo β y distancia L (O₁ a A)')
ax3.tick_params(axis='y', labelcolor='g')
ax3.grid(True, alpha=0.3)

# L vs tiempo (distancia de O1 al punto de unión A)
ax3b = ax3.twinx()
ax3b.plot(t, L_variable * 1000, 'orange', linewidth=2, alpha=0.8, label='L')
ax3b.set_ylabel('L: O₁→A (mm)', color='orange')
ax3b.tick_params(axis='y', labelcolor='orange')
ax3b.axhline(y=R*1000, color='red', linestyle=':', alpha=0.5, label=f'R={R*1000:.0f}mm')

# Posición del martillo vs tiempo
ax4 = fig1.add_subplot(gs1[1, 1])
ax4.plot(t, y_c * 1000, 'm-', linewidth=2)  # Convertir a mm
ax4.set_xlabel('Tiempo (s)')
ax4.set_ylabel('y_c (mm)')
ax4.set_title('Posición vertical del martillo')
ax4.grid(True, alpha=0.3)

# L vs Phi (muestra cómo varía L con el ángulo de entrada)
ax5 = fig1.add_subplot(gs1[2, 0])
ax5.plot(np.degrees(phi), L_variable * 1000, 'orange', linewidth=2)
ax5.set_xlabel('φ (grados)')
ax5.set_ylabel('L: O₁→A (mm)')
ax5.set_title('Distancia L (O₁ al punto A) vs φ')
ax5.axhline(y=R*1000, color='red', linestyle='--', alpha=0.5, label=f'R={R*1000:.0f}mm (máx)')
ax5.grid(True, alpha=0.3)
ax5.legend()

# Beta vs Theta
ax6 = fig1.add_subplot(gs1[2, 1])
ax6.plot(np.degrees(theta), np.degrees(beta), 'orange', linewidth=2)
ax6.set_xlabel('θ (grados)')
ax6.set_ylabel('β (grados)')
ax6.set_title('Relación β vs θ')
ax6.grid(True, alpha=0.3)

plt.suptitle('Análisis de Ángulos y Posición del Mecanismo', fontsize=14, fontweight='bold')

# Figura 2: Velocidades
fig2 = plt.figure(figsize=(14, 10))
gs2 = GridSpec(3, 2, figure=fig2, hspace=0.3, wspace=0.3)

# Omega_1 vs tiempo
ax7 = fig2.add_subplot(gs2[0, 0])
ax7.plot(t, omega_1, 'b-', linewidth=2)
ax7.set_xlabel('Tiempo (s)')
ax7.set_ylabel('ω₁ (rad/s)')
ax7.set_title('Velocidad angular de O₁B')
ax7.grid(True, alpha=0.3)
ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Omega_3 vs tiempo
ax8 = fig2.add_subplot(gs2[0, 1])
ax8.plot(t, omega_3, 'r-', linewidth=2)
ax8.set_xlabel('Tiempo (s)')
ax8.set_ylabel('ω₃ (rad/s)')
ax8.set_title('Velocidad angular del eslabón BC')
ax8.grid(True, alpha=0.3)
ax8.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Velocidad del martillo vs tiempo
ax9 = fig2.add_subplot(gs2[1, 0])
ax9.plot(t, dot_y_c * 1000, 'g-', linewidth=2)  # Convertir a mm/s
ax9.set_xlabel('Tiempo (s)')
ax9.set_ylabel('ẏ_c (mm/s)')
ax9.set_title('Velocidad vertical del martillo')
ax9.grid(True, alpha=0.3)
ax9.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# V_af vs tiempo
ax10 = fig2.add_subplot(gs2[1, 1])
ax10.plot(t, V_af * 1000, 'm-', linewidth=2)  # Convertir a mm/s
ax10.set_xlabel('Tiempo (s)')
ax10.set_ylabel('V_{a/f} (mm/s)')
ax10.set_title('Velocidad relativa a/f')
ax10.grid(True, alpha=0.3)
ax10.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Comparación de velocidades angulares
ax11 = fig2.add_subplot(gs2[2, :])
ax11.plot(t, omega_1, 'b-', linewidth=2, label='ω₁ (O₁B)')
ax11.plot(t, omega_3, 'r-', linewidth=2, label='ω₃ (BC)')
ax11.plot(t, np.ones_like(t) * omega_2, 'k--', linewidth=2, label='ω₂ (motor)')
ax11.set_xlabel('Tiempo (s)')
ax11.set_ylabel('Velocidad angular (rad/s)')
ax11.set_title('Comparación de velocidades angulares')
ax11.legend()
ax11.grid(True, alpha=0.3)
ax11.axhline(y=0, color='k', linestyle='--', alpha=0.5)

plt.suptitle('Análisis de Velocidades del Mecanismo', fontsize=14, fontweight='bold')

# Figura 3: Aceleraciones
fig3 = plt.figure(figsize=(14, 10))
gs3 = GridSpec(2, 2, figure=fig3, hspace=0.3, wspace=0.3)

# Alpha_1 vs tiempo
ax12 = fig3.add_subplot(gs3[0, 0])
ax12.plot(t, alpha_1, 'b-', linewidth=2)
ax12.set_xlabel('Tiempo (s)')
ax12.set_ylabel('α₁ (rad/s²)')
ax12.set_title('Aceleración angular de O₁B')
ax12.grid(True, alpha=0.3)
ax12.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Alpha_3 vs tiempo
ax13 = fig3.add_subplot(gs3[0, 1])
ax13.plot(t, alpha_3, 'r-', linewidth=2)
ax13.set_xlabel('Tiempo (s)')
ax13.set_ylabel('α₃ (rad/s²)')
ax13.set_title('Aceleración angular del eslabón BC')
ax13.grid(True, alpha=0.3)
ax13.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Aceleración del martillo vs tiempo
ax14 = fig3.add_subplot(gs3[1, 0])
ax14.plot(t, a_c, 'g-', linewidth=2)
ax14.set_xlabel('Tiempo (s)')
ax14.set_ylabel('a_c (m/s²)')
ax14.set_title('Aceleración vertical del martillo')
ax14.grid(True, alpha=0.3)
ax14.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Comparación de aceleraciones angulares
ax15 = fig3.add_subplot(gs3[1, 1])
ax15.plot(t, alpha_1, 'b-', linewidth=2, label='α₁ (O₁B)')
ax15.plot(t, alpha_3, 'r-', linewidth=2, label='α₃ (BC)')
ax15.set_xlabel('Tiempo (s)')
ax15.set_ylabel('Aceleración angular (rad/s²)')
ax15.set_title('Comparación de aceleraciones angulares')
ax15.legend()
ax15.grid(True, alpha=0.3)
ax15.axhline(y=0, color='k', linestyle='--', alpha=0.5)

plt.suptitle('Análisis de Aceleraciones del Mecanismo', fontsize=14, fontweight='bold')

# ============================================================
# ESTADÍSTICAS
# ============================================================
print("\n" + "="*60)
print("ESTADÍSTICAS DEL MECANISMO")
print("="*60)
print(f"\nParámetros geométricos:")
print(f"  d (distancia O1-O2) = {d*1000:.1f} mm")
print(f"  r (longitud barra motriz O2-A) = {r*1000:.1f} mm")
print(f"  R (longitud barra O1-B, FIJA) = {R*1000:.1f} mm")
print(f"  K (longitud barra BC) = {K*1000:.1f} mm")
print(f"  D (distancia horizontal O1 a C) = {D*1000:.1f} mm")
print(f"\nParámetros de masa (modelo PLA):")
print(f"  Densidad PLA = {rho_pla:.0f} kg/m³")
print(f"  Espesor de piezas = {espesor*1000:.1f} mm")
print(f"  m₂ (barra O2-A) = {m2*1000:.2f} g")
print(f"  m₃ (barra O1-B) = {m3*1000:.2f} g")
print(f"  m₄ (barra BC) = {m4*1000:.2f} g")
print(f"  m_martillo (herramienta en C) = {m_martillo*1000:.1f} g")
print(f"  Masa total del mecanismo = {(m2+m3+m4+m_martillo)*1000:.2f} g")
print(f"\nDistancia L (O1 a punto A, VARIABLE):")
print(f"  L máx = {np.max(L_variable)*1000:.2f} mm (A cerca del extremo B)")
print(f"  L mín = {np.min(L_variable)*1000:.2f} mm (A cerca de O1)")
print(f"  Rango de deslizamiento = {(np.max(L_variable) - np.min(L_variable))*1000:.2f} mm")
print(f"\nParámetros del motor:")
print(f"  ω₂ = {omega_2:.2f} rad/s ({np.degrees(omega_2):.2f} °/s)")
print(f"  α₂ = {alpha_2:.2f} rad/s²")
print(f"\nCarrera del martillo:")
print(f"  y_c máx = {np.max(y_c)*1000:.2f} mm")
print(f"  y_c mín = {np.min(y_c)*1000:.2f} mm")
print(f"  Carrera = {(np.max(y_c) - np.min(y_c))*1000:.2f} mm")
print(f"\nVelocidad del martillo:")
print(f"  ẏ_c máx = {np.max(dot_y_c)*1000:.2f} mm/s")
print(f"  ẏ_c mín = {np.min(dot_y_c)*1000:.2f} mm/s")
print(f"\nAceleración del martillo:")
print(f"  a_c máx = {np.max(a_c):.2f} m/s²")
print(f"  a_c mín = {np.min(a_c):.2f} m/s²")
print(f"\nVelocidades angulares:")
print(f"  ω₁ máx = {np.max(omega_1):.2f} rad/s")
print(f"  ω₁ mín = {np.min(omega_1):.2f} rad/s")
print(f"  ω₃ máx = {np.max(omega_3):.2f} rad/s")
print(f"  ω₃ mín = {np.min(omega_3):.2f} rad/s")
print("="*60 + "\n")

# Guardar figuras
print("Guardando gráficas...")
fig1.savefig('graficas_angulos_posicion.png', dpi=300, bbox_inches='tight')
fig2.savefig('graficas_velocidades.png', dpi=300, bbox_inches='tight')
fig3.savefig('graficas_aceleraciones.png', dpi=300, bbox_inches='tight')
print("Gráficas guardadas exitosamente!")
print(f"  - graficas_angulos_posicion.png")
print(f"  - graficas_velocidades.png")
print(f"  - graficas_aceleraciones.png")

# plt.show()  # Comentado para ejecución no interactiva
print("\nProceso completado. Las gráficas están listas para usar.")
