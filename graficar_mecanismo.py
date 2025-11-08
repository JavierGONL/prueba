"""
Script para graficar las variables cinemáticas del mecanismo de retorno rápido
Genera gráficas de ángulos, velocidades y aceleraciones en función del tiempo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Configuración de matplotlib para español
plt.rcParams['axes.formatter.use_locale'] = True
plt.rcParams['font.size'] = 10

# ============================================================
# PARÁMETROS DEL MECANISMO
# ============================================================
# Dimensiones geométricas (en metros)
# Pivotes fijos:
d = 0.10  # Distancia entre O1 y O2 (m)
# δ (delta) solicitado: 6.65 mm => 0.00665 m
D = 0.00665  # Distancia horizontal de O1 a C (m)

# Barras:
r = 0.07  # Longitud de la barra motriz O2-A (m)
R = 0.2  # Longitud FIJA de la barra O1-B (m)
K = 0.07  # Longitud de la barra BC (m)

# Anchos (en el plano) usados para masas e inercias (consistentes con estimaciones de masa)
b_R = 0.02  # ancho de la barra O1-B (m)
b_r = 0.02  # ancho de la barra O2-A (m)


# Parámetros del motor
omega_2 = 4 # Velocidad angular del motor (rad/s) - constante
alpha_2 = 0   # Aceleración angular del motor (rad/s²) - constante

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
n_puntos = 1000
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
    Ecuación del documento: sin(θ) = (r * sin(φ)) / L, con L = sqrt(r² + d² + 2rd*cos(φ))
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
    Aceleración vertical del martillo usando el método del lazo vectorial.
    Fórmula (calculos_unificados):
    ÿ_c = R * { [ α₁ A − ω₁² B − ω₁ ẏ_c cosθ ] D_en − ω₁ A ( ẏ_c + R ω₁ cosθ ) } / D_en²
    con A = D sinθ − y_c cosθ, B = D cosθ + y_c sinθ, D_en = R sinθ − y_c.
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
    
    numerador = (
        (alpha_1 * A - omega_1**2 * B - omega_1 * dot_y_c * np.cos(theta)) * D_en
        - omega_1 * A * (dot_y_c + R * omega_1 * np.cos(theta))
    )
    
    # Evitar división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        a_c = R * numerador / (D_en**2)
        a_c = np.where(np.abs(D_en) < 1e-6, 0, a_c)
    
    return a_c

# ============================================================
# BÚSQUEDA DEL D ÓPTIMO (opcional)
# ============================================================
def evaluar_configuracion_D(D_cand, t_eval):
    """Evalúa singularidades y carrera para un valor de D dado.
    Retorna dict con: safe (bool), min_cosb, min_den, stroke.
    """
    phi_e = calcular_phi(t_eval, omega_2)
    theta_e = calcular_theta_desde_phi(phi_e, r, d)
    beta_e = calcular_beta(theta_e, R, D_cand, K)
    y_e = calcular_y_c(theta_e, R, beta_e, K)
    # Para métricas de estabilidad
    cosb = np.cos(beta_e)
    Den = R * np.sin(theta_e) - y_e
    min_cosb = np.min(np.abs(cosb))
    min_den = np.min(np.abs(Den))
    stroke = np.max(y_e) - np.min(y_e)
    return {
        'safe': True,
        'min_cosb': float(min_cosb),
        'min_den': float(min_den),
        'stroke': float(stroke),
    }

def buscar_D_optimo(D_min=0.05, D_max=0.18, n_samples=60, eps_cosb=5e-2, eps_den=1e-3):
    """Barre D en [D_min, D_max] para maximizar la carrera evitando singularidades.
    Criterio de viabilidad: min|cosβ| >= eps_cosb y min|R sinθ - y_c| >= eps_den.
    Devuelve (D_opt, info_dict) o (None, None) si no hay candidato viable.
    """
    t_eval = np.linspace(0, t_final, 360)  # muestreo grueso para optimización
    Ds = np.linspace(D_min, D_max, n_samples)
    candidatos = []
    for D_c in Ds:
        info = evaluar_configuracion_D(D_c, t_eval)
        if info['min_cosb'] >= eps_cosb and info['min_den'] >= eps_den:
            candidatos.append((D_c, info))
    if not candidatos:
        return None, None
    # Maximizar carrera; desempatar por mayor min_cosb y min_den
    candidatos.sort(key=lambda x: (x[1]['stroke'], x[1]['min_cosb'], x[1]['min_den']), reverse=True)
    return candidatos[0][0], candidatos[0][1]

# ============================================================
# CÁLCULOS PRINCIPALES
# ============================================================
# Opcional: buscar D óptimo antes de calcular todo
realizar_optimizacion_D = True
if realizar_optimizacion_D:
    print("Buscando D óptimo (evitando singularidades y maximizando carrera)...")
    D_opt, info_opt = buscar_D_optimo(D_min=0.05, D_max=0.18, n_samples=60, eps_cosb=5e-2, eps_den=1e-3)
    if D_opt is None:
        # Relajar umbrales si no se encontró candidato
        print("  No se encontró D con umbrales estrictos; relajando criterios...")
        D_opt, info_opt = buscar_D_optimo(D_min=0.05, D_max=0.18, n_samples=60, eps_cosb=1e-2, eps_den=5e-4)
    if D_opt is not None:
        print(f"  D recomendado: {D_opt:.5f} m  |  carrera≈{info_opt['stroke']:.4f} m, min|cosβ|≈{info_opt['min_cosb']:.3f}, min|Den|≈{info_opt['min_den']:.4f}")
        D = D_opt
    else:
        print("  No se pudo determinar un D seguro; se usará el valor actual.")

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
# FUERZAS (BÁSICO: MARTILLO Y ARTICULACIÓN B)
# ============================================================
# Fuerza de contacto en el martillo (signo positivo hacia arriba)
# C(t) = m_c (g - a_c)
C_t = m_martillo * (g - a_c)

# Aceleración del centroide del eslabón BC 
"""
Centroide del eslabón BC ubicado a K/2 desde B hacia C.
Vector r_BG (de B a G): r_BG = (-(K/2) sinβ, -(K/2) cosβ)
Aceleración de B: a_B = R*[ -α1 sinθ - ω1^2 cosθ,  α1 cosθ - ω1^2 sinθ]
Cin emática rígida: a_G = a_B + α3×r_BG - ω3^2 r_BG, con α3×r = α3[-r_y, r_x]
"""
r_BG_x = -(K/2.0) * np.sin(beta)
r_BG_y = -(K/2.0) * np.cos(beta)

# Aceleración del punto B
a_Bx = -R * (alpha_1 * np.sin(theta) + (omega_1**2) * np.cos(theta))
a_By =  R * (alpha_1 * np.cos(theta) - (omega_1**2) * np.sin(theta))

# Aceleración del centroide G de BC
a_BC_x = a_Bx + (-alpha_3 * r_BG_y) - (omega_3**2) * r_BG_x
a_BC_y = a_By + ( alpha_3 * r_BG_x) - (omega_3**2) * r_BG_y


# Fuerzas en la articulación B del eslabón BC
B_x = m4 * a_BC_x
B_y = m4 * (g + a_BC_y) + C_t

# ============================================================
# REACCIONES EN O1, FUERZA EN A Y PAR DEL MOTOR τ(t)
#   Según calculos_unificados: sistema lineal en (O1x, O1y, A)
#   y fórmula cerrada para τ: τ = I_O2*α2 + r*A*cos(φ-θ) + (r/2)*m2*g*cosφ
# ============================================================

# Aceleración centroidal del eslabón O1-B (a una distancia λ desde O1)
lam = 0.1093458  # Parámetro λ proporcionado (m)
a_o1x = -lam * (alpha_1 * np.sin(theta) + (omega_1**2) * np.cos(theta))
a_o1y =  lam * (alpha_1 * np.cos(theta) - (omega_1**2) * np.sin(theta))

# Inercias (aprox. prisma rectangular en el plano, eje z)
Ibar_o1 = (1.0/12.0) * m3 * (R**2 + b_R**2)  # respecto al centroide de O1-B
Ibar_o2 = (1.0/12.0) * m2 * (r**2 + b_r**2)  # respecto al centroide de O2-A
I_O2 = Ibar_o2 + m2 * (r/2.0)**2            # respecto a O2 (teorema ejes paralelos)

# Parámetro lambda (distancia de O1 al centroide de O1-B)
# Ya fijado arriba como valor solicitado

# Resolver el sistema 3x3 por instante para O1x, O1y, A
O1x = np.zeros_like(t)
O1y = np.zeros_like(t)
A_reac = np.zeros_like(t)

for i in range(len(t)):
    th = theta[i]
    L_i = L_variable[i]
    # Matriz de coeficientes para [O1x, O1y, A]
    M = np.array([
        [1.0, 0.0, -np.sin(th)],
        [0.0, 1.0,  np.cos(th)],
        [lam*np.sin(th), -lam*np.cos(th), (L_i - lam)]
    ])
    # Término independiente
    rhs1 = m3 * a_o1x[i] + B_x[i]
    rhs2 = m3 * (a_o1y[i] + g) + B_y[i]
    rhs3 = Ibar_o1 * alpha_1[i] - lam * (B_x[i]*np.sin(th) - B_y[i]*np.cos(th))
    bvec = np.array([rhs1, rhs2, rhs3])

    try:
        sol = np.linalg.solve(M, bvec)
    except np.linalg.LinAlgError:
        # En casos degenerados (singularidad), usar solución de mínimos cuadrados
        sol, *_ = np.linalg.lstsq(M, bvec, rcond=None)

    O1x[i], O1y[i], A_reac[i] = sol

# Par del motor τ(t)
tau = I_O2 * alpha_2 + r * A_reac * np.cos(phi - theta) + (r/2.0) * m2 * g * np.cos(phi)

# ============================================================
# GENERACIÓN DE GRÁFICAS (una figura por curva)
# ============================================================

SALIDAS_DIR = os.path.join(os.path.dirname(__file__), 'salidas')
os.makedirs(SALIDAS_DIR, exist_ok=True)

def guardar_fig(x, y, xlabel, ylabel, titulo, nombre, color='b-', convertir_mm=False):
    """Guarda una figura individual dentro de la carpeta 'salidas'.
    nombre: nombre base del archivo (se guarda en SALIDAS_DIR).
    Si convertir_mm=True multiplica y por 1000 para mostrar en mm.
    """
    fig = plt.figure(figsize=(6,4))
    y_plot = y * 1000.0 if convertir_mm else y
    plt.plot(x, y_plot, color, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.grid(True, alpha=0.3)
    ruta = os.path.join(SALIDAS_DIR, nombre)
    fig.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Ángulos y posición
guardar_fig(t, phi, 'Tiempo (s)', 'φ (rad)', 'Ángulo φ vs tiempo', 'phi_vs_t.png')
guardar_fig(t, theta, 'Tiempo (s)', 'θ (rad)', 'Ángulo θ vs tiempo', 'theta_vs_t.png', color='r-')
guardar_fig(t, beta, 'Tiempo (s)', 'β (rad)', 'Ángulo β vs tiempo', 'beta_vs_t.png', color='g-')
guardar_fig(t, L_variable, 'Tiempo (s)', 'L (m)', 'Distancia L vs tiempo', 'L_vs_t.png', color='orange')
guardar_fig(t, y_c, 'Tiempo (s)', 'y_c (m)', 'Posición vertical martillo vs tiempo', 'yc_vs_t.png', color='m-')
guardar_fig(phi, L_variable, 'φ (rad)', 'L (m)', 'L vs φ', 'L_vs_phi.png', color='orange')
guardar_fig(theta, beta, 'θ (rad)', 'β (rad)', 'β vs θ', 'beta_vs_theta.png', color='orange')

# Velocidades
guardar_fig(t, omega_1, 'Tiempo (s)', 'ω₁ (rad/s)', 'Velocidad angular ω₁', 'omega1_vs_t.png')
guardar_fig(t, omega_3, 'Tiempo (s)', 'ω₃ (rad/s)', 'Velocidad angular ω₃', 'omega3_vs_t.png', color='r-')
guardar_fig(t, dot_y_c, 'Tiempo (s)', 'ẏ_c (m/s)', 'Velocidad vertical martillo', 'dot_yc_vs_t.png', color='g-')
guardar_fig(t, V_af, 'Tiempo (s)', 'V_{a/f} (m/s)', 'Velocidad relativa a/f', 'Vaf_vs_t.png', color='m-')
# Comparación velocidades angulares
fig_cmp_vel = plt.figure(figsize=(6,4))
plt.plot(t, omega_1, 'b-', linewidth=2, label='ω₁')
plt.plot(t, omega_3, 'r-', linewidth=2, label='ω₃')
plt.plot(t, np.ones_like(t)*omega_2, 'k--', linewidth=2, label='ω₂')
plt.xlabel('Tiempo (s)'); plt.ylabel('ω (rad/s)'); plt.title('Comparación velocidades angulares (SI)'); plt.grid(True, alpha=0.3); plt.legend()
fig_cmp_vel.savefig(os.path.join(SALIDAS_DIR, 'comparacion_velocidades_angulares.png'), dpi=300, bbox_inches='tight'); plt.close(fig_cmp_vel)

# Aceleraciones
guardar_fig(t, alpha_1, 'Tiempo (s)', 'α₁ (rad/s²)', 'Aceleración angular α₁', 'alpha1_vs_t.png')
guardar_fig(t, alpha_3, 'Tiempo (s)', 'α₃ (rad/s²)', 'Aceleración angular α₃', 'alpha3_vs_t.png', color='r-')
guardar_fig(t, a_c, 'Tiempo (s)', 'a_c (m/s²)', 'Aceleración vertical martillo', 'ac_vs_t.png', color='g-')
fig_cmp_acc = plt.figure(figsize=(6,4))
plt.plot(t, alpha_1, 'b-', linewidth=2, label='α₁')
plt.plot(t, alpha_3, 'r-', linewidth=2, label='α₃')
plt.xlabel('Tiempo (s)'); plt.ylabel('α (rad/s²)'); plt.title('Comparación aceleraciones angulares (SI)'); plt.grid(True, alpha=0.3); plt.legend()
fig_cmp_acc.savefig(os.path.join(SALIDAS_DIR, 'comparacion_aceleraciones_angulares.png'), dpi=300, bbox_inches='tight'); plt.close(fig_cmp_acc)

# Fuerzas y reacciones
guardar_fig(t, C_t, 'Tiempo (s)', 'C (N)', 'Fuerza en martillo C(t)', 'C_vs_t.png')
guardar_fig(t, B_x, 'Tiempo (s)', 'B_x (N)', 'Fuerza en B - X', 'Bx_vs_t.png', color='r-')
guardar_fig(t, B_y, 'Tiempo (s)', 'B_y (N)', 'Fuerza en B - Y', 'By_vs_t.png', color='g-')
guardar_fig(t, O1x, 'Tiempo (s)', 'O1x (N)', 'Reacción O1x', 'O1x_vs_t.png', color='tab:purple')
guardar_fig(t, O1y, 'Tiempo (s)', 'O1y (N)', 'Reacción O1y', 'O1y_vs_t.png', color='tab:brown')
guardar_fig(t, A_reac, 'Tiempo (s)', 'A (N)', 'Fuerza normal ranura A', 'A_vs_t.png', color='tab:olive')
guardar_fig(t, tau, 'Tiempo (s)', 'τ (N·m)', 'Par del motor τ(t)', 'tau_vs_t.png', color='k-')

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
print(f"\nFuerzas (convención: + hacia arriba en C, +X a la derecha):")
print(f"  C(t) máx = {np.max(C_t):.2f} N")
print(f"  C(t) mín = {np.min(C_t):.2f} N")
print(f"  B_x máx = {np.max(B_x):.2f} N")
print(f"  B_x mín = {np.min(B_x):.2f} N")
print(f"  B_y máx = {np.max(B_y):.2f} N")
print(f"  B_y mín = {np.min(B_y):.2f} N")
print(f"\nReacciones y par del motor:")
print(f"  O1x máx = {np.max(O1x):.2f} N, mín = {np.min(O1x):.2f} N")
print(f"  O1y máx = {np.max(O1y):.2f} N, mín = {np.min(O1y):.2f} N")
print(f"  A (normal en ranura) máx = {np.max(A_reac):.2f} N, mín = {np.min(A_reac):.2f} N")
print(f"  τ(t) máx = {np.max(tau):.2f} N·m, mín = {np.min(tau):.2f} N·m")
print("="*60 + "\n")

# Listado de archivos generados
graficos = [
 'phi_vs_t.png','theta_vs_t.png','beta_vs_t.png','L_vs_t.png','yc_vs_t.png','L_vs_phi.png','beta_vs_theta.png',
 'omega1_vs_t.png','omega3_vs_t.png','dot_yc_vs_t.png','Vaf_vs_t.png','comparacion_velocidades_angulares.png',
 'alpha1_vs_t.png','alpha3_vs_t.png','ac_vs_t.png','comparacion_aceleraciones_angulares.png',
 'C_vs_t.png','Bx_vs_t.png','By_vs_t.png','O1x_vs_t.png','O1y_vs_t.png','A_vs_t.png','tau_vs_t.png'
]
print('Gráficas individuales guardadas en carpeta \'salidas\':')
for gname in graficos:
    print(f'  - {os.path.join("salidas", gname)}')
print("\nProceso completado. Las gráficas individuales están listas para usar dentro de 'salidas/'.")
