# Configuración del Mecanismo de Retorno Rápido

## Descripción del Mecanismo

Este es un mecanismo de retorno rápido tipo Whitworth con las siguientes características:

### Pivotes Fijos
- **O₁**: Pivote fijo en el origen (0, 0)
- **O₂**: Pivote fijo a distancia `d` de O₁ (en x = d, y = 0)

### Barras y Eslabones

#### 1. Barra Motriz O₂-A
- **Longitud**: `r` (radio de la barra motriz)
- **Punto O₂**: Fijo, conectado al motor
- **Punto A**: Se mueve en círculo de radio `r` alrededor de O₂
- **Ángulo**: φ (phi) - Varía con el tiempo según ω₂
- **Motor**: Gira con velocidad angular constante ω₂

#### 2. Barra Rígida O₁-B
- **Longitud**: `L` (FIJA - la barra es rígida)
- **Punto O₁**: Pivote fijo
- **Punto A**: Está sobre esta barra a distancia L de O₁
- **Punto B**: Está sobre esta barra (puede estar en el extremo o más allá de A)
- **Ángulo**: θ (theta) - Se determina por la posición de A

**IMPORTANTE**: El punto A está simultáneamente en:
- La barra motriz O₂-A (a distancia r de O₂)
- La barra rígida O₁-B (a distancia L de O₁)

Esta restricción cinemática determina la relación entre φ y θ.

#### 3. Eslabón B-C
- **Longitud**: `K` (constante)
- **Punto B**: Conectado a la barra O₁-B
- **Punto C**: Martillo (se mueve verticalmente)
- **Ángulo**: β (beta) - Determinado por la geometría

### Parámetros Actuales

```python
d = 0.12 m = 120 mm    # Distancia entre pivotes O1 y O2
L = 0.20 m = 200 mm    # Longitud de la barra O1-B (RÍGIDA)
r = 0.08 m = 80 mm     # Radio de la barra motriz O2-A
R = 0.10 m = 100 mm    # Posición de B en O1-B
K = 0.25 m = 250 mm    # Longitud del eslabón B-C
D = 0.15 m = 150 mm    # Parámetro geométrico horizontal
ω₂ = 10 rad/s          # Velocidad angular del motor
```

### Cinemática del Mecanismo

#### Posición del punto A
```
A_x = d + r·cos(φ)
A_y = 0 + r·sin(φ)
```

#### Ángulo θ de la barra O₁-B
Como A debe estar a distancia L de O₁:
```
θ = atan2(A_y, A_x)
```

#### Velocidad angular de O₁-B
```
ω₁ = (ω₂ · r / L) · cos(φ - θ)
```

#### Ángulo β del eslabón B-C
```
β = arcsin((R·cos(θ) - D) / K)
```

#### Velocidad angular del eslabón B-C
```
ω₃ = -(ω₁ · R · sin(θ)) / (K · cos(β))
```

### Resultados de la Simulación

Con los parámetros actuales:

**Carrera del martillo**: 133.33 mm
- Posición máxima: -171.67 mm
- Posición mínima: -305.00 mm

**Velocidad del martillo**:
- Velocidad máxima: ±402.5 mm/s
- La velocidad es simétrica (no hay retorno rápido pronunciado con estos parámetros)

**Aceleración del martillo**:
- Aceleración máxima: 4.02 m/s²
- Aceleración mínima: -4.84 m/s²

**Velocidades angulares**:
- ω₁: ±4.00 rad/s (oscila)
- ω₃: ±0.55 rad/s (oscila)
- ω₂: 10.00 rad/s (constante)

### Archivos Generados

1. **graficar_mecanismo.py**: Script principal para generar gráficas
   - Gráficas de ángulos y posición
   - Gráficas de velocidades
   - Gráficas de aceleraciones

2. **animar_mecanismo.py**: Script para generar animación GIF
   - Muestra el movimiento del mecanismo
   - Gráficas dinámicas de velocidad/aceleración
   - Información instantánea

3. **Gráficas generadas**:
   - graficas_angulos_posicion.png
   - graficas_velocidades.png
   - graficas_aceleraciones.png

### Notas

- La barra O₁-B es **rígida** (longitud L constante)
- El punto A es la **restricción cinemática** que conecta ambas barras
- Los parámetros pueden ajustarse al inicio de cada script
- Para obtener un retorno rápido más pronunciado, ajustar las relaciones entre L, r, R y K
