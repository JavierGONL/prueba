# Configuración rápida de LaTeX (MiKTeX + Strawberry Perl)

Este archivo describe pasos para terminar la configuración del entorno LaTeX en Windows cuando ya tienes instalados Strawberry Perl y MiKTeX.
# Configuración rápida de LaTeX (MiKTeX + Strawberry Perl)

Este archivo describe pasos para terminar la configuración del entorno LaTeX en Windows cuando ya tienes instalados Strawberry Perl y MiKTeX.

Requisitos verificados por el autor: Strawberry Perl, Perl disponible en PATH y MiKTeX instalados.

1) Verificar instalaciones (PowerShell)

    perl -v
    pdflatex --version
    mpm --version   # opcional: administrador de paquetes MiKTeX
    latexmk --version   # si está instalado

Si alguno de los comandos no existe, asegúrate de añadir la ruta correspondiente al PATH o reinstalar.

2) Instalar latexmk (recomendado)

latexmk automatiza la secuencia de compilación (pdflatex/bibtex/biber/xdvipdfmx...). Puedes instalarlo desde MiKTeX Console (Buscar "latexmk") o con la herramienta de línea `mpm`:

    mpm --admin --install=latexmk

Nota: `mpm --admin` puede necesitar ejecutar PowerShell/terminal como Administrador.

3) Uso del script `build.ps1` incluido

He añadido `build.ps1` en la raíz del proyecto. Uso recomendado desde PowerShell en la raíz del repo:

7) Instalar STIX Two Math en MiKTeX (opcional, recomendado)

Si quieres usar la fuente `STIX Two Math` como fuente matemática (por ejemplo `\setmathfont{STIX Two Math}`), puedes instalar el paquete `stix2-otf` en MiKTeX. He añadido un script `install_stix2.ps1` que automatiza la instalación y muestra la ruta al archivo `STIXTwoMath-Regular.otf`.

Para ejecutarlo desde la raíz del repo:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install_stix2.ps1
```

El script realiza:
- `mpm --install=stix2-otf`
- `initexmf --update-fndb` y `initexmf --mkmaps`
- busca la ruta con `kpsewhich STIXTwoMath-Regular.otf` y la imprime.

Una vez instalada la fuente, en tus .tex puedes usar (con lualatex):

```tex
\usepackage{fontspec}
\setmathfont{STIX Two Math}
```

Si por alguna razón necesitas la ruta absoluta (ej. `\setmathfont[Path=...]{STIXTwoMath-Regular.otf}`), el script mostrará la ruta instalada (normalmente algo como `C:/Users/<usuario>/AppData/Local/Programs/MiKTeX/fonts/opentype/public/stix2-otf/STIXTwoMath-Regular.otf`).

    # Compilar un archivo .tex específico
    powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1 "latex\calculos\3. LAZO VECTORIAL.tex"

    El script detecta `latexmk` y lo usa si está disponible. Si no, ejecuta una secuencia de `lualatex`, `bibtex` (si hace falta) y dos pasadas adicionales de `lualatex`.

4) Archivo de configuración `.latexmkrc`

El repositorio incluye una `.latexmkrc` con opciones recomendadas para `pdflatex` (modo nonstop y synctex). Puedes personalizarla si usas `xelatex` o `lualatex`.

5) Problemas comunes y soluciones

- Si faltan paquetes durante la compilación, abre MiKTeX Console y habilita la instalación automática de paquetes o instala el paquete concreto.
- Si `latexmk` no se instala con `mpm`, usa MiKTeX Console (UI) para buscar e instalar `latexmk`.
- Si perl no se encuentra, asegúrate de que la carpeta `C:\Strawberry\perl\bin` (o similar) esté en PATH.

6) Verificación rápida

Desde PowerShell en la raíz del repo:

        perl -v ; lualatex --version ; latexmk --version

Si `latexmk` no está presente, el script seguirá funcionando con `pdflatex`/`bibtex`.

Si quieres, puedo intentar ejecutar el script aquí en tu repo para compilar `latex\calculos\3. LAZO VECTORIAL.tex` y reportar el resultado; dime si quieres que lo haga ahora.
