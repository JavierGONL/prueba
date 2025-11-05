<#
Instala el paquete stix2-otf en MiKTeX (si no está instalado), actualiza la FNDB y muestra
la ruta del archivo STIXTwoMath-Regular.otf para referenciarlo en tus .tex.

Uso: ejecutar desde PowerShell en la raíz del repo:
    powershell -NoProfile -ExecutionPolicy Bypass -File .\install_stix2.ps1
#>

Write-Host "Comprobando mpm (MiKTeX Package Manager)..."
try { 
    & mpm --version 2>$null
} catch {
    Write-Error "mpm no está disponible en PATH. Abre MiKTeX Console o añade mpm al PATH."; exit 1
}

Write-Host "Intentando instalar stix2-otf (si no está ya instalado)..."
& mpm --install=stix2-otf

Write-Host "Actualizando FNDB y mapas de fuentes..."
& initexmf --update-fndb
& initexmf --mkmaps

Write-Host "Buscando la ubicación de STIXTwoMath-Regular.otf..."
$path = & kpsewhich STIXTwoMath-Regular.otf
if ($path) {
    Write-Host "Fuente encontrada en: $path"
} else {
    Write-Warning "No se encontró STIXTwoMath-Regular.otf con kpsewhich. Si no existe, instala la fuente manualmente o revisa MiKTeX Console."
}

Write-Host "Listo. Puedes usar en tus .tex: \\setmathfont{STIX Two Math} o referenciar la ruta completa si lo prefieres."
