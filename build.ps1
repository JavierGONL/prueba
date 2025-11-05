Param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$file
)

Set-StrictMode -Version Latest

try {
    $fileFull = (Resolve-Path $file -ErrorAction Stop).ProviderPath
} catch {
    Write-Error "No se encontró el archivo: $file"
    exit 1
}

$fileDir = Split-Path $fileFull -Parent
Push-Location $fileDir

$filename = [System.IO.Path]::GetFileName($fileFull)

# Si latexmk está disponible, usarlo forzando LuaLaTeX
if (Get-Command latexmk -ErrorAction SilentlyContinue) {
    Write-Host "Usando latexmk para compilar $filename con LuaLaTeX..."
    & latexmk -lualatex -interaction=nonstopmode -file-line-error $filename
    $exit = $LASTEXITCODE
    Pop-Location
    exit $exit
}

Write-Host "latexmk no encontrado. Usando secuencia lualatex/bibtex como fallback..."

if (-not (Get-Command lualatex -ErrorAction SilentlyContinue)) {
    Write-Error "lualatex no está disponible en PATH. Instala MiKTeX y asegúrate de que lualatex esté en PATH."
    Pop-Location
    exit 1
}

& lualatex -interaction=nonstopmode -synctex=1 $filename
if ($LASTEXITCODE -ne 0) { Write-Error "pdflatex falló (código $LASTEXITCODE)"; Pop-Location; exit $LASTEXITCODE }

# Ejecutar bibtex si parece que se necesita
$aux = [System.IO.Path]::ChangeExtension($filename, ".aux")
$needBib = $false
if (Test-Path $aux) {
    $auxcont = Get-Content $aux -Raw -ErrorAction SilentlyContinue
    if ($auxcont -match "\\bibdata") { $needBib = $true }
}
if ($needBib) {
    if (Get-Command bibtex -ErrorAction SilentlyContinue) {
        Write-Host "Ejecutando bibtex..."
        $bname = [System.IO.Path]::GetFileNameWithoutExtension($filename)
        & bibtex $bname
    } else {
        Write-Warning "Se necesita bibtex pero no está en PATH. Omisión de paso bibliográfico."
    }
}

& lualatex -interaction=nonstopmode -synctex=1 $filename
& lualatex -interaction=nonstopmode -synctex=1 $filename

Write-Host "Compilación completada. Revisa el PDF generado en la carpeta del .tex si no hubo errores."
Pop-Location
exit 0
