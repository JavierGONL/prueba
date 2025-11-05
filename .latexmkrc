$pdf_mode = 1;
# Usar lualatex como engine por defecto para generar PDF
$pdflatex = 'lualatex -interaction=nonstopmode -synctex=1 %O %S';
$clean_ext = 'synctex.gz';

# Evitar parar en advertencias menores
$latex = 'lualatex -interaction=nonstopmode -synctex=1 %O %S';
