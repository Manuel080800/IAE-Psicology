@echo off

echo Script de fuerza bruta:
echo.
FOR /L %%i IN (1,1,102) DO (
  echo.
  echo Ejecutando la iteracion # %%i :
  python "Explicadores_ANN_Psicologia_Clasificacion_Fuerza_Bruta.py"
  echo. )
echo Proceso terminado.
pause