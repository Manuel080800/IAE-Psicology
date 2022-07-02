# Script de fuerza bruta:
import subprocess

print("Script de fuerza bruta:")
print()
for i in 102:
    print("Ejecutando la iteracion " + str(i + 1))
    print()
    subprocess.call("Explicadores_ANN_Psicologia_Clasificacion_Fuerza_Bruta.py", shell=True)
    print()
print("Proceso terminado.")
input()