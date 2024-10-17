import pandas as pd
import matplotlib.pyplot as plt
import generateSDH as gSDH

# Cargar el archivo CSV
file_title = 'csv_outputs/ITSC_A40_Repetition01.csv'
db = pd.read_csv(file_title) 

tmp = db.index + 1

# Extraer las columnas correspondientes a colA, colB y colC
pA = db['A']
pB = db['B']
pC = db['C']

# Definir el rango de índices
nmi = 0
nmf = 250

# Graficar los datos
plt.plot(tmp[nmi:nmf], pA[nmi:nmf], label='A', color='blue', linewidth=2)
plt.plot(tmp[nmi:nmf], pB[nmi:nmf], label='B', color='red')
plt.plot(tmp[nmi:nmf], pC[nmi:nmf], label='C', color='green')

plt.title('ITSC')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend(title="Fases")
plt.show()

# Graficar un par de histogramas
gSDH.process_csv_file(file_title, ['A', 'B', 'C'], True, "test.csv")