import generateSDH as gSDH

num_repetitions = 5 # Repeticiones por columna
columns = ['A', 'B', 'C']  # Lista de columnas a procesar
files = ['A10', 'A20', 'A30', 'A40',
         'B10', 'B20', 'B30', 'B40',
         'C10', 'C20', 'C30', 'C40', 'HLT']

# Iterar sobre las repeticiones
for file in files:
    for i in range(1, num_repetitions + 1):
        file_title = f'csv_outputs/ITSC_{file}_Repetition{i:02d}.csv'
        gSDH.process_csv_file(file_title, columns, False, 'dataset.csv')