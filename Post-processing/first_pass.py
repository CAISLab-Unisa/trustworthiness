import pandas as pd
from openpyxl import load_workbook
import re

# path
trustability_file = r"./TRustability_Benign_Compute1.xlsx"
moderate_file = r".\Cartel1.xlsx"
output_file = r".\Trustability_Malign_Updated69.xlsx"

# upload the trustability file as a workbook to maintain formula and formatting
wb = load_workbook(trustability_file)
template_ws = wb["Moderate_Alexnet_alexnet"]  # sheet taken as an example

# Upload the Trustability file (taking the “ALEXNET_ALEXNET” sheet as a template)
trustability_xls = pd.ExcelFile(trustability_file)
template_df = trustability_xls.parse("Moderate_Alexnet_alexnet", header=1)  

moderate_xls = pd.ExcelFile(moderate_file)
moderate_df = moderate_xls.parse(moderate_xls.sheet_names[0], header=0)  

# Check if the “Gradcam-Consensus” column exists in Trustability.
if "Gradcam-Consensus" not in template_df.columns:
    raise ValueError("Errore: la colonna 'Gradcam-Consensus' non è presente nel file Trustability.")

# Checks whether the “Immagine” column exists in both DataFrames.

if "Immagine" not in template_df.columns or "Immagine" not in moderate_df.columns:
    raise ValueError("Errore: la colonna 'Immagine' deve essere presente in entrambi i file.")

# Data alignment using “Immagine”

moderate_df.set_index("Immagine", inplace=True)
template_df.set_index("Immagine", inplace=True)

def format_sheet_name(column_name):
    match = re.search(r'([\w]+)_scratch-([\w]+)_scratch', column_name)  
    if match:
        name = f"Mali_{match.group(1)}_{match.group(2)}"
    else:
        name = f"Mali_{column_name}"

    return name[:31]  

# Get the columns of “Moderate” (excluding “Image”).
columns = moderate_df.columns.tolist()

# Creating new sheets for each column in Moderate.xlsx.
for col in columns:
    # Crea un nuovo foglio copiando il template
    new_ws = wb.copy_worksheet(template_ws)

    sheet_name = format_sheet_name(col)
    new_ws.title = sheet_name  

    # Inserts values from the Gradcam-Consensus column in the RIGHT row without overwriting the title.
    for row_idx, img in enumerate(template_df.index, start=3):  
        if img in moderate_df.index:
            new_ws["C" + str(row_idx)] = moderate_df.loc[img, col]  

#save
wb.save(output_file)
print(f"✅ File salvato con successo: {output_file}")






