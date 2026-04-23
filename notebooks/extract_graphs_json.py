import json

def consolidate_notebooks():
    nbs = [
        "NJorge1_Ingestion_Datos.ipynb.ipynb",
        "NJorge2_Preprocesamiento_y_Calidad.ipynb.ipynb",
        "NJorge3_Entrenamiento_Modelo_Hibrido.ipynb"
    ]
    
    # Base structure for Notebook 4
    nb4 = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.1"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    # Keywords that typically indicate a plotting cell
    plot_keywords = ['plt.', 'sns.', 'px.', 'plotly', 'hist(', 'scatter(', 'boxplot(', '.plot(']
    
    for nb_file in nbs:
        print(f"Procesando {nb_file}...")
        try:
            with open(nb_file, 'r', encoding='utf-8') as f:
                nb = json.load(f)
        except Exception as e:
            print(f"Error al leer {nb_file}: {e}")
            continue
            
        cells = nb.get('cells', [])
        nb4['cells'].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# Gráficas de {nb_file}\n"]
        })
        
        i = 0
        while i < len(cells):
            cell = cells[i]
            
            if cell['cell_type'] == 'code':
                source = "".join(cell.get('source', []))
                
                # Check if it contains plotting code
                is_plot = any(kw in source for kw in plot_keywords)
                
                if is_plot:
                    print(f"  -> Encontrada celda de gráfica en índice {i}")
                    # Look back for markdown explanations
                    j = i - 1
                    md_cells = []
                    while j >= 0 and cells[j]['cell_type'] == 'markdown':
                        # Stop if the markdown is a major header (we might just want the immediate preceding text)
                        # Or we just take the single immediate preceding markdown cell to be safe
                        # To be safe, let's take all contiguous previous markdown cells up to a limit or major header
                        md_source = "".join(cells[j].get('source', []))
                        
                        # Stop if it's a level 1 header
                        md_cells.append(cells[j])
                        if md_source.strip().startswith('# '):
                            break
                        j -= 1
                        
                    # Add markdown cells in correct order
                    for md_cell in reversed(md_cells):
                        nb4['cells'].append(md_cell)
                        
                    # Add code cell
                    nb4['cells'].append(cell)
            i += 1
            
    with open("NJorge4_Graficas_Contenido.ipynb", 'w', encoding='utf-8') as f:
        json.dump(nb4, f, indent=1)
    print("Consolidación terminada con éxito en NJorge4_Graficas_Contenido.ipynb.")

if __name__ == "__main__":
    consolidate_notebooks()
