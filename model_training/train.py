import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import copy

from dataset import PropertyImageDataset
from cnn_model import get_property_model

def train_model():
    print("="*60)
    print("INICIANDO ENTRENAMIENTO PROFESIONAL CNN (FASE 3)")
    print("="*60)
    
    # Auto-detectar dispositivo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de entrenamiento: {device}")
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_PATH = BASE_DIR / "data_pipeline" / "data" / "processed" / "propiedades_etiquetadas.csv"
    
    if not CSV_PATH.exists():
        print(f"Error: No se encuentra el dataset en {CSV_PATH}")
        return
        
    print("1. Solucionando Data Leakage: Dividiendo por propiedades (no por fotos)...")
    df = pd.read_csv(CSV_PATH)
    
    # Filtrar válidos antes de dividir
    valid_states = ["A reformar", "Buen estado", "Lujo"]
    df = df[df['estado_conservacion'].isin(valid_states)]
    
    if len(df) == 0:
        print("No hay propiedades válidas etiquetadas para entrenar.")
        return
        
    # Split 80/20 garantizando que las 5 fotos de la misma casa van al mismo saco
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['estado_conservacion'])
    print(f"Casas para Entrenar: {len(train_df)} | Casas para Validar: {len(val_df)}")
    
    # 2. Cargar Datasets con Data Augmentation
    print("2. Cargando imágenes y aplicando Data Augmentation al Train Set...")
    train_dataset = PropertyImageDataset(dataframe=train_df, root_dir=BASE_DIR, is_train=True)
    val_dataset = PropertyImageDataset(dataframe=val_df, root_dir=BASE_DIR, is_train=False)
    
    print(f"Total Fotos Entrenamiento: {len(train_dataset)} | Total Fotos Validación: {len(val_dataset)}")
    
    # DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 3. Inicializar Modelo
    model = get_property_model(num_classes=3)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # 4. Bucle de Entrenamiento con Early Stopping
    num_epochs = 30 # Aumentado, confiaremos en el Early Stopping
    patience = 5
    best_f1 = 0.0
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print("\n3. Iniciando Bucle de Aprendizaje (con Early Stopping)...")
    
    for epoch in range(num_epochs):
        print(f"\nÉpoca {epoch+1}/{num_epochs}")
        print("-" * 15)
        
        # --- FASE ENTRENAMIENTO ---
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Train - Loss: {epoch_loss:.4f}")
        
        # --- FASE VALIDACIÓN ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Val   - Loss: {val_epoch_loss:.4f} | F1-Score (Macro): {val_f1:.4f}")
        
        # --- EARLY STOPPING CHECK ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            print("  --> ¡Nuevo récord F1-Score! Guardando pesos en memoria.")
        else:
            epochs_no_improve += 1
            print(f"  --> Sin mejora ({epochs_no_improve}/{patience} paciencia)")
            if epochs_no_improve >= patience:
                print(f"\n[!] EARLY STOPPING ACTIVADO: El modelo dejó de aprender en la época {epoch+1-patience}.")
                break
                
    # 5. Guardar y Terminar
    print("\n4. Entrenamiento Finalizado.")
    print(f"Mejor F1-Score alcanzado: {best_f1:.4f}")
    
    # Imprimir Matriz de Confusión del mejor intento
    print("\nMatriz de Confusión Final:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    model.load_state_dict(best_model_wts)
    save_path = BASE_DIR / "model_training" / "cnn_model_pesos.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Pesos finales guardados en: {save_path}")

if __name__ == "__main__":
    train_model()
