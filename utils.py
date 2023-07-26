import librosa
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from matplotlib import pyplot as plt

def save_confusion_matrix(confusion_mat, labels, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def save_classification_report(classification_rep, save_path):
    lines = classification_rep.split('\n')
    if len(lines) < 5:
        print("Invalid classification report format.")
        return
    
    col_labels = lines[0].split()
    row_labels = []
    cell_text = []

    for line in lines[2:-4]:
        row = line.split()
        if len(row) >= len(col_labels):
            row_labels.append(row[0])
            cell_text.append(row[1:])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    ax.table(cellText=cell_text,
             colLabels=col_labels,
             rowLabels=row_labels,
             cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


