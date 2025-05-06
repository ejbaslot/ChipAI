import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Data from logs
folds = [
    {
        'train_loss': [1.6521, 1.3090, 1.2071, 1.1105, 1.0002, 0.9607, 0.9165, 0.8485, 0.8537, 0.8062, 0.7772, 0.7276, 0.7480, 0.6843, 0.6498, 0.6308, 0.6178, 0.6249, 0.6252, 0.5922],
        'val_loss': [2.7869, 1.6174, 1.4794, 1.7165, 1.2655, 1.1903, 1.1798, 1.2041, 1.0304, 1.4469, 1.0202, 0.9564, 1.1212, 1.4582, 1.0171, 0.8551, 0.7925, 0.8110, 0.6641, 0.8649],
        'train_accuracy': [0.7255, 0.8608, 0.8660, 0.8930, 0.9034, 0.8982, 0.9008, 0.9149, 0.8943, 0.9111, 0.9008, 0.9253, 0.9021, 0.9304, 0.9459, 0.9497, 0.9472, 0.9369, 0.9420, 0.9523],
        'val_accuracy': [0.7018, 0.9075, 0.9280, 0.8406, 0.9229, 0.9100, 0.9640, 0.9075, 0.9357, 0.7763, 0.9075, 0.8817, 0.8149, 0.6889, 0.8483, 0.9126, 0.9383, 0.9023, 0.9640, 0.8740]
    },
    {
        'train_loss': [1.6453, 1.3542, 1.2116, 1.1140, 1.0878, 0.9949, 0.9671, 0.9368, 0.8939, 0.8801, 0.8756, 0.8502, 0.8423, 0.8139, 0.8081, 0.8027, 0.7845, 0.7672, 0.7386, 0.7678],
        'val_loss': [1.7676, 1.8973, 2.1501, 1.9036, 1.6855, 1.3919, 1.3973, 1.6231, 1.3235, 1.2898, 1.1125, 1.1237, 1.0807, 1.0858, 1.0640, 1.0193, 0.9086, 0.8677, 0.8986, 0.8846],
        'train_accuracy': [0.7220, 0.8430, 0.8816, 0.8983, 0.9022, 0.9279, 0.9266, 0.9292, 0.9408, 0.9395, 0.9331, 0.9382, 0.9421, 0.9498, 0.9447, 0.9408, 0.9524, 0.9459, 0.9614, 0.9421],
        'val_accuracy': [0.8737, 0.8686, 0.7655, 0.7680, 0.8428, 0.9304, 0.9021, 0.8119, 0.8892, 0.8737, 0.9407, 0.9278, 0.9175, 0.9149, 0.9149, 0.9278, 0.9510, 0.9665, 0.9356, 0.9330]
    },
    {
        'train_loss': [1.6425, 1.3877, 1.1965, 1.1468, 1.0631, 1.0075, 0.9597, 0.9088, 0.8907, 0.8566, 0.8171, 0.8064, 0.7751, 0.7667, 0.7421, 0.7188, 0.7534, 0.7549, 0.7137, 0.7073],
        'val_loss': [2.2899, 1.6001, 1.4585, 1.3911, 2.3889, 1.5042, 1.3245, 1.1853, 1.3887, 1.3760, 1.4887, 1.1700, 1.1120, 0.9635, 0.9139, 0.9446, 1.0173, 0.9253, 0.8353, 0.8439],
        'train_accuracy': [0.7156, 0.8430, 0.8816, 0.8855, 0.8970, 0.8867, 0.8983, 0.9086, 0.9112, 0.9112, 0.9318, 0.9279, 0.9485, 0.9472, 0.9537, 0.9562, 0.9369, 0.9266, 0.9562, 0.9575],
        'val_accuracy': [0.7036, 0.8814, 0.9021, 0.8943, 0.6340, 0.8119, 0.8402, 0.8660, 0.8376, 0.7912, 0.7732, 0.8454, 0.9046, 0.9227, 0.9485, 0.9253, 0.8918, 0.9124, 0.9536, 0.9613]
    }
]

final_training = {
    'train_loss': [1.5816, 1.2111, 1.0994, 0.9918, 0.9178, 0.8345, 0.7793, 0.7512, 0.7266, 0.7086, 0.6645, 0.6520, 0.6570, 0.6322, 0.6184, 0.6044, 0.6025, 0.5995, 0.5803, 0.5957],
    'train_accuracy': [0.7296, 0.8764, 0.8850, 0.9047, 0.9013, 0.9219, 0.9227, 0.9253, 0.9193, 0.9210, 0.9288, 0.9322, 0.9210, 0.9313, 0.9330, 0.9356, 0.9270, 0.9322, 0.9408, 0.9305]
}

# Plotting Accuracy and Loss
plt.figure(figsize=(15, 12))

# Fold Plots
for i, fold in enumerate(folds, 1):
    epochs = range(1, len(fold['train_loss']) + 1)
    
    # Loss Plot
    plt.subplot(4, 2, (i-1)*2 + 1)
    plt.plot(epochs, fold['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, fold['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Fold {i} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy Plot
    plt.subplot(4, 2, (i-1)*2 + 2)
    plt.plot(epochs, fold['train_accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, fold['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title(f'Fold {i} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

# Final Training Plots
epochs = range(1, len(final_training['train_loss']) + 1)
plt.subplot(4, 2, 7)
plt.plot(epochs, final_training['train_loss'], 'b-', label='Training Loss')
plt.title('Final Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(4, 2, 8)
plt.plot(epochs, final_training['train_accuracy'], 'b-', label='Training Accuracy')
plt.title('Final Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('metrics_plots.png')

# Simulated Confusion Matrix (5-class problem)
np.random.seed(42)
num_samples = 1000
true_labels = np.random.randint(0, 5, num_samples)
pred_labels = true_labels.copy()
# Introduce ~10% error rate (based on ~90% accuracy)
errors = np.random.choice(num_samples, size=int(0.1 * num_samples), replace=False)
pred_labels[errors] = np.random.randint(0, 5, size=len(errors))

cm = confusion_matrix(true_labels, pred_labels)
classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix (Simulated)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')