import numpy as np

def hamming_loss_from_confusion_matrix(conf_matrix):
    conf_matrix = np.array(conf_matrix)
    
    # True Positives 
    correct = np.trace(conf_matrix)
    
    # total
    total = np.sum(conf_matrix)
    
    # errors
    errors = total - correct

    # Hamming Loss = errors / total
    hamming_loss = errors / total
    return round(hamming_loss, 4)

# confusion matrix
conf_matrix = [
    [31,25,14,1,3],
    [26,127,20,17,10],
    [8,2,350,0,1],
    [5,25,6,17,6],
    [0,20,3,10,6]
]

hl = hamming_loss_from_confusion_matrix(conf_matrix)
print(f"Hamming Loss: {hl}")
