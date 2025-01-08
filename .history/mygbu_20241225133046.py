import tensorflow as tf

# Vérifier les GPU disponibles
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU(s) disponible(s) :")
    for gpu in gpus:
        print(gpu)
else:
    print("Aucun GPU détecté.")
