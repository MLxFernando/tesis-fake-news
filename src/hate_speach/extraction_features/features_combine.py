import numpy as np

class FeatureCombiner:
    """
    Clase para combinar matrices de características.
    """
    def __init__(self):
        pass

    @staticmethod
    def combine_features(feature_matrix_1, feature_matrix_2, feature_matrix_3):
        """
        Combina tres matrices de características en una sola matriz.

        Args:
            feature_matrix_1 (scipy.sparse.csr.csr_matrix or np.ndarray): Primera matriz de características (TF-IDF).
            feature_matrix_2 (np.ndarray): Segunda matriz de características (Sentimientos).
            feature_matrix_3 (np.ndarray): Tercera matriz de características (Adicionales).

        Returns:
            np.ndarray: Matriz combinada.
        """
        # Convertir a array si es necesario
        if not isinstance(feature_matrix_1, np.ndarray):
            feature_matrix_1 = feature_matrix_1.toarray()

        combined_features = np.concatenate([feature_matrix_1, feature_matrix_2, feature_matrix_3], axis=1)
        return combined_features
