import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Pour les graphiques

# Étape 1 : Charger et prétraiter les données (Preprocessing)
def preprocess_data(df):
    """
    Cette fonction prétraite les données du Titanic.
    - Supprime les colonnes non pertinentes.
    - Gère les valeurs manquantes (Missing Values).
    - Encode les variables catégorielles (Categorical Encoding).
    - Retourne les caractéristiques (Features), la cible (Target), et les noms des colonnes.
    """
    # Supprimer les colonnes non pertinentes
    df = df.drop(['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'], axis=1)

    # Gérer les valeurs manquantes (Missing Values)
    df['age'].fillna(df['age'].median(), inplace=True)  # Remplacer les âges manquants par la médiane
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)  # Remplacer les ports manquants par le mode
    df['fare'].fillna(df['fare'].median(), inplace=True)  # Remplacer les tarifs manquants par la médiane

    # Encoder les variables catégorielles (Categorical Encoding)
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})  # Encoder 'male' en 0 et 'female' en 1
    df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # Encoder les ports d'embarquement en nombres

    # Séparer les caractéristiques (Features) et la cible (Target)
    X = df.drop('survived', axis=1).values  # Toutes les colonnes sauf 'survived'
    y = df['survived'].values  # La colonne 'survived' est la cible (Target)

    return X, y, df.drop('survived', axis=1).columns.tolist()  # Retourne les noms des colonnes pour référence

# Étape 2 : Normalisation Z-Score (Z-Score Normalization)
def z_score_normalization(X):
    """
    Cette fonction normalise les données en utilisant la méthode Z-Score.
    - Calcule la moyenne (Mean) et l'écart-type (Standard Deviation) de chaque colonne.
    - Applique la formule de normalisation : (X - mean) / std.
    - Retourne les données normalisées, la moyenne (Mean) et l'écart-type (Standard Deviation).
    """
    mean = np.mean(X, axis=0)  # Moyenne (Mean) de chaque colonne
    std = np.std(X, axis=0)  # Écart-type (Standard Deviation) de chaque colonne
    X_normalized = (X - mean) / std  # Appliquer la normalisation Z-Score
    return X_normalized, mean, std

# Étape 3 : Implémenter l'algorithme KNN (K-Nearest Neighbors)
class KNN:
    """
    Cette classe implémente l'algorithme K-Nearest Neighbors (KNN).
    - k : Nombre de voisins (Neighbors) à considérer.
    - fit : Entraîne le modèle avec les données d'entraînement (Training Data).
    - predict : Prédit la classe (Class) pour de nouvelles données.
    """
    def __init__(self, k=3):
        self.k = k  # Nombre de voisins (Neighbors) (par défaut 3)

    def fit(self, X_train, y_train):
        """
        Entraîne le modèle KNN.
        - X_train : Données d'entraînement (Training Data).
        - y_train : Étiquettes (Labels) des données d'entraînement.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Prédit les étiquettes (Labels) pour les données de test (Test Data).
        - X_test : Données de test (Test Data).
        - Retourne les prédictions (Predictions).
        """
        predictions = [self._predict(x) for x in X_test]  # Prédire pour chaque point de test
        return np.array(predictions)

    def _predict(self, x):
        """
        Prédit la classe (Class) pour un seul point de données.
        - x : Un point de données.
        - Retourne la classe prédite (Predicted Class).
        """
        # Calculer les distances entre x et tous les points d'entraînement (Training Data)
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Trier les distances et obtenir les indices des k plus proches voisins (Nearest Neighbors)
        k_indices = np.argsort(distances)[:self.k]
        
        # Extraire les étiquettes (Labels) des k plus proches voisins
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Retourner la classe la plus fréquente (Majority Voting)
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

    def _euclidean_distance(self, x1, x2):
        """
        Calcule la distance euclidienne (Euclidean Distance) entre deux points.
        - x1, x2 : Deux points de données.
        - Retourne la distance.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

# Étape 4 : Entraîner le modèle KNN (Training the KNN Model)
def train_knn(X_train, y_train):
    """
    Entraîne un modèle KNN.
    - X_train : Données d'entraînement (Training Data).
    - y_train : Étiquettes (Labels) des données d'entraînement.
    - Retourne le modèle entraîné (Trained Model).
    """
    knn = KNN(k=3)  
    knn.fit(X_train, y_train)  
    return knn

# Étape 5 : Obtenir les entrées de l'utilisateur pour un passager (User Input)
def get_user_input(mean, std):
    """
    Demande à l'utilisateur de saisir les informations d'un passager.
    - mean : Moyenne (Mean) des données d'entraînement (pour la normalisation).
    - std : Écart-type (Standard Deviation) des données d'entraînement (pour la normalisation).
    - Retourne les données normalisées du passager.
    """
    print("Please enter the following details for the passenger:")
    pclass = int(input("Passenger Class (1, 2, or 3): "))
    sex = input("Sex (male or female): ").strip().lower()
    age = float(input("Age: "))
    sibsp = int(input("Number of Siblings/Spouses Aboard: "))
    parch = int(input("Number of Parents/Children Aboard: "))
    fare = float(input("Fare: "))
    embarked = input("Port of Embarkation (C, Q, or S): ").strip().upper()

    # Encoder les variables catégorielles (Categorical Encoding)
    sex_encoded = 0 if sex == 'male' else 1
    embarked_encoded = {'C': 0, 'Q': 1, 'S': 2}[embarked]

    # Créer un tableau de caractéristiques (Features) pour le passager
    passenger_data = np.array([pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]).reshape(1, -1)

    # Normaliser les données du passager en utilisant la moyenne (Mean) et l'écart-type (Standard Deviation) des données d'entraînement
    passenger_data_normalized = (passenger_data - mean) / std

    return passenger_data_normalized

# Étape 6 : Diviser les données en ensembles d'entraînement et de test (Train-Test Split)
def split_data(X, y, test_size=0.2, random_state=None):
    """
    Divise les données en ensembles d'entraînement (Training Set) et de test (Test Set).
    - X : Caractéristiques (Features).
    - y : Étiquettes (Labels).
    - test_size : Proportion des données à utiliser pour le test (par défaut 20%).
    - random_state : Graine pour la reproductibilité (Reproducibility).
    - Retourne X_train, X_test, y_train, y_test.
    """
    if random_state:
        np.random.seed(random_state)  # Fixer la graine pour la reproductibilité (Reproducibility)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)  # Mélanger les indices
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    split_idx = int(X.shape[0] * (1 - test_size))  # Indice de séparation
    X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
    y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]
    return X_train, X_test, y_train, y_test

# Étape 7 : Calculer les métriques manuellement (Manual Metrics Calculation)
def calculate_metrics(y_true, y_pred):
    """
    Calcule les métriques de performance (Performance Metrics) : matrice de confusion (Confusion Matrix), précision (Precision), rappel (Recall), F1-score.
    - y_true : Étiquettes réelles (True Labels).
    - y_pred : Étiquettes prédites (Predicted Labels).
    - Retourne un dictionnaire contenant les métriques.
    """
    # Matrice de confusion (Confusion Matrix)
    tp = np.sum((y_true == 1) & (y_pred == 1))  # Vrais positifs (True Positives)
    fp = np.sum((y_true == 0) & (y_pred == 1))  # Faux positifs (False Positives)
    fn = np.sum((y_true == 1) & (y_pred == 0))  # Faux négatifs (False Negatives)
    tn = np.sum((y_true == 0) & (y_pred == 0))  # Vrais négatifs (True Negatives)

    # Précision (Precision), Rappel (Recall), F1-score
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {
        'confusion_matrix': np.array([[tn, fp], [fn, tp]]),
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Étape 8 : Afficher la matrice de confusion (Confusion Matrix)
def plot_confusion_matrix(cm):
    """
    Affiche la matrice de confusion (Confusion Matrix) sous forme de graphique.
    - cm : Matrice de confusion (Confusion Matrix).
    """
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Died', 'Survived'], rotation=45)
    plt.yticks(tick_marks, ['Died', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Ajouter les valeurs dans chaque cellule
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.show()

# Étape 9 : Visualiser le passager prédit dans le contexte des données (Data Visualization)
def visualize_predicted_passenger(X_train, y_train, passenger_data, feature_names, feature1='sex', feature2='fare'):
    """
    Affiche un graphique des données d'entraînement (Training Data) avec le passager prédit (Predicted Passenger).
    - X_train : Données d'entraînement (Training Data).
    - y_train : Étiquettes (Labels) des données d'entraînement.
    - passenger_data : Données du passager prédit (Predicted Passenger).
    - feature_names : Noms des colonnes (Column Names).
    - feature1, feature2 : Caractéristiques (Features) à afficher sur les axes.
    """
    try:
        idx1 = feature_names.index(feature1)  # Indice de la première caractéristique (Feature)
        idx2 = feature_names.index(feature2)  # Indice de la deuxième caractéristique (Feature)
    except ValueError as e:
        print(f"Error: {e}. Please check if the feature names are correct.")
        return

    # Afficher les points des données d'entraînement (Training Data)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, idx1], X_train[:, idx2], c=y_train, cmap='coolwarm', alpha=0.6, label='Training Data')
    plt.colorbar(label='Survival Status (0 = Died, 1 = Survived)')

    # Afficher le point du passager prédit (Predicted Passenger)
    plt.scatter(passenger_data[:, idx1], passenger_data[:, idx2], 
                color='red', marker='*', s=200, label='Predicted Passenger')

    # Ajouter des labels et un titre
    plt.xlabel(feature1.capitalize())
    plt.ylabel(feature2.capitalize())
    plt.title(f'Scatter Plot of {feature1.capitalize()} vs {feature2.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Étape 10 : Fonction principale (Main Function)
if __name__ == "__main__":
    # Charger et prétraiter les données (Preprocessing)
    train_data_path = r"C:\Users\MG KALI\Desktop\License AD\Machine Learning\Groupe 5-Titanic Survivor Prediction\Using KNN\KNN From Scratch\titanic_cleaned_updated.csv"  # Chemin vers les données
    train_data = pd.read_csv(train_data_path)
    X, y, feature_names = preprocess_data(train_data)  # Obtenir les noms des colonnes (Column Names)

    # Normaliser les caractéristiques (Features) avec Z-Score
    X_normalized, mean, std = z_score_normalization(X)

    # Diviser les données en ensembles d'entraînement (Training Set) et de test (Test Set)
    X_train, X_test, y_train, y_test = split_data(X_normalized, y, test_size=0.2, random_state=42)

    # Entraîner le modèle KNN (Training the KNN Model)
    knn = train_knn(X_train, y_train)

    # Prédire sur l'ensemble de test (Test Set)
    y_pred = knn.predict(X_test)

    # Calculer la précision (Accuracy)
    accuracy = np.mean(y_pred == y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Obtenir les entrées de l'utilisateur pour un passager (User Input)
    passenger_data = get_user_input(mean, std)

    # Prédire la survie du passager (Predict Survival)
    prediction = knn.predict(passenger_data)

    # Afficher le résultat
    if prediction == 1:
        print("Prediction: Survived")
    else:
        print("Prediction: Died")

    # Demander à l'utilisateur s'il veut voir la matrice de confusion (Confusion Matrix) et les métriques (Metrics)
    show_confusion_matrix = input("Do you want to see the confusion matrix and metrics? (yes/no): ").strip().lower()
    if show_confusion_matrix in ['yes', 'y']:
        # Calculer les métriques (Metrics)
        metrics = calculate_metrics(y_test, y_pred)
        cm = metrics['confusion_matrix']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1_score']

        # Afficher les métriques (Metrics)
        print("\nMetrics:")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        # Afficher la matrice de confusion (Confusion Matrix)
        plot_confusion_matrix(cm)

    # Demander à l'utilisateur s'il veut voir la visualisation des données (Data Visualization)
    show_visualization = input("Do you want to see the data visualization? (yes/no): ").strip().lower()
    if show_visualization in ['yes', 'y']:
        # Visualiser le passager prédit (Predicted Passenger) dans le contexte des données
        visualize_predicted_passenger(X_train, y_train, passenger_data, feature_names, feature1='sex', feature2='fare')