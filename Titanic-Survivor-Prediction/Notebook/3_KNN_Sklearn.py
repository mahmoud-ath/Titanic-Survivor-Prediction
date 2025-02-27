import numpy as np  # Importation de numpy pour la manipulation de tableaux
import pandas as pd  # Importation de pandas pour le traitement des données
from sklearn.neighbors import KNeighborsClassifier  # Importation du classifieur KNN
from sklearn.model_selection import train_test_split  # Importation pour diviser les données en ensembles d'entraînement et de test
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score  # Importation des métriques pour évaluer le modèle
from sklearn.preprocessing import StandardScaler  # Importation de l'échelle standard pour la normalisation des données
import matplotlib.pyplot as plt  # Importation de matplotlib pour la visualisation des données

def load_data(df):
    """
        Charger les données et les diviser en caractéristiques (X) et cible (y).
    Arguments :
        df (DataFrame) : Jeu de données nettoyé.
    Retourne :
        X (ndarray) : Caractéristiques.
        y (ndarray) : Cible (survécu ou non).
    """
    # Séparation des caractéristiques et de la cible
    X = df.drop('survived', axis=1).values  # Les caractéristiques sont toutes les colonnes sauf 'survived'
    y = df['survived'].values  # La cible est la colonne 'survived'
    return X, y  # Retourne les caractéristiques et la cible

def train_knn(X_train, y_train):
    """
        Entraîner un modèle KNN sur les données d'entraînement.
    Arguments :
        X_train (ndarray) : Caractéristiques d'entraînement.
        y_train (ndarray) : Étiquettes cibles d'entraînement.
    Retourne :
        knn (KNeighborsClassifier) : Modèle KNN entraîné.
    """
    # Initialisation du modèle KNN avec 3 voisins
    knn = KNeighborsClassifier(n_neighbors=3)
    # Entraînement du modèle KNN
    knn.fit(X_train, y_train)
    return knn  

def get_user_input():
    """
        Demander à l'utilisateur de saisir les données d'un passager.
    Retourne :
        passenger_data (ndarray) : Caractéristiques des passagers sous forme de tableau numpy.
    """
    # Demande des informations à l'utilisateur
    print("Enter passenger details:")
    pclass = int(input("Passenger class (1, 2, or 3): "))  
    sex = input("Sex (male or female): ").strip().lower()  
    age = float(input("Age: "))  
    sibsp = int(input("Number of siblings/spouses aboard: "))  
    parch = int(input("Number of parents/children aboard: "))  
    fare = float(input("Fare: "))  
    embarked = input("Port of Embarkation (C, Q, or S): ").strip().upper()  

    # Encodage des variables 'sex' et 'embarked'
    sex_encoded = 0 if sex == 'male' else 1  
    embarked_encoded = {'C': 0, 'Q': 1, 'S': 2}[embarked]  

    # Retourne les données du passager sous forme de tableau numpy
    passenger_data = np.array([pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]).reshape(1, -1)
    return passenger_data  # Retourne les données sous forme de tableau

if __name__ == "__main__":
    train_data_path = r"\Titanic-Survivor-Prediction\Data\tita_knn_sklearn.csv"  # Chemin vers le fichier de données d'entraînement
    train_data = pd.read_csv(train_data_path)  # Chargement des données dans un DataFrame

    # Vérifie si des valeurs manquantes sont présentes et les remplit avec la médiane
    if train_data.isnull().any().any():
        print("Warning: Dataset contains missing values. Filling missing values...")
        train_data.fillna(train_data.median(), inplace=True)

    X, y = load_data(train_data)  # Chargement des données et séparation en caractéristiques et cible
    scaler = StandardScaler()  # Initialisation du normaliseur
    X_scaled = scaler.fit_transform(X)  # Normalisation des caractéristiques

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    knn = train_knn(X_train, y_train)  
    y_pred = knn.predict(X_test)  

    accuracy = np.mean(y_pred == y_test)  
    print(f"Model Accuracy: {accuracy * 100:.2f}%")  

    passenger_data = get_user_input()  
    passenger_data_scaled = scaler.transform(passenger_data)  # Normalisation des données du passager
    prediction = knn.predict(passenger_data_scaled)  # Prédiction pour le passager

    # Affichage du résultat de la prédiction
    result = "Survived" if prediction == 1 else "Did not survive"
    print(f"Prediction: {result}")

    # Affichage de la matrice de confusion si demandé
    if input("Show confusion matrix? (yes/no): ").strip().lower() in ['yes', 'y']:
        cm = confusion_matrix(y_test, y_pred)  # Calcul de la matrice de confusion
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Affichage de la matrice de confusion
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Did not survive', 'Survived'], rotation=45)
        plt.yticks(tick_marks, ['Did not survive', 'Survived'])
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Ajout des valeurs dans la matrice de confusion
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.tight_layout()
        plt.show()

        # Affichage des métriques de performance
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("\nMetrics:")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

    # Affichage de la visualisation des données si demandé
    if input("Show data visualization? (yes/no): ").strip().lower() in ['yes', 'y']:
        feature1 = 'sex'  
        feature2 = 'fare'  
        feature_names = train_data.drop('survived', axis=1).columns  # Noms des caractéristiques
        idx1 = list(feature_names).index(feature1)  # Index de la première caractéristique
        idx2 = list(feature_names).index(feature2)  # Index de la deuxième caractéristique

        # Création du graphique de dispersion
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train[:, idx1], X_train[:, idx2], c=y_train, cmap='coolwarm', alpha=0.6, label='Training Data')
        plt.colorbar(label='Survival Status (0 = Did not survive, 1 = Survived)')

        # Ajout du passager prédit sur le graphique
        plt.scatter(passenger_data_scaled[:, idx1], passenger_data_scaled[:, idx2], color='red', marker='*', s=200, label='Predicted Passenger')

        plt.xlabel(feature1.capitalize())  
        plt.ylabel(feature2.capitalize()) 
        plt.title(f'Scatter Plot of {feature1.capitalize()} vs {feature2.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()  
