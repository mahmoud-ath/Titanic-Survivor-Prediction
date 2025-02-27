# 🚢 Titanic Survivor Prediction  

## 📌 Project Overview  
This project aims to predict the survival of Titanic passengers based on
 their characteristics (e.g., age, gender, class). The dataset is preprocessed, 
 and two machine learning models—K-Nearest Neighbors (KNN) and Random Forest—are implemented 
 from scratch and using Scikit-Learn to compare their performance.  

## 📂 Project Structure  
```
Groupe5-Titanic-Survivor-Prediction/
│── data/  
│   ├── tita_knn_sklearn.csv  # cleaned data 
│   ├── titanic_data_scratch.csv   # data forn knn from scratch 
│  
│── notebooks/  
│   ├── 1_Data_Cleaning.ipynb  # Data preprocessing  
│   ├── 2_KNN_From_Scratch.ipynb  # KNN implemented manually  
│   ├── 3_KNN_Sklearn.ipynb  # KNN using Scikit-Learn  
│   ├── 4_Compare_KNN.ipynb  # Comparing KNN models  
│   ├── 5_RandomForest.ipynb  # Random Forest from scratch  
│   
│  
│── results/  
│   ├── 1_knn_comparison.png  # KNN model performance visualization  
│   ├── 2_rf_comparison.png  # Random Forest model performance visualization  
│   ├── Report Titanic Survivor Prediction.pdf  # Final project report  
│  
│── requirements.txt  # List of dependencies  
│── README.md  # Project Overview & Usage Guide  
│── .gitignore  # Ignoring unnecessary files  
│── LICENSE  # License file (optional)  
```

## 🔧 Setup & Installation  

### 1️⃣ Prerequisites  
Ensure you have Python 3.8+ installed.  

### 2️⃣ Install Dependencies  
Run the following command in your terminal:  

     pip install -r requirements.txt


### 3️⃣ Run Jupyter Notebooks  
Launch Jupyter Notebook and open the project notebooks:  

jupyter notebook


## 📊 Model Implementation  
### 🔹 K-Nearest Neighbors (KNN)
- Implemented from scratch using NumPy.  
- Also implemented using Scikit-Learn for comparison.  
- Performance evaluation through accuracy and confusion matrix.  

### 🔹 Random Forest
- Implemented from scratch using decision trees.  
- Compared with Scikit-Learn's RandomForestClassifier.  
- Performance analysis using feature importance and accuracy.  

## 📌 Results & Insights  
- The comparison of custom implementations and Scikit-Learn models helps understand algorithm efficiency and accuracy.  
- The Random Forest model performed better in terms of prediction accuracy compared to KNN.  
- The final results and visualizations can be found in the `results/` directory.  

## 🤝 Contributing  
Contributions are welcome! Feel free to open an issue or submit a pull request.  

## 📜 License  
This project is licensed under the MIT License.  

---
