## README: Thesis: Sinkhole Susceptibility Analysis Using Machine Learning

### Overview
This project involves analyzing sinkhole susceptibility using a variety of machine learning algorithms. The dataset contains multiple geological and environmental features, which are used to predict the occurrence of sinkholes. The analysis includes data preprocessing, feature selection, model building, and evaluation of different machine learning models. 
In the article, you can mention the following:

1. **Pre-processing of the Dataset**: The initial pre-processing of the dataset was conducted using ArcGIS, ensuring that the data was accurately prepared for further analysis.

2. **Exporting the RF Model to ArcGIS**: After training the Random Forest (RF) model, it was exported to ArcGIS to generate a detailed sinkhole susceptibility map for the study area, allowing for a seamless integration of the predictive model with advanced geospatial visualization tools.

https://drive.google.com/file/d/1SvmCKVHu_VF9WaQYwLK3qyx9SqVIRrgA/view?usp=sharing

### Requirements
Ensure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install them using pip:
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Data
The dataset `sinkhole_data.csv` contains the following columns:
- `ID`: Unique identifier for each observation.
- `Sinkhole`: Target variable (1 indicates sinkhole occurrence, 0 indicates no sinkhole).
- `Land use`: Land use type.
- `Substrate`: Type of substrate.
- `DTS`: Distance to streams.
- `DTM`: Distance to mines.
- `HD`: Hydraulic head difference.
- `Depth to Bedrock`: Depth to bedrock.
- `DTF`: Distance to fault.
- `DWTSA`: Depth to water table of the surficial aquifer.
- `DTD`: Distance to depressions.

### Steps

#### 1. Importing Required Libraries
```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, precision_recall_fscore_support, precision_recall_curve
```

#### 2. Loading and Understanding the Data
```python
df = pd.read_csv('C:/Users/MUILI OLANREWAJU/Research Thesis/Data/Processed data/sinkhole_data.csv')
print(df.head())
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())
print(df.isna().sum())
```

#### 3. Preprocessing the Data
- Slice relevant columns.
- Re-order columns to have non-categorical variables first.
- Perform one-hot encoding for categorical variables.

```python
df = df[df.columns[1:]]
categorical_variables = ['Substrate', 'Land use', 'Sinkhole']
non_categorical_variables = list(set(df.columns) - set(categorical_variables))

order = non_categorical_variables + categorical_variables
df = df[order]
```

#### 4. Feature Scaling
Standardize the dataset before feeding it into machine learning models.
```python
scaler = StandardScaler()
X = scaler.fit_transform(df.iloc[:, :-1].values)
y = df['Sinkhole'].values
```

#### 5. Principal Component Analysis (PCA)
Visualize the dataset in two dimensions using PCA.
```python
pca = PCA(n_components=2, random_state=0)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
```

#### 6. Model Building
Split the dataset into training and testing sets.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

#### 7. Applying Machine Learning Algorithms
Several machine learning algorithms are applied with hyperparameter tuning using GridSearchCV:
- **Random Forest**
- **K Nearest Neighbors**
- **Logistic Regression**
- **Support Vector Machine**
- **Multilayer Perceptron Neural Network**

Example for Random Forest:
```python
param_grid = {
    'n_estimators': [200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': np.arange(10, 30, 2),
    'criterion': ['gini', 'entropy']
}
grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=param_grid)
grid.fit(X_train, y_train)
print('Best score:', grid.best_score_)
print('Best hyperparameters:', grid.best_params_)
```

#### 8. Model Evaluation
Evaluate the models using confusion matrix, accuracy, ROC AUC, precision, recall, and F1 score.
```python
rf_conf_mat = confusion_matrix(y_test, rf_pred)
rf_acc = accuracy_score(y_test, rf_pred)
rf_roc_auc = roc_auc_score(y_test, rf_proba[:, 1])

print('Accuracy:', rf_acc)
print('ROC AUC:', rf_roc_auc)
```

#### 9. Save Results and Figures
Save PCA plots and model performance results for further analysis.
```python
plt.savefig(os.path.join('..data\\figures', 'pca.png'), dpi=300)
```

### Conclusion
The project involves detailed analysis using various machine learning techniques to predict sinkhole susceptibility. The optimal model and its performance metrics provide insights into which features contribute the most to predicting sinkhole occurrences.

### Author
Olanrewaju Muili

### License
This project is licensed under the MIT License.

### Acknowledgments
Thanks to The Doe Run Company for providing the dataset and support for this project.


## Data

Sinkholes 
http://publicfiles.dep.state.fl.us/otis/gis/data/FGS_SUBSIDENCE_INCIDENTS.zip

Streams
https://geodata.dep.state.fl.us/search?q=geology&amp;sort=-modified

Depressions
http://publicfiles.dep.state.fl.us/otis/gis/data/LAND_SURFACE_ELEVATION_24.zip

Active mines
https://floridadep.gov/fgs

Fault
https://geodata.dep.state.fl.us/search?q=geology&amp;sort=-modified

Substrate
http://publicfiles.dep.state.fl.us/OTIS/GIS/data/GEOLOGY_STRATIGRAPHY.zip

Depth to bedrock
https://geodata.dep.state.fl.us/search?q=geology&amp;sort=-modified

Depth to water table of Surficial aquifer
https://geodata.dep.state.fl.us/search?q=geology&amp;sort=-modified

Head difference
https://geodata.dep.state.fl.us/search?q=geology&amp;sort=-modified

Land use
http://publicfiles.dep.state.fl.us/otis/gis/data/STATEWIDE_LANDUSE.zip


## Literature

https://pdfs.semanticscholar.org/608b/627afb47103d4456ea64abf5a9f74dbec1f8.pdf

https://www.sciencedirect.com/science/article/abs/pii/S0022169420305096?via%3Dihub

https://www.mdpi.com/2076-3417/10/15/5047

https://www.nature.com/articles/s41598-019-43705-6

https://onlinelibrary.wiley.com/doi/10.1002/ldr.3255
