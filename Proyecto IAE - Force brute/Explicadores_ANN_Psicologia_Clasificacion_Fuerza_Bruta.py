#!/usr/bin/env python
# coding: utf-8

# # Explicando ANN (Clasificación) para identificar depresión por fuerza bruta

# ## Preparación de los datos

# Lectura de los datos de acuerdo a sus datasets

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Dataset_MO_ENG.csv")
dq = pd.read_csv("Dataset_QS_STATUS.csv")
dm = pd.read_csv("Dataset_QS_METRICS.csv")


# Eliminación de las preguntas fisicas-relativas

# In[2]:


df = df.drop(df.columns[102:-1], axis=1)


# Determinación de la pregunta continua por fuerza bruta

# In[3]:


numberQuestion = dq.iloc[:, 0]
statusQuestion = dq.iloc[:, 1]

numberSelect = None

index = -1
for status in statusQuestion:
    index += 1
    if status == 'None':
        numberSelect = numberQuestion[index]
        break
        
if numberSelect == None:
    exit()
    
numberSelect


# Eliminación de la pregunta continua

# In[4]:


df.drop(df.columns[numberSelect - 1],axis=1, inplace=True)
df


# Agrupación de los targets en las nuevas clases Low, Medium y High

# In[5]:


dic = { 1: 0 , 2: 0, 3:1, 4:2, 5:2} 
df['Target'] = df['Target'].map(dic)

train_cols = df.columns[0:-1]
label = df.columns[-1]
X = df[train_cols]
y = df[label]
target_names=["Low","Medium","High"]
df


# ## Equilibración de los datos

# Verificación del equilibrio de los datos

# In[6]:


import numpy as np
from imblearn.over_sampling import SMOTE

y.value_counts()


# Uso del oversampling con SMOTE

# In[7]:


random_state = 13
oversample = SMOTE(random_state=random_state)
X, y = oversample.fit_resample(X, y)

y.value_counts()


# ## Preparación del entrenamiento de la red neuronal

# Asignación de los datos de entrenamiento

# In[8]:


seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)


# ## Entrenamiento de la red neuronal

# Entrenamiento por medio de diferentes conbinaciones por medio de la validación cruzada con alpha regular

# In[9]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

cv_scores_mean=[]
cv_scores_std=[]

regul_param_range = 10.0 ** -np.arange(-2, 7)

for regul_param in regul_param_range:
    mlp=MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='adam', alpha=regul_param, 
             learning_rate='constant', learning_rate_init=0.0001, max_iter=100000, random_state=seed)

    scores = cross_val_score(mlp, X, y, cv=5, scoring='f1_macro')
    
    cv_scores_mean.append(scores.mean())
    cv_scores_std.append(scores.std())

cv_scores_mean, cv_scores_std


# Generación de la grafica de curva de aprendizaje con alpha regular

# In[10]:


import matplotlib.pyplot as plt

plt.plot(np.log10(regul_param_range), cv_scores_mean, color="g", label="Test")
lower_limit = np.array(cv_scores_mean) - np.array(cv_scores_std)
upper_limit = np.array(cv_scores_mean) + np.array(cv_scores_std)
plt.fill_between(np.log10(regul_param_range), lower_limit, upper_limit, color="#DDDDDD")

plt.title("Curva de aprendizaje")
plt.xlabel("Alpha 10^{X}"), plt.ylabel("F1"), plt.legend(loc="best")
plt.tight_layout()
# plt.show()


# Entrenamiento por medio de diferentes conbinaciones por medio de la validación cruzada con alpha en 1

# In[11]:


cv_scores_mean=[]
cv_scores_std=[]

regul_param_range = 10.0 ** -np.arange(0, 7)

for regul_param in regul_param_range:
    mlp=MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='adam', alpha=1, 
             learning_rate='constant', learning_rate_init=regul_param, max_iter=100000, random_state=seed)
    
    scores = cross_val_score(mlp, X, y, cv=5, scoring='f1_macro')
    
    cv_scores_mean.append(scores.mean())
    cv_scores_std.append(scores.std())

cv_scores_mean, cv_scores_std


# Generación de la grafica de curva de aprendizaje con alpha regular

# In[12]:


plt.plot(np.log10(regul_param_range), cv_scores_mean, color="g", label="Test")

lower_limit = np.array(cv_scores_mean) - np.array(cv_scores_std)
upper_limit = np.array(cv_scores_mean) + np.array(cv_scores_std)
plt.fill_between(np.log10(regul_param_range), lower_limit, upper_limit, color="#DDDDDD")

plt.title("Curva de aprendizaje")
plt.xlabel("Learning Rate 10^{-X}"), plt.ylabel("F1"), plt.legend(loc="best")
plt.tight_layout()
# plt.show()


# ## Generación del modelo final de la red neuronal

# Se genera el modelo final de la red neuronal a partir de los datos anteriormente entrenados

# In[13]:


mlp=MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='adam', alpha=1, 
             learning_rate='constant', learning_rate_init=0.0001, max_iter=100000, random_state=77)

mlp.fit(X_train, y_train)


# Se guarda el modelo final de la red neuronal

# In[14]:


import joblib

joblib.dump(mlp,"modelo_depresion.pkl")


# ## Preparación de los datos para la explicación de la red neuronal

# Lectura de los datos de acuerdo a sus datasets

# In[15]:


df = pd.read_csv("Dataset_MO_ENG.csv")
dq = pd.read_csv("Dataset_QS_STATUS.csv")


# Eliminación de las preguntas fisicas-relativas

# In[16]:


df = df.drop(df.columns[102:-1], axis=1)


# Determinación de la pregunta continua por fuerza bruta

# In[17]:


numberQuestion = dq.iloc[:, 0]
statusQuestion = dq.iloc[:, 1]

numberSelect = None

index = -1
for status in statusQuestion:
    index += 1
    if status == 'None':
        numberSelect = numberQuestion[index]
        break
        
if index == -1:
    exit()
    
numberSelect


# Eliminación de la pregunta continua

# In[18]:


df.drop(df.columns[numberSelect - 1],axis=1, inplace=True)
df


# Agrupación de los targets en las nuevas clases Low, Medium y High

# In[19]:


dic = { 1: 0 , 2: 0, 3:1, 4:2, 5:2} 
df['Target'] = df['Target'].map(dic)

train_cols = df.columns[0:-1]
label = df.columns[-1]
X = df[train_cols]
y = df[label]
target_names=["Low","Medium","High"]
df


# Equilibrio de los datos anteriormente leidos

# In[20]:


y.value_counts()


# Uso del oversampling con SMOTE

# In[21]:


random_state = 13
oversample = SMOTE(random_state=random_state)
X, y = oversample.fit_resample(X, y)

y.value_counts()


# Lectura del modelo final de la red neuronal

# In[22]:


seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
mlp= joblib.load("modelo_depresion.pkl")


# # Preparación de la explicabilidad del red neuronal

# Uso del counterfactuals para la explicabilidad

# In[23]:


import dice_ml

d = dice_ml.Data(dataframe=df, continuous_features=[], outcome_name='Target')

mlp = joblib.load("modelo_depresion.pkl")
m = dice_ml.Model(model=mlp, backend="sklearn")

exp = dice_ml.Dice(d, m, method="random")

e1 = exp.generate_counterfactuals(query_instances=X_test[0:2], total_CFs=2, desired_class=2)
e1.visualize_as_dataframe(show_only_changes=True)


# # Explicación de la red neuronal

# Explicación del modelo por medio de la matrix de confusión

# In[24]:


from sklearn.metrics import ConfusionMatrixDisplay

X_test.columns = X.columns.str.replace(r".", "")

matrixConfision = ConfusionMatrixDisplay.from_estimator (mlp, X_test, y_test, display_labels=['low','med','high'], cmap=plt.cm.Blues)
matrixConfision.ax_.set_title('Matrix de confusión')

# plt.show()


# Obtención de los valores de importancia a partir la matrix de confisión

# In[25]:


from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = mlp.predict(X_test)

print('Matrix de confusión:')
print(confusion_matrix(y_test,y_pred))
print()
print('Classification accuracy =',accuracy_score(y_test,y_pred)*100,'%\n')
print(classification_report(y_test,y_pred))


# # Almacenamiento de los datos

# Se recuperan los valores de importancia y se almacenan para escritura

# In[26]:


from sklearn import metrics
clf_rep = metrics.precision_recall_fscore_support(y_test,y_pred)
clf_rep


# Se prepara los datos recuperados para su escritura

# In[27]:


clf_rep
accuracy = accuracy_score(y_test,y_pred)*100
precision = [clf_rep[0][0], clf_rep[0][1], clf_rep[0][2]]
recall = [clf_rep[1][0], clf_rep[1][1], clf_rep[1][2]]
f1_score = [clf_rep[2][0], clf_rep[2][1], clf_rep[2][2]]
support = [clf_rep[3][0], clf_rep[3][1], clf_rep[3][2]]

metrics = pd.DataFrame([[numberSelect,accuracy,precision[0],recall[0],f1_score[0],support[0],
                                              precision[1],recall[1],f1_score[1],support[1],
                                              precision[2],recall[2],f1_score[2],support[2]]],
                        columns = ['Question','Acurracy global','Precision_0','Recall_0','F1_score_0', 'Support_0',
                                                                'Precision_1','Recall_1','F1_score_1', 'Support_1',
                                                                'Precision_2','Recall_2','F1_score_2', 'Support_2'])

metrics


# Se escriben los datos preparados en el dataset correspondiente

# In[28]:


metrics.to_csv(r"Dataset_QS_METRICS.csv", mode = 'a', header = False, index = False)


# Se actualiza el estatus de la pregunta continua por fuerza bruta

# In[29]:


dq.loc[index,'Status'] = 'Iterated'
dq.to_csv(r"Dataset_QS_STATUS.csv", index = False)


# In[ ]:




