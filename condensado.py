import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

dados = pd.read_excel('testebase.xlsx')
dados['febre'] = dados['febre'].map({1: 1, 2: 0})
dados['mialgia'] = dados['mialgia'].map({1: 1, 2: 0})
dados['cefaleia'] = dados['cefaleia'].map({1: 1, 2: 0})
dados['enxetema'] = dados['enxetema'].map({1: 1, 2: 0})
dados['RESULTADO'] = dados['RESULTADO'].map({10: 1, 5: 0})

valores_ausente = SimpleImputer(strategy='most_frequent')
dados[['febre', 'mialgia', 'cefaleia', 'enxetema']] = valores_ausente.fit_transform(
    dados[['febre', 'mialgia', 'cefaleia', 'enxetema']])


X = dados[['febre', 'mialgia', 'cefaleia', 'enxetema']]
y = dados['RESULTADO']

# knn

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)  # usar o train test split é usar validação cruzada aliada com o random state pra evitar variações nas metricas
knn = KNeighborsClassifier(
    n_neighbors=5, weights='uniform', metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
precisao = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Acurácia: {acuracia:.2f}')
print(f'Precisão: {precisao:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')
# sempre usando duas casas decimais e isso se repete pelo código


# arvore de decisão
dt = DecisionTreeClassifier(
    criterion='entropy', min_samples_split=2, min_samples_leaf=1, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred)
precision_dt = precision_score(y_test, y_pred)
recall_dt = recall_score(y_test, y_pred)
f1_dt = f1_score(y_test, y_pred)

print('decision tree metricas')
print(f'Acurácia: {accuracy_dt:.2f}')
print(f'Precisão: {precision_dt:.2f}')
print(f'Recall: {recall_dt:.2f}')
print(f'F1-Score: {f1_dt:.2f}')

# regressão logística

lr = LogisticRegression(solver='lbfgs', penalty='l2',
                        max_iter=100, random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred)
precision_lr = precision_score(y_test, y_pred)
recall_lr = recall_score(y_test, y_pred)
f1_lr = f1_score(y_test, y_pred)
print('lr metricas')
print(f'Acurácia: {accuracy_lr:.2f}')
print(f'Precisão: {precision_lr:.2f}')
print(f'Recall: {recall_lr:.2f}')
print(f'F1-Score: {f1_lr:.2f}')


# mlp
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam',
                    learning_rate_init=0.001, max_iter=200, batch_size=32, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred)
precision_mlp = precision_score(y_test, y_pred)
recall_mlp = recall_score(y_test, y_pred)
f1_mlp = f1_score(y_test, y_pred)
print('mlp metricas')
print(f'Acurácia: {accuracy_mlp:.2f}')
print(f'Precisão: {precision_mlp:.2f}')
print(f'Recall: {recall_mlp:.2f}')
print(f'F1-Score: {f1_mlp:.2f}')


# random forest
rf = RandomForestClassifier(
    n_estimators=100, criterion='entropy', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred)
precision_rf = precision_score(y_test, y_pred)
recall_rf = recall_score(y_test, y_pred)
f1_rf = f1_score(y_test, y_pred)
print('random forest metricas')
print(f'Acurácia: {accuracy_rf:.2f}')
print(f'Precisão: {precision_rf:.2f}')
print(f'Recall: {recall_rf:.2f}')
print(f'F1-Score: {f1_rf:.2f}')


# função para saber se a pessoa tem ou não tem dengue
def prever_dengue(febre, mialgia, cefaleia, enxetema):
    sintomas = np.array([[febre, mialgia, cefaleia, enxetema]])
    probabilidade = knn.predict_proba(sintomas)
    return probabilidade[0][1]


febre = 0
mialgia = 0
cefaleia = 0
enxetema = 1

probabilidade_dengue = prever_dengue(febre, mialgia, cefaleia, enxetema)
print(f'Probabilidade de dengue: {probabilidade_dengue:.2f}')
