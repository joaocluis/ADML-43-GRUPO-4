**Este projeto utiliza 5 modelos de inteligência artificial para prever se um paciente tem dengue com base nos sintomas apresentados. Os dados incluem informações sobre febre, cefaleia, mialgia e exantema. O projeto é configurado para ser executado em um contêiner Docker, garantindo portabilidade e facilidade de execução em diferentes ambientes.**

**Em nosso trabalho decidimos usar uma planilha a parte, visto que usar a base de dados dada diretamente ficaria muito pesado para subir para o github, podendo dar erros no projeto. Então usamos a base de dados fornecida (dengue_sinan) extraímos dados de 80 pacientes e colocamos em uma planilha a parte e em cima dessa aplicamos os algoritimos de machine learning. Esse trabalho está sendo modificado em virtude da finalização do projeto, a fim de servir como portifólio** 

### **knn:**
Acurácia: 0.71 Precisão: 0.75 Recall: 0.69 F1-Score: 0.72

### **árvore de decisão:**

Acurácia: 0.75 Precisão: 0.82 Recall: 0.69 F1-Score: 0.75

### **regressão logística:**

Acurácia: 0.75 Precisão: 0.82 Recall: 0.69 F1-Score: 0.75

### **mlp:**

Acurácia: 0.71 Precisão: 0.75 Recall: 0.69 F1-Score: 0.72

### **random forest:**

Acurácia: 0.71 Precisão: 0.75 Recall: 0.69 F1-Score: 0.72


### Análise KNN:
O modelo KNN tem uma acurácia de 0.71, o que indica que ele está correto em 71% das previsões. A precisão é relativamente boa, sugerindo que quando o modelo prevê que um paciente realmente está doente, ele está certo 75% das vezes. O recall de 0.69 indica que ele consegue identificar 69% das ocorrências reais dos pacientes doentes. O F1-Score de 0.72 sugere um equilíbrio tanto entre a previsão quanto com o recall

### Análise Árvore de Decisão:
A árvore de decisão mostra uma acurácia de 0.75, que é superior ao KNN. A precisão é alta (0.82), indicando que o modelo é muito bom em prever que o paciente realmente está doente. No entanto, o recall permanece em 0.69, o que sugere que o modelo pode estar perdendo alguns dados de pacientes realmente doentes. O F1-Score de 0.75 reflete uma boa harmonia entre precisão e recall, sendo este modelo um pouco melhor para o caso do que o knn

## Análise Regressão Logística:
A regressão logística apresenta exatamente as mesmas métricas que a árvore de decisão. A acurácia de 0.75 e a precisão de 0.82 indicam que o modelo é eficaz em saber se o paciente está realmente doente. O recall de 0.69 mostra que o modelo tem a mesma dificuldade em capturar todas os casos positivos. O F1-Score de 0.75 sugere um desempenho equilibrado semelhante ao da árvore de decisão.

## Análise MLP:
O MLP tem uma acurácia de 0.71, que é idêntica ao KNN e ao Random Forest. A precisão e o recall são também os mesmos de KNN, com um F1-Score de 0.72. Isso indica que o MLP tem um desempenho comparável ao KNN, com uma ligeira vantagem no equilíbrio entre precisão e o recall

## Análise Random Forest:
O Random Forest apresenta métricas idênticas ao KNN e ao MLP. A acurácia de 0.71 e o F1-Score de 0.72 sugerem um equilíbrio entre precisão (0.75) e recall (0.69), mas não supera as métricas de árvore de decisão e regressão logística.


### nome dos participantes:

João Luis da Cruz de Souza

Renato Marcelo Ramos

Guilherme Ferreira

Pedro Fernandes

Kelly Santos

Gustavo Carvalho

Marla Eduarda 






