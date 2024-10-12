# inference
from loguru import logger
import os
import matplotlib.pyplot as plt
import numpy as np
# import keras.utils as image
from keras.preprocessing import image
from keras.models import load_model
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay

# load model
saved_model = load_model("model.h5")

# add log
logger.add('predict_log.log')

# test dataset path
folderlist = []
test_path = "./test/"
black_path = test_path+'black/'
grizzly_path = test_path+'grizzly/'
panda_path = test_path+'panda/'
polar_path = test_path+'polar/'
teddy_path = test_path+'teddy/'
folderlist.append(black_path)
folderlist.append(grizzly_path)
folderlist.append(panda_path)
folderlist.append(polar_path)
folderlist.append(teddy_path)

y_true = [0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10
y_pred = []

for i in range(len(folderlist)):
    test_list = os.listdir(folderlist[i])
    for j in range(len(test_list)):
        img = image.load_img(folderlist[i]+test_list[j])
        img = img.resize((256, 256))
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)
        output = saved_model.predict(img)
        result = np.argmax(output, axis=1)[0]
        y_pred.append(result)
        print(result)
        # record wrong result
        if result != i:
            logger.info("result: "+str(result)+", correct: "+str(i)+", image name: "+test_list[j])
print(y_pred)

# evaluate
average_param  = "macro" 
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average=average_param)
recall = recall_score(y_true, y_pred, average=average_param)
f1 = f1_score(y_true, y_pred, average=average_param)
kappa = cohen_kappa_score(y_true, y_pred)

logger.info(f'Accuracy: {accuracy}')
logger.info(f'Precision: {precision}')
logger.info(f'Recall: {recall}')
logger.info(f'F1-score: {f1}')
logger.info(f'Cohen\'s Kappa: {kappa}')

# round
logger.info(f'Accuracy: {round(accuracy, 3)}')
logger.info(f'Precision: {round(precision, 3)}')
logger.info(f'Recall: {round(recall, 3)}')
logger.info(f'F1-score: {round(f1, 3)}')
logger.info(f'Cohen\'s Kappa: {round(kappa, 3)}')

# confusion matrix
cm = confusion_matrix(y_true, y_pred)

# use seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['black', 'grizzly', 'panda', 'polar', 'teddy'], 
            yticklabels=['black', 'grizzly', 'panda', 'polar', 'teddy'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig("seaborn.png")

# use ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=['black', 'grizzly', 'panda', 'polar', 'teddy'])
disp.plot(cmap='Blues')
plt.savefig("ConfusionMatrixDisplay.png")
