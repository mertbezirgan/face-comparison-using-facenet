import pandas as pd
import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt
from prettytable import PrettyTable
import seaborn as sns

# best_threshold = 0.8009
best_threshold = 0.7954

labeled_data = pd.read_csv("formatted-7000-data.csv")
# labeled_data = pd.read_csv("formatted-data.csv")
real_results = np.array(labeled_data["label"])
test_results = []

same_file = open("same_7000.txt")
# same_file = open("same.txt")
different_file = open("different_7000.txt")
# different_file = open("different.txt")

same_data = []
different_data = []
### read numerical data from output files
for l in same_file.readlines():
    dat = l.split("\\n")
    same_data.append(float(dat[0]))

for l in different_file.readlines():
    dat = l.split("\\n")
    different_data.append(float(dat[0]))

### label calculated data for given threshold that goes with same order as labeled data
for i in range(0, len(same_data)):
    if same_data[i] < best_threshold:
        test_results.append(1)
    else:
        test_results.append(0)

    if different_data[i] < best_threshold:
        test_results.append(1)
    else:
        test_results.append(0)


### calculate metrics
accuracy = metrics.accuracy_score(real_results, test_results)
f1 = metrics.f1_score(real_results, test_results)
mcc = metrics.matthews_corrcoef(real_results, test_results)
precision = metrics.precision_score(real_results, test_results)
recall = metrics.recall_score(real_results, test_results)
conf_ma = metrics.confusion_matrix(real_results, test_results)



### draw graphs for given metrics
fig = plt.figure()

gs = fig.add_gridspec(2, 5)

ax3 = fig.add_subplot(gs[0, 0::])

ax4 = fig.add_subplot(gs[1, 0::])



x = np.arange(5)

ax3.bar(x, [accuracy, f1, mcc, precision, recall])

ax3.set_title('Metrics')

ax3.set_xticklabels(['', 'Acc', 'F1', 'MCC', 'Precision', 'Recall'])


ax4.set_title('Confusion Matrix')

sns.set(font_scale=1.0)

ax4 = sns.heatmap( conf_ma, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt="d")

labels = ["0", "1"]

ax4.set_xticklabels(labels)

ax4.set_yticklabels(labels)

ax4.set(ylabel="True Label", xlabel="Predicted Label")



plt.show()

table = PrettyTable(['Accuracy', 'F1', 'MCC', 'Precision', 'Recall'])

table.add_row([accuracy, f1, mcc, precision, recall])

print(table)

print("")
