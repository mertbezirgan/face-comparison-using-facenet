import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn import metrics

### load results to find best threshold for this algorithm
same_file = open("same_7000.txt")
different_file = open("different_7000.txt")
# open("same.txt")
# different_file = open("different.txt")

same_data = []
different_data = []

### labeled data goes as true labeled false labeled as order so we merge our results like that
for l in same_file.readlines():
    dat = l.split("\\n")
    same_data.append(float(dat[0]))

for l in different_file.readlines():
    dat = l.split("\\n")
    different_data.append(float(dat[0]))

print(len(same_data))
print(len(different_data))


### method for testing threshold for metrics between start_threshold and end_threshold      increase_threshold is step size
def threshold_analyses(data, start_threshold, end_threshold, increase_threshold):
    (same, different) = data

    metrics = []
    mcc_metrics = []
    f1_metrics = []
    same_metrics = []
    diff_metrics = []


    threshold = 0.0
    for _threshold in np.arange(start_threshold, end_threshold - increase_threshold, increase_threshold):
        try:

            metric = calculate_metrics((same, different), _threshold)
            metrics.append(metric)
            mcc_metrics.append(metric["mcc"])
            f1_metrics.append(metric["f1"])
            same_metrics.append(metric["accuracy"]["same"])
            diff_metrics.append(metric["accuracy"]["different"])

            if np.abs(metric["accuracy"]["same"] - metric["accuracy"]["different"]) < 0.00001:  ### line for finding best threshold
                threshold = _threshold
                _metric = metric
        except ZeroDivisionError:
            print("div by zero")


    ### plot results to graph
    fig = plt.figure()

    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])

    ax2 = fig.add_subplot(gs[1, 0])

    ax3 = fig.add_subplot(gs[:, 1:])

    fig.suptitle("Threshold vs Scores")

    ax1.plot(np.arange(start_threshold + increase_threshold, end_threshold - increase_threshold, increase_threshold), mcc_metrics)

    ax1.axvline(x=threshold, color='r')

    ax1.set_title("Best MCC vs Threshold")

    ax2.plot(np.arange(start_threshold + increase_threshold, end_threshold - increase_threshold, increase_threshold), f1_metrics)

    ax2.axvline(x=threshold, color='r')

    ax2.set_title("Best F1 vs Threshold")

    ax3.plot(np.arange(start_threshold + increase_threshold, end_threshold - increase_threshold, increase_threshold), same_metrics,

             color='b', label='Same')

    ax3.plot(np.arange(start_threshold + increase_threshold, end_threshold - increase_threshold, increase_threshold), diff_metrics,

             color='g', label='Different')

    ax3.axvline(x=threshold, color='r')

    ax3.legend()

    ax3.set_title("Best Accuracy")

    fig.tight_layout()

    plt.show()


    print("For Threshold: {}".format(threshold))

    table = PrettyTable(['Same Acc', 'Different Acc', 'F1', 'MCC', 'Precision', 'Recall'])

    table.add_row([_metric["accuracy"]["same"], _metric["accuracy"]["different"],

                   _metric["f1"], _metric["mcc"], _metric["precision"], _metric["recall"]])

    print(table)


### calculate metrics for given data in tuple for given threshold
def calculate_metrics(data, threshold):
    threshold = threshold

    (same, different) = data

    true_pos = same[np.where(same < threshold)]

    false_neg = same[np.where(same >= threshold)]

    true_neg = different[np.where(different >= threshold)]

    false_pos = different[np.where(different < threshold)]




    nof_true_pos = true_pos.shape[0]

    nof_false_neg = false_neg.shape[0]

    nof_true_neg = true_neg.shape[0]

    nof_false_pos = false_pos.shape[0]

    precision = nof_true_pos / (nof_true_pos + nof_false_pos)

    recall = nof_true_pos / (nof_true_pos + nof_false_neg)

    metric = {"dataset": data,

              "data": {"tp": true_pos, "fn": false_pos, "tn": true_neg, "fp": false_pos},

              "nof": {"tp": nof_true_pos, "fn": nof_false_neg, "tn": nof_true_neg, "fp": nof_false_pos},

              "accuracy": {"same": nof_true_pos * 1.0 / len(same), "different": nof_true_neg * 1.0 / len(different)},

              "f1": 2.0 * (precision * recall) / (precision + recall), "precision": precision, "recall": recall,

              "mcc": (nof_true_pos * nof_true_neg - nof_false_pos * nof_false_neg) / np.sqrt(

                  (nof_true_pos + nof_false_pos) * (nof_true_pos + nof_false_neg) * (nof_true_neg + nof_false_neg) * (

                          nof_true_neg + nof_false_neg))}

    return metric


threshold_analyses((np.array(same_data), np.array(different_data)), max(min(same_data), 0.0001), max(different_data), 0.0001)