
import sys
import pickle

def mean(x):
    return sum(x) / len(x)

if len(sys.argv) != 2:
    print("Wrong number of arguments...")
    sys.exit()

report_file = sys.argv[1]

with open(report_file, "rb") as ipf:
    reports = pickle.load(ipf)



print("Number of reports: ", len(reports))

CVs = reports[0]["Cross_validations"]

accuracy = CVs["mean_test_accuracy"]
print("Number of iterators: ", len(accuracy))
print("Mean_test_accuracy: ", mean(accuracy))

precision = CVs["mean_test_precision"]
print("Mean_test_precision: ", mean(precision))

roc_auc = CVs["mean_test_roc_auc"]
print("Mean_test_roc_auc: ", mean(roc_auc))
