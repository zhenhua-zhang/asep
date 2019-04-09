import sys
import shap
import pickle
from asep.predictor import ASEPredictor

with open(sys.argv[1], 'rb') as my_file:
    my_object = pickle.load(my_file)

my_x_matrix = my_object.x_matrix[
    [
        'EncExp', 'bStatistic', 'minDistTSS', 'minDistTSE', 'Freq10000bp',
        'Sngl10000bp', 'cDNApos', 'cHmmTxWk', 'Rare10000bp', 'GerpN', 'GC',
        'cHmmTx'
    ]
]
my_model = my_object.fetch_model()

explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(my_x_matrix)

shap.summary_plot(shap_values, my_x_matrix)
