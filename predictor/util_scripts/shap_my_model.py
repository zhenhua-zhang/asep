import sys
import shap
import pickle
import matplotlib.pyplot as plt

sys.path.append("/home/umcg-zzhang/Documents/git/asep/predictor/")
from asep.predictor import ASEPredictor

# Expected features
expected_features = [
    'EncExp', 'Sngl10000bp', 'GerpN', 'Rare10000bp', 'Dist2Mutation',
    'cHmmQuies', 'bStatistic', 'minDistTSS', 'Sngl1000bp', 'Freq10000bp',
    'minDistTSE', 'EncNucleo','cHmmTxWk', 'cHmmTx', 'cHmmReprPCWk', 'GC',
    'RawScore'
]

height = 9  # plot height
width = 16  # plot width

with open(sys.argv[1], 'rb') as my_file:
    my_object = pickle.load(my_file)

first_k_rows = int(sys.argv[2])

my_x_matrix = my_object.x_matrix.loc[:first_k_rows, ]

my_model = my_object.fetch_model()
plt.clf()  # Clean figures created in model

print("Creating explainer ...", file=sys.stderr)
explainer = shap.TreeExplainer(my_model)

print("Creating shap values ...", file=sys.stderr)
shap_values = explainer.shap_values(my_x_matrix)

with open("shap_values.pkl", 'wb') as pkl_opt:
    pickle.dump(shap_values, pkl_opt)

print("Creating dots summary plot ...", file=sys.stderr)
shap.summary_plot(shap_values, my_x_matrix, plot_type='dot', show=False)
current_fig = plt.gcf()
current_fig.set_figheight(height)
current_fig.set_figwidth(width)
current_fig.set_frameon(False)
plt.savefig("summary_plot_dots.pdf")
plt.savefig("summary_plot_dots.png")
plt.clf()

print("Creating violin summary plot ...", file=sys.stderr)
shap.summary_plot(shap_values, my_x_matrix, plot_type="violin", show=False)
current_fig = plt.gcf()
current_fig.set_figheight(height)
current_fig.set_figwidth(width)
current_fig.set_frameon(False)
plt.savefig("summary_plot_violin.pdf")
plt.savefig("summary_plot_violin.png")
plt.clf()

print("Creating bar summary plot ...", file=sys.stderr)
shap.summary_plot(shap_values, my_x_matrix, plot_type='bar', show=False)
current_fig = plt.gcf()
current_fig.set_figheight(height)
current_fig.set_figwidth(width)
current_fig.set_frameon(False)
plt.savefig("summary_plot_bar.pdf")
plt.savefig("summary_plot_bar.png")

