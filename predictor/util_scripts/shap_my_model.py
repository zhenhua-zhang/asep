import sys
import shap
import pickle
import matplotlib.pyplot as plt

sys.path.append("/home/umcg-zzhang/Documents/git/asep/predictor/")
from asep.model import ASEP

# Expected features
expected_features = [
    'Sngl10000bp', 'EncExp', 'GerpN', 'minDistTSS', 'bStatistic', 'cHmmQuies',
    'minDistTSE', 'cHmmTxWk', 'EncNucleo', 'Freq10000bp', 'cHmmReprPCWk',
    'Sngl1000bp', 'Rare10000bp', 'pLI_score', 'Dist2Mutation', 'GC',
    'EncH3K27Ac', 'EncH3K4Me3', 'EncH3K4Me1'
]

height = 9  # plot height
width = 16  # plot width

with open(sys.argv[1], 'rb') as my_file:
    my_object = pickle.load(my_file)

first_k_rows = int(sys.argv[2])

my_x_matrix = my_object.x_matrix.loc[:first_k_rows, ]

my_model = my_object.fetch_models()[0]
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
