import numpy as np
import matplotlib as mpl
from prompt_toolkit.layout.containers import HorizontalAlign
mpl.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(figsize=(7, 7))

# clinvar_overlap.vcf.gz: 155308                                                
# clinvar_overlap_pathogenic.vcf.gz: 34797                                      
# clinvar_overlap_likely_pathogenic.vcf.gz: 12487                               
# clinvar_overlap_benign.vcf.gz: 36616                                          
# clinvar_overlap_likely_benign.vcf.gz: 58587                                   
# clinvar_overlap_uncertain_significance.vcf.gz: 7382                           
# clinvar_overlap_other.vcf.gz: 78                                              
# clinvar_overlap_affects.vcf.gz: 1                                             
# clinvar_overlap_association.vcf.gz: 9                                         
# clinvar_overlap_coma.vcf.gz: 198                                              
# clinvar_overlap_conflicting.vcf.gz: 4075                                      
# clinvar_overlap_drug_response.vcf.gz: 27                                      
# clinvar_overlap_not_provided.vcf.gz: 1024                                     
# clinvar_overlap_protective.vcf.gz: 2                                          
# clinvar_overlap_risk_factor.vcf.gz: 25                                        
# clinvar_specific.vcf.gz: 199163                                               
# clinvar_specific_benign.vcf.gz: 2823                                          
# clinvar_specific_likely_benign.vcf.gz: 3300                                   
# clinvar_specific_pathogenic.vcf.gz: 20733                                     
# clinvar_specific_likely_pathogenic.vcf.gz: 4733                               
# clinvar_specific_uncertain_significance.vcf.gz: 142006                        
# clinvar_specific_affects.vcf.gz: 104                                          
# clinvar_specific_association.vcf.gz: 138                                      
# clinvar_specific_coma.vcf.gz: 1190                                            
# clinvar_specific_conflicting.vcf.gz: 12485                                    
# clinvar_specific_drug_response.vcf.gz: 268                                    
# clinvar_specific_not_provided.vcf.gz: 9262                                    
# clinvar_specific_other.vcf.gz: 1697                                           
# clinvar_specific_protective.vcf.gz: 34                                        
# clinvar_specific_risk_factor.vcf.gz: 390                                      

# clinvar_keys = [["benign", "likely_benign"], ["pathogenic", "lilely_pathogenic"], ["uncetain_significance"], ["affects", "association", "coma", "conflicting", "drug_response", "not_provided", "other", "protective", "risk_factor"]]
clinvar_keys = ["benign", "likely_benign", 
                "pathogenic", "lilely_pathogenic", 
                "uncetain_significance", 
                "affects", "association", "coma", "conflicting", "drug_response", "not_provided", "other", "protective", "risk_factor"
                ]

clinvar_overlap_values = np.array([
    [36616, 58587], 
    [34797, 12487], 
    [7382, 0] ])

clivar_overlap_other_values = np.array([
    [1, 9, 198, 4075, 27, 1024, 78, 2, 25] ])

clinvar_specific_values = np.array([
    [2823, 3300, 0, 0, 0, 0, 0, 0, 0], 
    [20733, 4733, 0, 0, 0, 0, 0, 0, 0], 
    [142006, 0, 0, 0, 0, 0, 0, 0, 0]])

clivar_specific_other_values = np.array([
    [104, 138, 1190, 12485, 268, 9262, 1697, 34, 390]
    ])

# gavin_overlap_benign.vcf.gz: 101518                                           
# gavin_overlap_pathogenic.vcf.gz: 48741                                        
# gavin_overlap_population.vcf.gz: 5049                                         
# gavin_overlap.vcf.gz: 155308                                                  
# gavin_specific_benign.vcf.gz: 46672                                           
# gavin_specific_pathogenic.vcf.gz: 10835                                       
# gavin_specific_population.vcf.gz: 262674                                      
# gavin_specific.vcf.gz: 320181

gavin_keys = ['benign', 'population', 'pathogenic']
gavin_specific_values = [46672, 262674, 10835]
gavin_overlap_values = [101518, 5049, 48741]

size=0.3
cmap = plt.get_cmap("tab20c")
inner_colors = cmap(np.arange(4) * 9)
outer_colors = cmap(np.arange(37))

explode = (0, 0, 0.1, 0.2)
label_distances = 0.5
textprops = dict(horizontalalignment='center')

# Sum is inner
wedges, texts, autotexts = axs.pie(
    clinvar_overlap_values.sum(axis=1), autopct="%0.2f", radius=1-size, 
    colors=inner_colors, wedgeprops=dict(width=size, edgecolor='w'),
    pctdistance=.85, rotatelabels=True)


# individual is outer
def modifiedPct(pct):
    if pct > 0.05:
        return "{:.2f}%".format(pct)
    return ''

wedges0, texts0, autotexts0 = axs.pie(
    clinvar_overlap_values.flatten(), autopct=modifiedPct, radius=1, 
    colors=outer_colors, wedgeprops=dict(width=size, edgecolor='w'),
    pctdistance=1.4, rotatelabels=True)



labels = ['benign', 'patho.', 'uncertain_sig', 'others']
axs.legend(wedges, labels, title='Clinical significance', loc='upper left')

plt.setp(autotexts, size=15, weight="bold")

fig.savefig("test.png")