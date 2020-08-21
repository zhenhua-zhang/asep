python shap_my_model.py \
    -o ../../publication/figures/shap/gtex.shap_plot \
    -n 15000 \
    -m ../../validating/outputs/predictor/allelicCounts/2020_Jul_12_23_37_02_gtex_gbc_ini100_ic6_oc10_mngs5_pli.exon.fdr0.05/train_model.pkl \
    -f png \
    -F GerpN bStatistic cHmmReprPCWk GerpRS cHmmTxWk minDistTSS cHmmTx cDNApos minDistTSE cHmmQuies relcDNApos pLI_score cHmmTxFlnk CDSpos protPos relProtPos EncH3K4Me3 priPhCons relCDSpos EncH3K4Me1

python shap_my_model.py \
    -o ../../publication/figures/shap/gtex.shap_plot \
    -n 15000 \
    -m ../../validating/outputs/predictor/allelicCounts/2020_Jul_12_23_37_02_gtex_gbc_ini100_ic6_oc10_mngs5_pli.exon.fdr0.05/train_model.pkl \
    -f svg \
    -F GerpN bStatistic cHmmReprPCWk GerpRS cHmmTxWk minDistTSS cHmmTx cDNApos minDistTSE cHmmQuies relcDNApos pLI_score cHmmTxFlnk CDSpos protPos relProtPos EncH3K4Me3 priPhCons relCDSpos EncH3K4Me1

python shap_my_model.py \
    -o ../../publication/figures/shap/gtex.shap_plot \
    -n 15000 \
    -m ../../validating/outputs/predictor/allelicCounts/2020_Jul_12_23_37_02_gtex_gbc_ini100_ic6_oc10_mngs5_pli.exon.fdr0.05/train_model.pkl \
    -f pdf \
    -F GerpN bStatistic cHmmReprPCWk GerpRS cHmmTxWk minDistTSS cHmmTx cDNApos minDistTSE cHmmQuies relcDNApos pLI_score cHmmTxFlnk CDSpos protPos relProtPos EncH3K4Me3 priPhCons relCDSpos EncH3K4Me1

python shap_my_model.py \
    -o ../../publication/figures/shap/bios.shap_plot \
    -n 15000 \
    -m ../../training/outputs/predictor/allelicCounts/2020_Jul_12_23_23_17_bios.gbc.ini100.ic6.oc10.mngs5.pli.exon.fdr0.05/train_model.pkl \
    -f png \
    -F GerpN bStatistic Dist2Mutation cDNApos cHmmReprPCWk cHmmQuies relcDNApos cHmmTx minDistTSE CDSpos GerpRS pLI_score cHmmTxWk minDistTSS protPos EncNucleo relProtPos relCDSpos RawScore priPhCons

python shap_my_model.py \
    -o ../../publication/figures/shap/bios.shap_plot \
    -n 15000 \
    -m ../../training/outputs/predictor/allelicCounts/2020_Jul_12_23_23_17_bios.gbc.ini100.ic6.oc10.mngs5.pli.exon.fdr0.05/train_model.pkl \
    -f pdf \
    -F GerpN bStatistic Dist2Mutation cDNApos cHmmReprPCWk cHmmQuies relcDNApos cHmmTx minDistTSE CDSpos GerpRS pLI_score cHmmTxWk minDistTSS protPos EncNucleo relProtPos relCDSpos RawScore priPhCons

python shap_my_model.py \
    -o ../../publication/figures/shap/bios.shap_plot \
    -n 15000 \
    -m ../../training/outputs/predictor/allelicCounts/2020_Jul_12_23_23_17_bios.gbc.ini100.ic6.oc10.mngs5.pli.exon.fdr0.05/train_model.pkl \
    -f svg \
    -F GerpN bStatistic Dist2Mutation cDNApos cHmmReprPCWk cHmmQuies relcDNApos cHmmTx minDistTSE CDSpos GerpRS pLI_score cHmmTxWk minDistTSS protPos EncNucleo relProtPos relCDSpos RawScore priPhCons

notify-send Job 'done'
