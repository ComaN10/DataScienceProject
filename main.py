from DataAnalysis import DataAnalysis, PlotTypes
from DimensionalityReduction import DimensionalityReduction
from FeaturingEngineering import FeatureExtraction,FeatureSelector
from HypothesisTesting import HypothesisTester
from PCAAnalysis import PCAAnalysis
import numpy as np

if __name__ == '__main__':

    data_analyze = DataAnalysis()

    data_analyze.pre_process(
        ignore=[data_analyze.target_name, "accelerations_category", "fetal_movement_category"]
    )

    # creating features #five percent
    data_analyze.print_outliers_info(
         ignore=[data_analyze.target_name, "accelerations_category", "fetal_movement_category"], severe_outliers=3
    )

    data_analyze.dataset = FeatureExtraction.create_features(data_analyze.dataset, [data_analyze.target_name])
    # wheel print 253 graphs
    # #data_analyze.view_features_pairwyse()´
    # data_analyze = DataAnalysis()
    # data_analyze.show_datainfo()
    # data_analyze.plot_features([PlotTypes().hist()]

    data_analyze.remove_null_values()
    data_analyze.remove_duplicates()



    data_analyze.pre_process(
        ignore=[data_analyze.target_name, "accelerations_category", "fetal_movement_category"]
    )
    #data_analyze.show_datainfo()
    data_analyze.show_correlations_heatmap()

    data_analyze.plot_features([
        PlotTypes().hist(),
        #PlotTypes().violin(),
        #PlotTypes().box(),
        #PlotTypes().scatter(),
        # PlotTypes().lines(), is not usefully
        #PlotTypes().bar(),
        #PlotTypes().lollypop()
        ], hist_number_of_bars_func=PlotTypes.hist_scots,
        columns=6,
        plot_size=5 #,
        #ignore=["accelerations_category", "fetal_movement_category"]
        )

    p = PCAAnalysis(data_analyze.dataset, data_analyze.get_targets(), 39)
    p.plot_explained_variance_ratio()
    #
    DimensionalityReduction = DimensionalityReduction(data_analyze.dataset, data_analyze.get_targets(), standardized=True)
    DimensionalityReduction.plot_3d_combinations(10, 15)
    # DimensionalityReduction.plot_projection(DimensionalityReduction.compute_pca(10), "PCA")
    # DimensionalityReduction.plot_projection(DimensionalityReduction.compute_lda(2), "LDA")
    DimensionalityReduction.plot_projection(DimensionalityReduction.compute_tsne(3,35), "TSNE")
    DimensionalityReduction.plot_projection(DimensionalityReduction.compute_lle(2,80), "LLE")
    DimensionalityReduction.plot_projection(DimensionalityReduction.compute_umap(2,15,0.4), "UMAP")
    # Perguntas numero de componentes

    # graficons no relatorio?
    # escolha do modelo?

    # Hypothesis 5:
    # Null hypothesis (H0): There is no significant difference in fetal health based on the histogram mean.
    # Alternative hypothesis (H1): There is a significant difference in fetal health based on the histogram mean.

    groups1, group_names1 = HypothesisTester.divide_series_in_groups_by_class(data_analyze, "accelerations")
    groups2, group_names2 = HypothesisTester.divide_series_in_groups_by_class(data_analyze, "fetal_movement")
    groups3, group_names3 = HypothesisTester.divide_series_in_groups_by_class(data_analyze, "uterine_contractions")
    groups4, group_names4 = HypothesisTester.divide_series_in_groups_by_class(data_analyze, "histogram_mean")
    groups = groups1 + groups2
    group_names = group_names1 + group_names2

    groups4 = FeatureExtraction.equalize_categories(groups4)

    print()
    #
    HypothesisTester.test_hypothesis_single_feature(groups4, group_names4)

    #resultado contraditorio
    print()
    HypothesisTester.test_hypothesis_between_feature(data_analyze.dataset["histogram_mean"], "histogram_mean", data_analyze.dataset[data_analyze.target_name], "target", )



