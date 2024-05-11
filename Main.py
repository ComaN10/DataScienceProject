from DataAnalysis import DataAnalysis, PlotTypes
from DimensionalityReduction import DimensionalityReduction
from FeaturingEngineering import FeatureExtraction,FeatureSelector
from HypothesisTesting import HypothesisTester
from PCAAnalysis import PCAAnalysis
import numpy as np

if __name__ == '__main__':

    # loading
    data_analyze = DataAnalysis()

    data_analyze.pre_process(
        ignore=[data_analyze.target_name]
    )

    data_analyze.show_datainfo()

    data_analyze.plot_features([
        PlotTypes().hist(),
        PlotTypes().violin()
    ], hist_number_of_bars_func=PlotTypes.hist_scots,
        columns=5,
        plot_size=5
    )

    data_analyze.print_outliers_info(ignore=[data_analyze.target_name])

    data_analyze.plot_features([
        PlotTypes().box(),
    ], hist_number_of_bars_func=PlotTypes.hist_scots,
        columns=5,
        plot_size=5,
        ignore=[data_analyze.target_name]
    )

    print("Feature Creation")
    new_dataset = FeatureExtraction.create_features(data_analyze.dataset, [data_analyze.target_name])
    # update principal data_analyze
    data_analyze.dataset = new_dataset

    # creating a new data_analyze to analyze the created features
    data_analysis_features_created = DataAnalysis()

    last_18_columns = new_dataset.columns[-18:]
    data_analysis_features_created.dataset = new_dataset[last_18_columns]

    data_analysis_features_created.remove_duplicates()
    data_analysis_features_created.remove_null_values()
    data_analysis_features_created.show_datainfo()

    # Distribution Analyses
    data_analysis_features_created.plot_features([
        PlotTypes().hist()
    ], hist_number_of_bars_func=PlotTypes.hist_scots,
        columns=5,
        plot_size=5)

    data_analysis_features_created.plot_features([
        PlotTypes().violin()
    ], hist_number_of_bars_func=PlotTypes.hist_scots,
        columns=5,
        plot_size=5
    )

    # Outliers
    data_analysis_features_created.print_outliers_info(
        ignore=[data_analyze.target_name, "accelerations_category", "fetal_movement_category"])

    data_analysis_features_created.plot_features([
        PlotTypes().box(),
    ], hist_number_of_bars_func=PlotTypes.hist_scots,
        columns=4,
        plot_size=5,
        ignore=[data_analysis_features_created.target_name, "accelerations_category", "fetal_movement_category"]
    )

    data_analyze.remove_null_values()
    data_analyze.remove_duplicates()

    # wheel print 253 graphs
    # #data_analyze.view_features_pairwyse()´
    # data_analyze = DataAnalysis()
    # data_analyze.show_datainfo()
    # data_analyze.plot_features([PlotTypes().hist()]

    data_analyze.show_correlations_heatmap(1, True)

    # Dimensional reduction
    pca_analysis = PCAAnalysis(data_analyze.dataset, data_analyze.get_targets(), len(data_analyze.dataset.columns))
    pca_analysis.plot_explained_variance_ratio()

    # saving dataset
    data_analyze.save_as_csv("Complete_treated_dataset")

    dataset_targets = data_analyze.get_targets()
    dataset_without_target = data_analyze.dataset.drop(columns=[data_analyze.target_name])

    featureSelector = FeatureSelector(dataset_without_target, dataset_targets)
    result_mmr = featureSelector.select_features_mrmr(5)

    # saving data
    np.save("Complete_treated_reduction_mmr.npy", result_mmr)

    DimensionalityReduction = DimensionalityReduction(dataset_without_target, dataset_targets, standardized=True)
    # DimensionalityReduction.plot_3d_combinations(10, 15)
    pca_result = DimensionalityReduction.compute_pca(14)
    np.save("Complete_treated_reduction_pca.npy", pca_result)

    # merge dataset
    data_analyze.dataset[data_analyze.target_name] = dataset_targets

    DimensionalityReduction.plot_projection(pca_result, "PCA")
    DimensionalityReduction.plot_projection(DimensionalityReduction.compute_lda(2), "LDA")
    DimensionalityReduction.plot_projection(DimensionalityReduction.compute_tsne(3,83), "TSNE")
    DimensionalityReduction.plot_projection(DimensionalityReduction.compute_lle(2,33), "LLE")
    DimensionalityReduction.plot_projection(DimensionalityReduction.compute_umap(2,15,0.4), "UMAP")

    # Hypothesis test
    print()

    lst_results = list()
    for feature in data_analyze.dataset.columns:

        h0 = f"H0 Entre {feature} e target não existe uma diferença significativa"
        print(h0)
        print(f"H1 Entre {feature} e target existe uma diferença significativa")

        result, normal_result = HypothesisTester.test_hypothesis_between_feature(
            data_analyze.dataset["histogram_mean"], "histogram_mean",
            data_analyze.dataset[data_analyze.target_name], "target",
            hypothesis=h0
        )
        lst_results.append({"H": h0, "result": result})

        print("\n")

        h0 = f"H0 Entre classes na variável {feature} não existe uma diferença significativa"
        print(h0)
        print(f"H1 Entre classes na variável {feature} existe uma diferença significativa")

        groups, group_names = HypothesisTester.divide_series_in_groups_by_class(data_analyze, "light_decelerations")
        result, normal_result = HypothesisTester.test_hypothesis_single_feature(groups, group_names, hypothesis=h0,test_mormality=False,normal=normal_result)

        lst_results.append({"H": h0, "result": result})

        print("---------------------------------------------------------------------",end="\n\n\n\n")

    # True is reject , show results
    for dic in lst_results:
        print(f"{dic['H']} result {dic['result']}")




