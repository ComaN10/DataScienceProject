from DataAnalysis import DataAnalysis, PlotTypes
from DimensionalityReduction import DimensionalityReduction
from FeaturingEngineering import FeatureEngineering


if __name__ == '__main__':

    data_analyze = DataAnalysis()

    # creating features
    data_analyze.print_outliers_info(
        ignore=[data_analyze.target_name, "accelerations_category", "fetal_movement_category"], severe_outliers=4)

    data_analyze.dataset = FeatureEngineering.create_features(data_analyze.dataset, [data_analyze.target_name])

    # wheel print 253 graphs
    # #data_analyze.view_features_pairwyse()´
    # data_analyze = DataAnalysis()
    # data_analyze.show_datainfo()
    # data_analyze.plot_features([PlotTypes().hist()]

    data_analyze.show_datainfo()

    data_analyze.pre_process(
        ignore=[data_analyze.target_name, "accelerations_category", "fetal_movement_category"])

    data_analyze.plot_features([
        PlotTypes().hist(),
        PlotTypes().violin(),
        #PlotTypes().box(),
        #PlotTypes().scatter(),
        # PlotTypes().lines(), is not usefully
        #PlotTypes().bar(),
        #PlotTypes().lollypop()
    ],hist_number_of_bars_func=PlotTypes.hist_scots, columns=6, plot_size=15)



    DimensionalityReduction = DimensionalityReduction(data_analyze.dataset, data_analyze.get_targets(), standardized=True)
    DimensionalityReduction.plot_all_3d()
    # DimensionalityReduction.plot_projection(DimensionalityReduction.compute_pca(2), "PCA")
    # DimensionalityReduction.plot_projection(DimensionalityReduction.compute_lda(2), "LDA")
    # DimensionalityReduction.plot_projection(DimensionalityReduction.compute_tsne(3), "TSNE")
    # DimensionalityReduction.plot_projection(DimensionalityReduction.compute_lle(2), "LLE")
    # DimensionalityReduction.plot_projection(DimensionalityReduction.compute_umap(2), "UMAP")
