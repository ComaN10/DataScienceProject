from DataAnalysis import DataAnalysis, PlotTypes
from DimensionalityReduction import DimensionalityReduction
from sklearn.datasets import load_iris

if __name__ == '__main__':

    data_analyze = DataAnalysis()

    # wheel print 253 graphs
    # #data_analyze.view_features_pairwyse()´
    # data_analyze = DataAnalysis()
    # data_analyze.show_datainfo()
    # data_analyze.plot_features([PlotTypes().hist()])
    #model_selection = FeatureEngineering()
    #model_selection.FeatureAdd()

    data_analyze.show_datainfo()

    data_analyze.pre_process(ignore=[data_analyze.target_name])

    data_analyze.plot_features([
        PlotTypes().hist(),
        PlotTypes().violin(),
        PlotTypes().box(),
        PlotTypes().scatter(),
        # PlotTypes().lines(), is not usefully
        PlotTypes().bar(),
        PlotTypes().lollypop()
    ],hist_number_of_bars_func=PlotTypes.hist_scots,columns=4,plot_size=5)

    DimensionalityReduction = DimensionalityReduction(data_analyze.dataset, data_analyze.get_targets(), standardized=True)
    DimensionalityReduction.plot_projection(DimensionalityReduction.compute_pca(2), "PCA")
    DimensionalityReduction.plot_projection(DimensionalityReduction.compute_lda(2), "LDA2")
    DimensionalityReduction.plot_projection(DimensionalityReduction.compute_tsne(2), "TSNE")
    DimensionalityReduction.plot_projection(DimensionalityReduction.compute_lle(2), "lle")
    DimensionalityReduction.plot_projection(DimensionalityReduction.compute_umap(2), "umap")

    #data_analyze.feature engeneering
