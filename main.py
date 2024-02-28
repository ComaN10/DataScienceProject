from DataAnalysis import DataAnalysis, PlotTypes
from ModelSelection import FeatureEngineering

if __name__ == '__main__':
    #data_analyze = DataAnalysis()
    #data_analyze.show_datainfo()
    #data_analyze.plot_features([PlotTypes().hist()])
    model_selection = FeatureEngineering()
    model_selection.FeatureAdd()
