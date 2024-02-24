from DataAnalysis import DataAnalysis,PlotTypes

if __name__ == '__main__':
    data_analyze = DataAnalysis()
    data_analyze.show_datainfo()
    data_analyze.plot_features([PlotTypes().hist()])
