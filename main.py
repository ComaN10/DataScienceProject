from DataAnalysis import DataAnalysis, PlotTypes

if __name__ == '__main__':

    data_analyze = DataAnalysis()

    data_analyze.show_datainfo()

    data_analyze.pre_process(["fetal_health"])

    data_analyze.plot_features([
        PlotTypes().hist(),
        PlotTypes().violin(),
        PlotTypes().box(),
        PlotTypes().scatter(),
        # PlotTypes().lines(), na presta
        PlotTypes().bar(),
        PlotTypes().lollypop()
    ],hist_number_of_bars_func=PlotTypes.hist_scots,columns=4,plot_size=5)

    #data_analyze.feature engeneering