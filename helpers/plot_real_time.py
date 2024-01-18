import pyqtgraph as pg


class Plotter:
    def __init__(self):
        self.app = pg.mkQApp()
        self.pw = pg.PlotWidget()
        self.plot = pg.plot()
        self.prepare_plot()

        # Prepare curves
        self.curves = []
        predicted = pg.PlotCurveItem(name="Real", pen=dict(color="red", width=3), skipFiniteCheck=True)
        self.plot.addItem(predicted)
        predicted.setPos(0, 0)
        self.curves.append(predicted)

        predicted = pg.PlotCurveItem(name="Predicted", pen=dict(color="blue", width=3), skipFiniteCheck=True)
        self.plot.addItem(predicted)
        predicted.setPos(0, 0)
        self.curves.append(predicted)

    def prepare_plot(self):
        self.plot.setWindowTitle('CO2 concentration prediction')
        self.plot.setLabel("left", "CO2 concentration (ppm)")
        self.plot.setLabel("bottom", "Iteration no.")
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True)

    def plot(self, real, predicted):
        self.curves[0].setData(real)
        self.curves[1].setData(predicted)
        self.app.processEvents()
