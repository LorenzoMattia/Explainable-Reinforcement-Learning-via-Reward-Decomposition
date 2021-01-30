import matplotlib.pyplot as plt

class GraphCollector(object):
    def __init__(self):
        self.x = []
        self.y = []
    
    def append(self, x, y):
        self.x.append(x)
        self.y.append(y)
        
    def plot(self, xlabel=None, ylabel=None):
        plt.plot(self.x, self.y, )
        if not xlabel is None:
            plt.xlabel(xlabel)
        if not ylabel is None:
            plt.ylabel(ylabel)
        plt.show()
        
    def __str__(self):
        points = ''
        for i in range(len(self.x)):
            points += f'[{self.x[i]},{self.y[i]}],'
        return f"Graph:\n   X={self.x}\n   Y={self.y}\n   Points: " + points[:-1]