from sidekick.plot.plot_model import visualize_model
from sidekick.nn.conv.googlenet import GoogLeNet

model= GoogLeNet.build(64, 64, 3, 200)
visualize_model(model, "glenet.jpg")