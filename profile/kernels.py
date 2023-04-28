import os



models = ["alexnet", "resnet50","vgg16","googlenet","inception_v3","densenet","mobilenet_v3_l","squeezenet"]  #

for model in models:
    print(model)
    cmd = "./kernel.sh "+model
    os.system(cmd)
'''
for model in models:
    cmd = "./multiinference " + model
    os.system(cmd)
'''
