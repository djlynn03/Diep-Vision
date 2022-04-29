import matplotlib.pyplot as plt

def analyzeBest():
    bestParameters = [line.replace("\n","").replace("[","").replace("]","").replace(",","").split(" ") for line in open("parameters/bestParameters.txt", "r").readlines()]
    bestLosses = [float(line[0]) for line in bestParameters]
    bestLRSteps = [float(line[1]) for line in bestParameters]
    bestLRs = [float(line[2]) for line in bestParameters]
    bestMoments = [float(line[3]) for line in bestParameters]
    bestWds = [float(line[4]) for line in bestParameters]
    bestGs = [float(line[5]) for line in bestParameters]

    plt.title("Loss vs Learning Rate Steps")
    plt.scatter(bestLRSteps, bestLosses)
    plt.show()
    plt.title("Loss vs Learning Rate")
    plt.scatter(bestLRs, bestLosses)
    plt.xscale("log")
    plt.show()
    plt.title("Loss vs Momentums")
    plt.scatter(bestMoments, bestLosses)
    plt.show()
    plt.title("Loss vs Weight Drops")
    plt.scatter(bestWds, bestLosses)
    plt.show()
    plt.title("Loss vs Gammas")
    plt.scatter(bestGs, bestLosses)
    plt.show()

def analyzeAll(lrThresh=1e-12):
    totalLosses = [float(line.replace("\n","")) for line in open("parameters/losses.txt").readlines()]
    totalLRSteps = [float(line.replace("\n","")) for line in open("parameters/lrSteps.txt").readlines()]
    totalLRs = [float(line.replace("\n","")) for line in open("parameters/lrs.txt").readlines()]
    totalMoments = [float(line.replace("\n","")) for line in open("parameters/moments.txt").readlines()]
    totalWds = [float(line.replace("\n","")) for line in open("parameters/wds.txt").readlines()]
    totalGs = [float(line.replace("\n","")) for line in open("parameters/gs.txt").readlines()]
    threshLRs = [num for num in totalLRs if num < lrThresh]
    threshLosses = [num for num in totalLosses if totalLRs[totalLosses.index(num)] < lrThresh]

    # plt.title("Loss vs Learning Rate Steps")
    # plt.scatter(totalLRSteps, totalLosses)
    # plt.show()
    plt.title("Loss vs Learning Rate")
    plt.scatter(totalLRs, totalLosses)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    # plt.title("Loss vs Learning Rate Adjusted")
    # plt.scatter(threshLRs, threshLosses)
    # plt.show()
    # plt.title("Loss vs Momentums")
    # plt.scatter(totalMoments, totalLosses)
    # plt.show()
    # plt.title("Loss vs Weight Drops")
    # plt.scatter(totalWds, totalLosses)
    # plt.show()
    # plt.title("Loss vs Gammas")
    # plt.scatter(totalGs, totalLosses)
    # plt.show()
    
# analyzeBest()
analyzeAll()