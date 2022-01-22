import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from dataclasses import dataclass


@dataclass
class ChartContainer:
    reference: np.array
    twentyfive: np.array
    thirthytwo: np.array
    fifty: np.array
    blende: int
    pos: str





if __name__ == '__main__':
    
    
    
    # Messung 1 auswerten
    basepath = r"..\mtf-capture\Cropped"

    # container = {"f2-8" : [], "f4": [], "f5-6": [],
    #              "f8": [], "f16": []}
    container = {"25LP":[],"32LP":[], "50LP":[],"Referenz": []}

    for chart in os.listdir(basepath):
        lp, blende, pos = chart.split("_")
        data = cv2.imread(os.path.join(basepath, chart))
        
        


        inner = {
            "lp": lp,
            "blende": float(blende[1:].replace("-",".")),
            "position": pos.split(".")[0],
            "rawdata": data,
            "stdev": data.std(),
        }

        container[lp].append(inner)
        # container[blende].append(inner)

        # container.append(inner)


    # for inner in container:
    #     if inner[]





    
    plt.show()
    axes = plt.gca()
    axes.set_xlim(2.8, 16)
    axes.set_ylim(0, 30)
    LPs=["25LP","32LP","50LP"];
    
    for LP in LPs:
        hl, = axes.plot([], [],'--x')
        for inner in container[LP]:
    
            if inner["position"] == "bottom":
                xdata=np.append(hl.get_xdata(), inner["blende"])
                SortIdx=np.argsort(xdata)
                for ref in container["Referenz"]:
                    if ref["blende"] == inner["blende"]:
                        curref =ref["stdev"]/100
                        break
    
                hl.set_xdata(xdata[SortIdx])
                hl.set_ydata(np.append(hl.get_ydata(), (inner["stdev"]/curref))[SortIdx])
    
            # plt.draw()
    
    
    plt.xticks([2.8,4,5.6,8,16])
    plt.legend(LPs)
    plt.title("MTF als Funktion der Blende\nam Rand")
    plt.xlabel("Blende")
    plt.ylabel("Kontrast in %")
    plt.grid()
    plt.savefig("MTF_Unten_Messung1.png",dpi=600)

    
    
    
    plt.show()
    axes = plt.gca()
    axes.set_xlim(2.8, 16)
    axes.set_ylim(0, 50)
    LPs=["25LP","32LP","50LP"];
    
    for LP in LPs:
        hl, = axes.plot([], [],'--x')
        for inner in container[LP]:
    
            if inner["position"] == "center":
                xdata=np.append(hl.get_xdata(), inner["blende"])
                SortIdx=np.argsort(xdata)
                for ref in container["Referenz"]:
                    if ref["blende"] == inner["blende"]:
                        curref =ref["stdev"]/100
                        break
    
                hl.set_xdata(xdata[SortIdx])
                hl.set_ydata(np.append(hl.get_ydata(), (inner["stdev"]/curref))[SortIdx])
    
            # plt.draw()
    
    
    plt.xticks([2.8,4,5.6,8,16])
    plt.legend(LPs)
    plt.title("MTF als Funktion der Blende\nmittig")
    plt.xlabel("Blende")
    plt.ylabel("Kontrast in %")
    plt.grid()
    plt.savefig("MTF_Mitte_Messung1.png",dpi=600)



    # Messung 2 auswerten
    
    basepath = r"..\mtf-capture\2teMessung\Cropped"

    # container = {"f2-8" : [], "f4": [], "f5-6": [],
    #              "f8": [], "f16": []}
    container = {"5LP": [], "10LP": [], "16LP": [], "25LP":[],"32LP":[], "50LP":[],"Referenz": []}

    for chart in os.listdir(basepath):
        lp, blende, pos = chart.split("_")
        data = cv2.imread(os.path.join(basepath, chart))
        
        


        inner = {
            "lp": lp,
            "blende": float(blende[1:].replace("-",".")),
            "position": pos.split(".")[0],
            "rawdata": data,
            "stdev": data.std(),
        }

        container[lp].append(inner)
        # container[blende].append(inner)

        # container.append(inner)


    # for inner in container:
    #     if inner[]

    
    plt.show()
    axes = plt.gca()
    axes.set_xlim(2.8, 16)
    axes.set_ylim(0, 150)
    LPs=["5LP","10LP","16LP","25LP"];
    
    for LP in LPs:
        hl, = axes.plot([], [],'--x')
        for inner in container[LP]:
    
            if inner["position"] == "bottom":
                xdata=np.append(hl.get_xdata(), inner["blende"])
                SortIdx=np.argsort(xdata)
                for ref in container["Referenz"]:
                    if ref["blende"] == inner["blende"]:
                        curref =ref["stdev"]/100
                        break
    
                hl.set_xdata(xdata[SortIdx])
                hl.set_ydata(np.append(hl.get_ydata(), (inner["stdev"]/curref))[SortIdx])
    
            # plt.draw()
    
    
    plt.xticks([2.8,4,5.6,8,16])
    plt.legend(LPs)
    plt.title("MTF als Funktion der Blende\nam Rand")
    plt.xlabel("Blende")
    plt.ylabel("Kontrast in %")
    plt.grid()
    plt.savefig("MTF_Unten_Messung2.png",dpi=600)
    
    
    
    
    plt.show()
    axes = plt.gca()
    axes.set_xlim(2.8, 16)
    axes.set_ylim(0, 150)
    LPs=["5LP","10LP","16LP","25LP"];
    
    for LP in LPs:
        hl, = axes.plot([], [],'--x')
        for inner in container[LP]:
    
            if inner["position"] == "center":
                xdata=np.append(hl.get_xdata(), inner["blende"])
                SortIdx=np.argsort(xdata)
                for ref in container["Referenz"]:
                    if ref["blende"] == inner["blende"]:
                        curref =ref["stdev"]/100
                        break
    
                hl.set_xdata(xdata[SortIdx])
                hl.set_ydata(np.append(hl.get_ydata(), (inner["stdev"]/curref))[SortIdx])
    
            # plt.draw()
    
    
    plt.xticks([2.8,4,5.6,8,16])
    plt.legend(LPs)
    plt.title("MTF als Funktion der Blende\nmittig")
    plt.xlabel("Blende")
    plt.ylabel("Kontrast in %")
    plt.grid()
    plt.savefig("MTF_Mittig_Messung2.png",dpi=600)
 
 
    


