import os
import subprocess
import re
import numpy as np
import json
import xml.etree.ElementTree as etree
 
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from processing_functions import readAmitexResultsRelaxation


from tensor_personal_functions import generate_isoC_from_E_nu, generate_trans_isoC_from_E_nu_G, isotropic_projector_Facu, printVoigt4,transverse_isotropic_projector

plt.rcParams.update({'font.size': 22})
plt.rcParams.update({"text.usetex": True})
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams["font.family"] = "Times New Roman"


# finds all files in path that have been processed with Amitex

dataPaths=[
    ("sampleB1/processed_x1-2009_y1-1722_z1-971/2022-06-15_09h31m26/AmitexFiles_microStructure_June2023/downSampled/",r"sampleB1_periodic2x4_manual_([0-9]{2})_([0-9]{2})"),
    ("sampleF1/processed_x56-1436_y80-1422_z1-970/2022-06-27_13h16m01/AmitexFiles_microStructure_June2023/downSampled/",r"sampleF1_manual_([0-9]{2})_([0-9]{2})")
]

plot_error_bar=True

rgb_lowTemp =(0.00,0.35,0.70)
rgb_highTemp=(1.00,0.25,0.10)

color={
    "lowTemp"       :rgb_lowTemp,
    "highTemp"      :rgb_highTemp,
    "lowTemp_meso"  :rgb_lowTemp,
    "highTemp_meso" :rgb_highTemp,
}


fiberCount      ={"lowTemp":[], "highTemp":[]}
fiberFraction   ={"lowTemp":[], "highTemp":[]}
poresFraction   ={"lowTemp":[], "highTemp":[]}
E_l_Amitex_t0   ={"lowTemp":[], "highTemp":[]}
E_l_Amitex_tEnd ={"lowTemp":[], "highTemp":[]}
E_t_Amitex      ={"lowTemp":[], "highTemp":[]}
nu_l_Amitex     ={"lowTemp":[], "highTemp":[]}
nu_t_Amitex     ={"lowTemp":[], "highTemp":[]}
G_l_Amitex      ={"lowTemp":[], "highTemp":[]}

def find(dataPath,searchStr):

    pathsList=[]

    for dirPath, dirName, filenames in os.walk(dataPath):
        for filename in filenames:
            if searchStr in filename:
                pathsList.append((dirPath,filename))

    return pathsList

searchStrList=[
    "log_mat",
    "C_hom.txt",
    "_mat.xml", # for number of fibers (zones count)
    "fiberVolumeFractions",
    "FiberStats.json"
    ]

dataDict={"lowTemp":{}, "highTemp":{}}

for dataPath, regex in dataPaths:
    for searchStr in searchStrList:
        pathsList=find(dataPath,searchStr)

        resultsFiles = [(d[0].replace(dataPath,""),d[1]) for d in pathsList if d]

        if searchStr == "fiberVolumeFractions":
            temp_type = "lowTemp"

        for dir, filename in resultsFiles:

            if "lowTemp" in dir:
                temp_type = "lowTemp"
            elif "highTemp" in dir:
                temp_type = "highTemp"

            if "sampleF1" in regex: 
                offset=24
            else:
                offset=0


            matchObj=re.match(regex,dir)

            instanceNum=int(matchObj.group(1))*3+int(matchObj.group(2))+offset

            if instanceNum not in dataDict[temp_type]:
                dataDict[temp_type][instanceNum]={
                        searchStr:os.path.join(dir,filename),
                        "dataPath":dataPath
                    }
            else:
                dataDict[temp_type][instanceNum][searchStr]=os.path.join(dir,filename)           
                dataDict[temp_type][instanceNum]["dataPath"]=dataPath

            if searchStr=="fiberVolumeFractions":
                with open(os.path.join(dataPath, dataDict[temp_type][instanceNum][searchStr]),'r') as f:
                    fiberStats=json.load(f)

                dataDict[temp_type][instanceNum]["fiberFraction"]=\
                    fiberStats["meanFiberMatrixFrac"]

                dataDict[temp_type][instanceNum]["poresFraction"]=\
                    fiberStats["meanPoresFrac"]

                # ################################

                parentDir=os.path.dirname(dataDict[temp_type][instanceNum][searchStr])

                pathFigureStats=os.path.join(dataPath,parentDir,"FiguresFiberStats")

                if not os.path.exists(pathFigureStats):
                    raise IOError("Need to run script fiberStatistics_Amitex.py on this dataset first.")

                with open(os.path.join(pathFigureStats,"FiberStats.json") ,'r' ) as f:
                    fiberStats=json.load(f)

                dataDict[temp_type][instanceNum]["meanLength"]=\
                    fiberStats["meanLength"]

                dataDict[temp_type][instanceNum]["meanDeviation"]=\
                    fiberStats["meanDeviation"]

            if searchStr=="FiberStats.json":
                with open(os.path.join(dataPath, dataDict[temp_type][instanceNum][searchStr]),'r') as f:
                    fiberStats=json.load(f)

                dataDict[temp_type][instanceNum]["fiberCount"]=\
                    fiberStats["fiberCount"]

for temp_type in dataDict:
    for instanceNum in dataDict[temp_type]:

        if "C_hom.txt" not in dataDict[temp_type][instanceNum]:
            continue # means this simulation crashed


        dataPath=dataDict[temp_type][instanceNum]["dataPath"]

        pathC_hom=os.path.join(dataPath, dataDict[temp_type][instanceNum]["C_hom.txt"] )

        if "log_mat" in dataDict[temp_type][instanceNum].keys():
            pathLog=os.path.join(dataPath, dataDict[temp_type][instanceNum]["log_mat"] )
        else:
            pathLog=None

        fiberFract=dataDict["lowTemp"][instanceNum]["fiberFraction"]

        fiberFraction["lowTemp"].append(fiberFract)

        poresFrac=dataDict["lowTemp"][instanceNum]["poresFraction"]

        poresFraction["lowTemp"].append(poresFrac)

        dataDict[temp_type][instanceNum]["Creep_results"]=readAmitexResultsRelaxation(pathC_hom)
        
        C_tilde_0=dataDict[temp_type][instanceNum]["Creep_results"]["C_hom"][0]

        axis=2
        E_l,E_t,nu_l,nu_t,G_l=transverse_isotropic_projector(C_tilde_0,axis)[-5:]
        
        E_l_Amitex_t0[temp_type] .append(E_l/1000.)
        E_t_Amitex[temp_type] .append(E_t/1000.)
        nu_l_Amitex[temp_type].append(nu_l/1000.)
        nu_t_Amitex[temp_type].append(nu_t/1000.)
        G_l_Amitex[temp_type] .append(G_l/1000.)

        fiberCount[temp_type].append(dataDict["lowTemp"][instanceNum]["fiberCount"])

        C_tilde_tEnd=dataDict[temp_type][instanceNum]["Creep_results"]["C_hom"][-1]

        axis=2
        E_l,E_t,nu_l,nu_t,G_l=transverse_isotropic_projector(C_tilde_tEnd,axis)[-5:]
        
        E_l_Amitex_tEnd[temp_type] .append(E_l/1000.)


import random

# index=[v for v in range(len(E_l_Amitex_t0["lowTemp"]))]
# random.shuffle(index)
# print(index)

# sample_B1 only
# index=[6, 15, 11, 12, 1, 18, 13, 0, 5, 8, 9, 4, 19, 21, 3, 20, 7, 14, 16, 2, 17, 10, 22]

# index=[24, 34, 43, 30, 21, 31, 37, 0, 15, 39, 44, 4, 1, 7, 27, 6, 22, 23, 38, 5, 29, 11, 10, 35, 20, 9, 33, 14, 41, 25, 2, 26, 36, 18, 16, 3, 8, 12, 13, 32, 28, 19, 42, 17, 40]

# [3, 13, 26, 33, 9, 32, 4, 27, 31, 34, 7, 6, 1, 39, 37, 20, 10, 29, 18, 40, 2, 21, 38, 17, 19, 24, 41, 15, 16, 5, 23, 42, 0, 43, 14, 11, 25, 30, 28, 36, 44, 22, 35, 8, 12]

index=[44, 22, 29, 13, 31, 23, 16, 25, 26, 42, 28, 1, 24, 21, 43, 36, 20, 41, 37, 2, 11, 39, 9, 27, 10, 8, 19, 32, 18, 38, 40, 30, 17, 7, 0, 33, 3, 5, 12, 15, 35, 34, 4, 6, 14]

E_l_Amitex_t0["lowTemp"]  =[E_l_Amitex_t0["lowTemp"][i] for i in index]
E_l_Amitex_tEnd["lowTemp"]=[E_l_Amitex_tEnd["lowTemp"][i] for i in index]

E_l_Amitex_t0["highTemp"]  =[E_l_Amitex_t0["highTemp"][i] for i in index]
E_l_Amitex_tEnd["highTemp"]=[E_l_Amitex_tEnd["highTemp"][i] for i in index]

plt.figure(figsize=[8,6],num="E_l vs FiberCount")
plt.scatter(fiberCount["lowTemp"], E_l_Amitex_t0["lowTemp"])
plt.scatter(fiberCount["highTemp"], E_l_Amitex_t0["highTemp"])
plt.xlabel("Fiber count")
plt.ylabel(r"$E_l$")

plt.ylim([0, 24])

plt.figure(figsize=[8,6],num="E_t vs FiberCount")
plt.scatter(fiberCount["lowTemp"], E_t_Amitex["lowTemp"])
plt.scatter(fiberCount["highTemp"], E_t_Amitex["highTemp"])
plt.xlabel("Fiber count")
plt.ylabel(r"$E_t$")

plt.ylim([0, 6])

mean_E_l_t0_vec    = {"lowTemp":[], "highTemp":[]}
mean_E_l_tEnd_vec    = {"lowTemp":[], "highTemp":[]}
stdError_E_l_t0_vec     = {"lowTemp":[], "highTemp":[]}
stdError_E_l_tEnd_vec     = {"lowTemp":[], "highTemp":[]}
fiberCount_vec  = {"lowTemp":[], "highTemp":[]}

mean_E_l_t0_vec["lowTemp"]=[np.mean(E_l_Amitex_t0["lowTemp"][:pos]) for pos in range(1,len(E_l_Amitex_t0["lowTemp"]))]
mean_E_l_t0_vec["highTemp"]=[np.mean(E_l_Amitex_t0["highTemp"][:pos]) for pos in range(1,len(E_l_Amitex_t0["highTemp"]))]

asymptote_E_l_t0={
    "lowTemp"   :[mean_E_l_t0_vec["lowTemp"][-1] ]*2, 
    "highTemp"  :[mean_E_l_t0_vec["highTemp"][-1]]*2
}


mean_E_l_tEnd_vec["lowTemp"]=[np.mean(E_l_Amitex_tEnd["lowTemp"][:pos]) for pos in range(1,len(E_l_Amitex_tEnd["lowTemp"]))]
mean_E_l_tEnd_vec["highTemp"]=[np.mean(E_l_Amitex_tEnd["highTemp"][:pos]) for pos in range(1,len(E_l_Amitex_tEnd["highTemp"]))]

stdError_E_l_t0_vec["lowTemp"]=[1.96*np.std(E_l_Amitex_t0["lowTemp"][:pos])/np.sqrt(pos) for pos in range(1,len(E_l_Amitex_t0["lowTemp"]))]
stdError_E_l_t0_vec["highTemp"]=[1.96*np.std(E_l_Amitex_t0["highTemp"][:pos])/np.sqrt(pos) for pos in range(1,len(E_l_Amitex_t0["highTemp"]))]
stdError_E_l_tEnd_vec["lowTemp"]=[1.96*np.std(E_l_Amitex_tEnd["lowTemp"][:pos])/np.sqrt(pos) for pos in range(1,len(E_l_Amitex_tEnd["lowTemp"]))]
stdError_E_l_tEnd_vec["highTemp"]=[1.96*np.std(E_l_Amitex_tEnd["highTemp"][:pos])/np.sqrt(pos) for pos in range(1,len(E_l_Amitex_tEnd["highTemp"]))]

fiberCount_vec["lowTemp"]=np.array([np.sum(fiberCount["lowTemp"][:pos]) for pos in range(1,len(fiberCount["lowTemp"]))])
fiberCount_vec["highTemp"]=np.array([np.sum(fiberCount["highTemp"][:pos]) for pos in range(1,len(fiberCount["highTemp"]))])

offset=100

fig, axs = plt.subplots(1, 2, figsize=[12,6],num="mean E_l vs FiberCount")

# axs[0].scatter(fiberCount_vec["lowTemp"], mean_E_l_t0_vec["lowTemp"])
if plot_error_bar:
    axs[0].errorbar(
        fiberCount_vec["lowTemp"],
        mean_E_l_t0_vec["lowTemp"],
        yerr=stdError_E_l_t0_vec["lowTemp"],
        elinewidth=2,
        color=color["lowTemp"],
        capsize=5.,
        fmt='o',
        label=r"$E_l(t=0)$, 21°C"
    )
    # axs[0].errorbar(
    #     fiberCount_vec["lowTemp"]+2*offset,
    #     mean_E_l_tEnd_vec["lowTemp"],
    #     yerr=std_E_l_tEnd_vec["lowTemp"],
    #     elinewidth=2,
    #     linestyle="--",
    #     color=color["lowTemp"],
    #     capsize=5.
    # )

fiberCount_asymptote={
    "lowTemp":[fiberCount_vec["lowTemp"][0], fiberCount_vec["lowTemp"][-1]],
    "highTemp":[fiberCount_vec["highTemp"][0], fiberCount_vec["highTemp"][-1]]
}


axs[0].plot(
    fiberCount_asymptote["lowTemp"],
    asymptote_E_l_t0["lowTemp"],
    linestyle="--",
    color=color["lowTemp"],
    label="Asymptotic trend"
    )

axs[0].set_xlabel("Fiber count")
axs[0].set_ylabel(r"$E_l(t=0)$ (GPa)")
axs[0].set_ylim([20, 25])


# axs[1].scatter(fiberCount_vec["highTemp"]+offset, mean_E_l_t0_vec["highTemp"])
if plot_error_bar:
    axs[1].errorbar(fiberCount_vec["highTemp"]+offset,
        mean_E_l_t0_vec["highTemp"],
        yerr=stdError_E_l_t0_vec["highTemp"],
        elinewidth=2,
        color=color["highTemp"],
        capsize=5.,
        fmt='o',
        label=r"$E_l(t=0)$, 120°C"
    )
    
    # axs[1].errorbar(
    #     fiberCount_vec["highTemp"]+3*+offset,
    #     mean_E_l_tEnd_vec["highTemp"],
    #     yerr=std_E_l_tEnd_vec["highTemp"],
    #     elinewidth=2,
    #     linestyle="--",
    #     color=color["highTemp"],
    #     capsize=5.
    # )

axs[1].plot(
    fiberCount_asymptote["highTemp"],
    asymptote_E_l_t0["highTemp"],
    linestyle="--",
    color=color["highTemp"],
    label="Asymptotic trend"
    )

axs[1].set_xlabel("Fiber count")
axs[1].set_ylabel(r"$E_l(t=0)$ (GPa)")
axs[1].set_ylim([20, 25])

plt.figtext(0.015,0.92,"a)",fontsize=28)
plt.figtext(0.51 ,0.92,"b)",fontsize=28)

plt.tight_layout()
axs[0].legend()
axs[1].legend()

saveFigure=True
outputPathFigure="/home/facu/Phd_Private/Redaction/Paper3_ThermoViscoelasticity_elsArticle/Images/RVE_CreepData/"

if saveFigure:
    plt.savefig(os.path.join(outputPathFigure,"meanE_l_vs_FiberCount.eps"),format='eps')



plt.figure(figsize=[8,6],num="STD E_l vs FiberCount")
plt.scatter(fiberCount_vec["lowTemp"], stdError_E_l_t0_vec["lowTemp"])
plt.scatter(fiberCount_vec["highTemp"], stdError_E_l_t0_vec["highTemp"])
plt.xlabel("Fiber count")
plt.ylabel("std E_l")

plt.show()