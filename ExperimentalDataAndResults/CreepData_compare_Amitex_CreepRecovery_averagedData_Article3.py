import os
from pandas import DataFrame, read_excel
import numpy as np

import scipy.interpolate

from processing_functions import readAmitexResults
from matplotlib import pyplot as plt

plt.rcParams.update({
    'font.size':22,
    "axes.formatter.limits":[-2,2],
    "font.family": "Times New Roman"
    })

from matplotlib.ticker import AutoMinorLocator   

from mpl_toolkits.axes_grid.inset_locator import inset_axes, InsetPosition,mark_inset

def errorFunc(xVec1,yVec1,xVec2,yVec2,xMax,titleStr=None,startPos=21,makeErrorPlots=False):

    # interpolate the first data pair onto the seconds x values
    stainInterpFunc = scipy.interpolate.interp1d(xVec1, yVec1)

    xInterp2=[x for x in xVec2 if x<=xMax]
    
    yVec1_interp=stainInterpFunc(xInterp2)

    xVec2_trunc=[x for x in xVec2 if x<=xMax]
    yVec2_trunc=[y for x,y in zip(xVec2,yVec2) if x<=xMax]


    eVec=[abs(y1-y2)/abs(y1) if (y1!=0 and y2!=0) else 0. for y1,y2 in zip(yVec1_interp[startPos:],yVec2_trunc[startPos:]) ]

    eMean=np.mean(eVec)

    if makeErrorPlots:
        plt.figure(num=f"error {titleStr}")
        plt.plot(xInterp2[startPos:],yVec1_interp[startPos:],label="interp")
        plt.plot(xVec1,yVec1,linestyle=":",label="exp")
        plt.plot(xVec2_trunc[startPos:],yVec2_trunc[startPos:],label="fft")

        plt.legend()

        plt.title("e={: 5.2%}".format(eMean))

    return eMean



commonPath="./CreepData_Vx30/"

outputPath="./"

plotList=[
    "dualScale",
    "dualScale_transverse",
]

plotIntralayer=True
plotInterLayer=True

makeInsetsTransverse=True
omitAx2=False
omitAx2_transverse=True

makeErrorPlots=False

saveFigures=True

################################################################################3

### results to compare against F samples

resultsSet={
    "dualScale":{
        "simulationResultsPath":"Homogenization_sampleF1_meso_large/CreepRecoveryResults_all_G_data_rightMatrix/",
     
        "resultsFiles":{
            "lowTemp" :"LowTemp/C_hom_averaged_no_porosity_rightMatrix_mesoscale_low_temp.txt",
            "highTemp":"HighTemp/C_hom_averaged_no_porosity_rightMatrix_mesoscale_high_temp.txt"
        },

        "experimentalResultsPath":"./CreepData_Vx30/Formated_for_Matlab/Averaged_F_",
        "timeShiftLowTemp":6.
    },

    "dualScale_transverse":{
        "simulationResultsPath":"Homogenization_sampleF1_meso_large/CreepRecoveryResults_all_G_data_rightMatrix/",
        
        "resultsFiles":{
            "lowTemp" :"LowTemp_transverse/C_hom_averaged_no_porosity_rightMatrix_mesoscale_low_temp_transverse.txt",
            "highTemp":"HighTemp_transverse/C_hom_averaged_no_porosity_rightMatrix_mesoscale_high_temp_transverse.txt"
        },

        "experimentalResultsPath":"./CreepData_Vx30/Formated_for_Matlab/Averaged_H_",
        "timeShiftLowTemp":6.
    },
}

sampleLabelList=[
    "lowTemp",
    "highTemp"
]

legendLabel={
    "lowTemp":"21°C",
    "highTemp":"120°C",
}

rgb_lowTemp =(0.00,0.35,0.70)
rgb_highTemp=(1.00,0.25,0.10)

colorBySample={
    "lowTemp":rgb_lowTemp, #medium teal blue
    "highTemp":rgb_highTemp,#orange-red      
}

listylesPerType={
    "experimental"  :"solid",
    "zz"            :"dotted",
    "xx"            :"dotted", # "dashed",
    "yy"            :"dotted", # "dashdot"
}

linewidthExperimental=1.5
linewidthSimulated=2.

fig, (ax1_left, ax1_right) = plt.subplots(1, 2,figsize=[16,8],num="Comparission simulation to averaged experimental results: {}".format(plotList[0]))

poissonBySample={}
poissonBySample_transverse={}
timeBySample_shiftedKEEP={}
timeBySample_shifted_transverseKEEP={}

axList=[ax1_left,ax1_right]

FFTresultsDict={}

errorFit_axial={}
errorFit_trans={}

for iFig,dataLabel in enumerate(plotList):

    if "_transverse" in dataLabel:
        labelAxial="Strain_xx"
        labelTrans="Strain_zz"
        makeInsets=makeInsetsTransverse
    else:
        labelAxial="Strain_zz"
        labelTrans="Strain_xx"
        makeInsets=True

    if dataLabel not in errorFit_axial.keys():
        errorFit_axial[dataLabel]={}
        errorFit_trans[dataLabel]={}

    maxAxialVal=0.
    minAxialVal=9999.

    maxTransVal=0.
    minTransVal=-9999.

    simulationResultsPath   =resultsSet[dataLabel]["simulationResultsPath"  ]
    experimentalResultsPath =resultsSet[dataLabel]["experimentalResultsPath"]
    resultsFiles            =resultsSet[dataLabel]["resultsFiles"]
    timeShiftLowTemp        =resultsSet[dataLabel]["timeShiftLowTemp"]

    FFTresultsDict[dataLabel]={}

    for key,path in resultsFiles.items():
        FFTresultsDict[dataLabel][key]=readAmitexResults(os.path.join(commonPath,simulationResultsPath,path))

        FFTresultsDict[dataLabel][key]["Poisson_l"]=-FFTresultsDict[dataLabel][key][labelTrans]/FFTresultsDict[dataLabel][key][labelAxial]

    if "Time(s)" in FFTresultsDict[dataLabel]["lowTemp"].keys():
        timeKey="Time(s)"
        timeFactor=60
    elif "Time(m)" in FFTresultsDict[dataLabel]["lowTemp"].keys():
        timeKey="Time(m)"
        timeFactor=1

    FFTresultsDict[dataLabel]["lowTemp"][timeKey]/=timeFactor
    FFTresultsDict[dataLabel]["highTemp"][timeKey]/=timeFactor

    ###########################################################################

    ax1=axList[iFig]

    if makeInsets:

        if "_transverse" in dataLabel:
            if not omitAx2_transverse:
                ax2 =fig.add_axes([0.53/2+iFig*0.5,0.48,0.423/2,0.28])
            ax3 =fig.add_axes(    [0.53/2+iFig*0.5,0.3 ,0.423/2,0.15])

        else:
            if not omitAx2:
                ax2 =fig.add_axes([0.55/2+iFig*0.5,0.45,0.42/2,0.4 ])
            ax3 =fig.add_axes(    [0.55/2+iFig*0.5,0.17,0.42/2,0.15])
            
        # # Mark the region corresponding to the inset axes on ax1 and draw lines
        # # in grey linking the two axes.
        skipAx2=False

        if "_transverse" in dataLabel:
            if omitAx2_transverse:
                skipAx2=True
            else:
                mark_inset(ax1, ax2, loc1=1, loc2=3, fc="none", ec='0.8')

            mark_inset(ax1, ax3, loc1=2, loc2=4, fc="none", ec='0.8')

        else:
            if omitAx2:
                skipAx2=True
            else:
                mark_inset(ax1, ax2, loc1=2, loc2=3, fc="none", ec='0.8')
    
            mark_inset(ax1, ax3, loc1=2, loc2=3, fc="none", ec='0.8')


    timeBySample={}
    stressBySample={}
    strainAxialBySample={}
    strainTransBySample={}

    for sampleLabel in sampleLabelList:

        fileDataFrame = read_excel(os.path.join(experimentalResultsPath+sampleLabel,"CreepData-1.0MPa.xlsx"),engine='openpyxl')

        timeBySample  [sampleLabel]=np.array(fileDataFrame.Time)
        stressBySample[sampleLabel]=np.array(fileDataFrame.Stress)
        strainAxialBySample [sampleLabel]=np.array(fileDataFrame.Strain_axial)
        strainTransBySample [sampleLabel]=np.array(fileDataFrame.Strain_trans)

        if sampleLabel=="lowTemp":
            timeBySample_shifted=timeBySample[sampleLabel]-timeShiftLowTemp 
        else:
            timeBySample_shifted=timeBySample[sampleLabel]

        if "_transverse" in dataLabel:
            poissonBySample_transverse[sampleLabel]=-strainTransBySample [sampleLabel]/strainAxialBySample [sampleLabel]
            timeBySample_shifted_transverseKEEP[sampleLabel]=timeBySample[sampleLabel]
            
        else:
            poissonBySample[sampleLabel]=-strainTransBySample [sampleLabel]/strainAxialBySample [sampleLabel]
            timeBySample_shiftedKEEP[sampleLabel]=timeBySample[sampleLabel]

        ax1.plot(
            timeBySample_shifted/60,
            strainAxialBySample[sampleLabel],
            label="Experiment, {}".format(legendLabel[sampleLabel]),
            linestyle=listylesPerType["experimental"],
            color=colorBySample[sampleLabel],
            linewidth=linewidthExperimental
            )

        errorFit_axial[dataLabel][sampleLabel]=errorFunc(
            timeBySample_shifted/60,
            strainAxialBySample[sampleLabel],
            FFTresultsDict[dataLabel][sampleLabel][timeKey],
            FFTresultsDict[dataLabel][sampleLabel][labelAxial],
            xMax=8,
            titleStr=f"{dataLabel} {sampleLabel} axial",
            makeErrorPlots=makeErrorPlots
        )

        errorFit_trans[dataLabel][sampleLabel]=errorFunc(
            timeBySample_shifted/60,
            strainTransBySample[sampleLabel],
            FFTresultsDict[dataLabel][sampleLabel][timeKey],
            FFTresultsDict[dataLabel][sampleLabel][labelTrans],
            xMax=8,
            titleStr=f"{dataLabel} {sampleLabel} trans",
            makeErrorPlots=makeErrorPlots
        )

        if makeInsets:
            if not skipAx2:
                ax2.plot(
                    timeBySample_shifted/60,
                    strainAxialBySample[sampleLabel],
                    label="Experiment, {}".format(legendLabel[sampleLabel]),
                    linestyle=listylesPerType["experimental"],
                    color=colorBySample[sampleLabel],
                    linewidth=linewidthExperimental
                    )

            maxAxialVal=max(max(strainAxialBySample[sampleLabel]),maxAxialVal)
            minAxialVal=min(max(strainAxialBySample[sampleLabel]),minAxialVal)


        if plotIntralayer:
            ax1.plot(
                timeBySample_shifted/60,
                strainTransBySample[sampleLabel],
                # label="Intra-layer, experiment" if sampleLabel=="lowTemp" else "",
                linestyle=listylesPerType["experimental"],
                color=colorBySample[sampleLabel],
                linewidth=linewidthExperimental
            )

            if makeInsets:
                ax3.plot(
                    timeBySample_shifted/60,
                    strainTransBySample[sampleLabel],
                    linestyle=listylesPerType["experimental"],
                    color=colorBySample[sampleLabel],
                    linewidth=linewidthExperimental
                )

                maxTransVal=min(min(strainTransBySample[sampleLabel]),maxTransVal)
                minTransVal=max(min(strainTransBySample[sampleLabel]),minTransVal)

    ax1.plot(
        FFTresultsDict[dataLabel]["lowTemp"][timeKey],
        FFTresultsDict[dataLabel]["lowTemp"][labelAxial],
        label="FFT, {}".format(legendLabel["lowTemp"]),
        linestyle=listylesPerType["zz"],
        linewidth=linewidthSimulated,
        color=rgb_lowTemp
    )

    if makeInsets:
        if not skipAx2:
            ax2.plot(
                FFTresultsDict[dataLabel]["lowTemp"][timeKey],
                FFTresultsDict[dataLabel]["lowTemp"][labelAxial],
                label="FFT, {}".format(legendLabel["lowTemp"]),
                linestyle=listylesPerType["zz"],
                linewidth=linewidthSimulated,
                color=rgb_lowTemp
            )

        maxAxialVal=max(max(FFTresultsDict[dataLabel]["lowTemp"][labelAxial]),maxAxialVal)
        minAxialVal=min(max(FFTresultsDict[dataLabel]["lowTemp"][labelAxial]),minAxialVal)

    if plotIntralayer:
        ax1.plot(
            FFTresultsDict[dataLabel]["lowTemp"][timeKey],
            FFTresultsDict[dataLabel]["lowTemp"][labelTrans],
            linestyle=listylesPerType["xx"],
            linewidth=linewidthSimulated,
            color=rgb_lowTemp
        )

        if makeInsets:
            ax3.plot(
                FFTresultsDict[dataLabel]["lowTemp"][timeKey],
                FFTresultsDict[dataLabel]["lowTemp"][labelTrans],
                linestyle=listylesPerType["xx"],
                linewidth=linewidthSimulated,
                color=rgb_lowTemp
            )

    if plotInterLayer:
        ax1.plot(
            FFTresultsDict[dataLabel]["lowTemp"][timeKey],
            FFTresultsDict[dataLabel]["lowTemp"]["Strain_yy"],
            linestyle=listylesPerType["yy"],
            linewidth=linewidthSimulated,
            color=rgb_lowTemp
        )


    ax1.plot(
        FFTresultsDict[dataLabel]["highTemp"][timeKey],
        FFTresultsDict[dataLabel]["highTemp"][labelAxial],
        label="FFT, {}".format(legendLabel["highTemp"]),
        linestyle=listylesPerType["zz"],
        linewidth=linewidthSimulated,
        color=rgb_highTemp
    )

    if makeInsets and not skipAx2:
        ax2.plot(
            FFTresultsDict[dataLabel]["highTemp"][timeKey],
            FFTresultsDict[dataLabel]["highTemp"][labelAxial],
            label="FFT, {}".format(legendLabel["highTemp"]),
            linestyle=listylesPerType["zz"],
            linewidth=linewidthSimulated,
            color=rgb_highTemp
        )

    if plotIntralayer:
        ax1.plot(
            FFTresultsDict[dataLabel]["highTemp"][timeKey],
            FFTresultsDict[dataLabel]["highTemp"][labelTrans],
            linestyle=listylesPerType["xx"],
            linewidth=linewidthSimulated,
            color=rgb_highTemp
        )

        if makeInsets:
            ax3.plot(
                FFTresultsDict[dataLabel]["highTemp"][timeKey],
                FFTresultsDict[dataLabel]["highTemp"][labelTrans],
                linestyle=listylesPerType["xx"],
                linewidth=linewidthSimulated,
                color=rgb_highTemp
            )

        maxTransVal=min(min(FFTresultsDict[dataLabel]["highTemp"][labelTrans]),maxTransVal)
        minTransVal=max(min(FFTresultsDict[dataLabel]["highTemp"][labelTrans]),minTransVal)

    if plotInterLayer: 
        ax1.plot(
            FFTresultsDict[dataLabel]["highTemp"][timeKey],
            FFTresultsDict[dataLabel]["highTemp"]["Strain_yy"],
            linestyle=listylesPerType["yy"],
            linewidth=linewidthSimulated,
            color=rgb_highTemp
        )

    if iFig==1:
        ax1.legend(fontsize=22)

    ax1.set_xlabel("Time (min)")
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.xaxis.set_minor_locator(AutoMinorLocator())

    ax1.set_ylabel("Strain (m/m)")

    ax1.set_xlim([-0.5,24])

    # this is to get 10^-5 instead of 1e-5
    ax1.yaxis.major.formatter._useMathText = True
    ax2.yaxis.major.formatter._useMathText = True
    ax3.yaxis.major.formatter._useMathText = True


    if "_transverse" not in dataLabel:
        ax1.set_ylim([minTransVal*1.9,maxAxialVal*1.3])

    if makeInsets:
        if not skipAx2:
            ax2.set_xlim([-0.2,8.5])
            ax2.set_ylim([minAxialVal*0.98,maxAxialVal*1.005])

        ax3.set_xlim([-0.2,8.5])
        if "_transverse" in dataLabel:
            ax3.set_ylim([minTransVal*0.6,maxTransVal*1.05])
        else:
            ax3.set_ylim([minTransVal*0.92,maxTransVal*1.010])

        ax3.invert_yaxis()

    ax1.plot([-1,24],[0,0],color=(0.3,0.3,0.3),linestyle='-',linewidth=0.4)

    plt.figure(num="Comparission simulation to averaged experimental results: {}".format(plotList[0]))

    fig.tight_layout()

    plt.figtext(0.015,0.95,"a)",fontsize=28)
    plt.figtext(0.515,0.95,"b)",fontsize=28)

    plt.figtext(0.082,0.5 ,"axial strain",fontsize=20)
    plt.figtext(0.082,0.3,"transverse strain",fontsize=20)

    plt.figtext(0.582,0.5 ,"axial strain",fontsize=20)
    plt.figtext(0.582,0.182,"transverse strain",fontsize=20)

##############
## error estimation printout

for iFig,dataLabel in enumerate(plotList):
    for sampleLabel in ["lowTemp","highTemp"]:

        print("\nDataLabel: {} \tSampleLabel: {} \tAxial error value: {:5.3%}".format(dataLabel,sampleLabel,errorFit_axial[dataLabel][sampleLabel]) )
        print("\nDataLabel: {} \tSampleLabel: {} \tTrans error value: {:5.3%}".format(dataLabel,sampleLabel,errorFit_trans[dataLabel][sampleLabel]) )

timeBySample_shifted_transverseKEEP[sampleLabel]
            
timeBySample_shiftedKEEP[sampleLabel]

# plot Poisson's ratio

figPoisson, axPoisson=plt.subplots(1, 1,figsize=[8,6],num="Poisson's ratio: Experimental results: {}".format(plotList[0]))

errorPoisson={}

plotPoisson_FFT=False

for iFig,dataLabel in enumerate(plotList):

    errorPoisson[dataLabel]={}

    for sampleLabel in sampleLabelList:

        if "_transverse" in dataLabel:
            pass

        else:

            tMin=0.15
            tMax=8.

            xVal=[t for t in timeBySample_shiftedKEEP[sampleLabel]/60 if tMin<t<tMax]
            yVal=[p for (p,t) in zip(poissonBySample[sampleLabel], timeBySample_shiftedKEEP[sampleLabel]/60) if tMin<t<tMax]

            axPoisson.plot(
                xVal,
                yVal,              
                linestyle=listylesPerType["experimental"],
                color=colorBySample[sampleLabel],
                linewidth=linewidthExperimental,
                label="Experiment, {}".format(legendLabel[sampleLabel])
            )

            meanVal=np.ones(len(yVal))*np.mean(yVal)

            errorPoisson[dataLabel][sampleLabel]=errorFunc(xVal, yVal, xVal, meanVal, xMax=10,makeErrorPlots=makeErrorPlots)

            print("\nDataLabel: {} \tSampleLabel: {} \tPoisson error value: {:5.3%}".format(
                dataLabel,
                sampleLabel,
                errorPoisson[dataLabel][sampleLabel]
                )
            )

            axPoisson.plot(
                xVal,
                meanVal,              
                linestyle="-.",
                color=colorBySample[sampleLabel],
                linewidth=linewidthSimulated,
                label="Constant approx., {}".format(legendLabel[sampleLabel])
            )

            if plotPoisson_FFT:
                xVal=[t for t in FFTresultsDict[dataLabel][sampleLabel]["Time(m)"] if tMin<t<tMax]
                yVal=[p for (p,t) in zip(FFTresultsDict[dataLabel][sampleLabel]["Poisson_l"], FFTresultsDict[dataLabel][sampleLabel]["Time(m)"]) if tMin<t<tMax]

                axPoisson.plot(
                    xVal,
                    yVal,              
                    linestyle=listylesPerType["zz"],
                    color=colorBySample[sampleLabel],
                    linewidth=linewidthSimulated,
                    label="FFT, {}".format(legendLabel[sampleLabel])
                )

# handles,labels = axPoisson.get_legend_handles_labels()

# handles = [
#     handles[0], 
#     handles[2], 
#     handles[1], 
#     handles[3],
# ]

# labels = [
#     labels[0], 
#     labels[2], 
#     labels[1], 
#     labels[3],
# ]

# axPoisson.legend(handles,labels)
axPoisson.legend()

axPoisson.set_ylim([0.33,0.38])

axPoisson.set_xlabel("Time (m)")
axPoisson.set_ylabel(r"$\nu_l$")

if saveFigures:

    fig.savefig(
        os.path.join(outputPath,"{}_combined.pdf".format(dataLabel)), 
        transparent=True
    )

    figPoisson.savefig(
        os.path.join(outputPath,"Poisson.pdf"), 
        transparent=True
    )


plt.show()