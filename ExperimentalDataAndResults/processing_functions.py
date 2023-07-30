import os
from tracemalloc import start
from matplotlib.cbook import flatten
import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib.ticker as mtick

from matplotlib.ticker import AutoMinorLocator   


plt.rcParams["axes.formatter.limits"]=[-2,2]

from pandas import DataFrame, ExcelWriter


def convert_E_nu_to_kappa_mu(E,nu):
    kappa   =E/(3*(1-2*nu))
    mu      =E/(2*(1+nu))
    return kappa,mu

def processData(
    Time,
    Force,
    Strain_axial,
    Strain_trans,
    ForceThresholdHigh,
    ForceThresholdLow,
    baseLineVerticalOffset_axial,
    baseLineVerticalOffset_trans,
    includeTimeIntervalBeforeAppliedLoad,
    areaCrossSection,
    T_correction=None,
    plotTempCorrection=False,
    plotStress=False,
    titleStr="",
    plotSeparateLoadings=False,
    normalizeBy="StrainAtFullLoad",
    textProperiesAtMaxStrain=False,
    exportPath=None,
    flattenTransverseStrainByParabola=False,
    excludeLoading=[],
    factorLengthOfLoadingAfterRelease=3,
    textAnnotations=True
    ):

    Strain_axial=np.array(Strain_axial)+baseLineVerticalOffset_axial
    Strain_trans=np.array(Strain_trans)+baseLineVerticalOffset_trans

    if flattenTransverseStrainByParabola:

        thresh=60

        posForceThresh=np.where(np.array(Force)<thresh)[0]

        polynomial=np.polyfit(np.array(Time)[posForceThresh],np.array(Strain_trans)[posForceThresh],deg=2)

        polyStrain=np.polyval(polynomial,Time)

        plt.figure(figsize=[8,6],num="Polyfit of strain transverse")
        plt.plot(Time,Strain_trans,label="input strain")

        plt.plot(Time,polyStrain,label="Polynomial")

        plt.plot(Time,Strain_trans-polyStrain,label="Strain with drift removed")

        Strain_trans-=polyStrain

        plt.legend()

        plt.grid()

        # plt.show()


    timeOffsetAfterLoadApplication=1    #to make sure loading level has stabilized
    timeOffsetAfterLoadDrop=1           #to make sure loading level has stabilized

    diffForce=np.array([Force[i+1]-Force[i] for i in range(len(Force)-1) ])

    #find positions where Force above thresh and derivative is small (after stabilization of load)
    positionsFullLoad=np.array([val for val in range(len(diffForce)) if abs(diffForce[val])<1. and Force[val]>ForceThresholdHigh])

    #identify gaps in stable regions above thresh
    diffPos=np.array([positionsFullLoad[i+1]-positionsFullLoad[i] for i in range(len(positionsFullLoad)-1) ])

    changePos=np.where(diffPos>100)[0]

    # this yields t1
    startPos=np.array(positionsFullLoad[[timeOffsetAfterLoadApplication]+[val+timeOffsetAfterLoadApplication for val in changePos]])

    #####

    #drops only valid after first rise (startPos)
    positionsNoLoad=np.array([val for val in range(len(diffForce)) \
        if val>startPos[0] and abs(diffForce[val])<1. and Force[val]<ForceThresholdLow])

    diffPos=np.array([positionsNoLoad[i+1]-positionsNoLoad[i] for i in range(len(positionsNoLoad)-1) ])

    changePosDecreases=np.where(diffPos>100)[0]

    #this yields t3
    dropPos=np.array(positionsNoLoad[[0]+[val+timeOffsetAfterLoadApplication for val in changePosDecreases]])

    # this yields t2
    endOfLoadPos=[]
    for d in dropPos:
        indexFullLoad=np.where(positionsFullLoad<d)[0][-1]

        # if positionsFullLoad[indexFullLoad] in endOfLoadPos:
            # raise ValueError("Cant find t2")
        endOfLoadPos.append(positionsFullLoad[indexFullLoad]) 

    # endOfLoadPos=np.array(positionsFullLoad[[offsetPeakStress-3]+[val+offsetPeakStress-3 for val in changePos]])


    Stress=[]
    StressStr=[]

    for p in startPos:
        Stress.append(Force[p]/areaCrossSection)
        StressStr.append("{:4.0f}".format(round(Force[p]/areaCrossSection)))

    data={
        "StressLabel":Stress,
        "Time"  :[],
        "t1":[],    #initial time with fully applied load
        "t2":[],    #last moment with fully applied load
        "t3":[],    #time of fully removed load
        "startTime":[],
        "Stress":[],
        "Force" :[],
        "Strain_axial":[],    
        "Strain_axial_corrected":[],    
        "Strain_axial_atFullLoad":[],
        "Strain_axial_atMaxStrain":[],
        "Strain_axial_offset":[],
        "Strain_axial_normalized":[],
        "Strain_trans":[],
        "Strain_trans_atFullLoad":[],
        "Strain_trans_max":[],
        "Strain_trans_offset":[],
        "Strain_trans_normalized":[],
        "Modulus_axial_atFullLoad":[],
        "Modulus_axial_atMaxStrain":[],
        "Poisson_axial_atFullLoad":[]
    }

    index=0
    startPos=list(startPos)+[len(Time)]

    endOfRecovery=[]#for export, keep only data for no load for the same duration as loading
    
    for i,s in enumerate(startPos[:-1]):
        if s<endOfLoadPos[i]<dropPos[i]<startPos[i+1]:
            intervalLoadApplied=dropPos[i]-s
            endOfRecovery.append(int(s+intervalLoadApplied*factorLengthOfLoadingAfterRelease))
        else:
            raise ValueError("square wave mis-identified")

    StressTimeSeries=[]

    for iStart,iEndLoad,iDrop,iEnd in zip(startPos[:-1],endOfLoadPos,dropPos,endOfRecovery):
        data["Time"]        .append(np.array(Time        [max(iStart-includeTimeIntervalBeforeAppliedLoad,0):iEnd-includeTimeIntervalBeforeAppliedLoad]))
        data["t1"].append(Time[iStart])
        data["t2"].append(Time[iEndLoad])
        data["t3"].append(Time[iDrop+timeOffsetAfterLoadDrop])

        data["Force"]       .append(np.array(Force[max(iStart-includeTimeIntervalBeforeAppliedLoad,0):iEnd-includeTimeIntervalBeforeAppliedLoad]))
        data["Stress"]      .append(data["Force"][-1] /areaCrossSection)
        maxStressVal=np.max(data["Stress"][index])

        initialLoadTime=0.01
        minStressVal=0.0001 #numerical problems can occur if left at zero

        if len(StressTimeSeries)==0:
            #time series should start at first loading for all specimens
            globalTimeOffset=Time[iStart]/60-initialLoadTime
        else:
            #first loading value should be non zero. afterwards, sharp climb from near zero

            StressTimeSeries.append((data["t1"][-1]/60-initialLoadTime-globalTimeOffset,minStressVal))
        
        StressTimeSeries.append((data["t1"][-1]/60-globalTimeOffset,maxStressVal))
        StressTimeSeries.append((data["t2"][-1]/60-globalTimeOffset,maxStressVal))
        StressTimeSeries.append((data["t3"][-1]/60-globalTimeOffset,minStressVal)) #Amitex crashes for stress value of 0.

        timeOffset_zeroThisSeries=data["Time"][-1][0] # each xlsx load case starts a t=0

        data["Time"][-1]-=timeOffset_zeroThisSeries#bring start time to 0 for each loading step
        data["t1"]  [-1]-=timeOffset_zeroThisSeries
        data["t2"]  [-1]-=timeOffset_zeroThisSeries
        data["t3"]  [-1]-=timeOffset_zeroThisSeries

        Strain_axial_offset=Strain_axial[max(iStart-includeTimeIntervalBeforeAppliedLoad,0)]
        data["Strain_axial_offset"]     .append(Strain_axial_offset)
        data["Strain_axial"]            .append(np.array(Strain_axial[max(iStart-includeTimeIntervalBeforeAppliedLoad,0):iEnd-includeTimeIntervalBeforeAppliedLoad])-Strain_axial_offset)
        if T_correction is not None:
            data["Strain_axial_corrected"]  .append(data["Strain_axial"][index]-T_correction[max(iStart-includeTimeIntervalBeforeAppliedLoad,0):iEnd-includeTimeIntervalBeforeAppliedLoad])

        data["Strain_axial_atFullLoad"] .append(Strain_axial[iStart+timeOffsetAfterLoadApplication]-Strain_axial_offset)
        data["Strain_axial_atMaxStrain"].append(max(data["Strain_axial"][index]))

        if T_correction is not None:
            if normalizeBy=="StrainAtFullLoad":
                data["Strain_axial_normalized"] .append(data["Strain_axial_corrected"][index]/data["Strain_axial_atFullLoad"][index])
            elif normalizeBy=="maxStrain":
                data["Strain_axial_normalized"] .append(data["Strain_axial_corrected"][index]/data["Strain_axial_atMaxStrain"][index])
            elif normalizeBy=="maxStress":
                data["Strain_axial_normalized"] .append(data["Strain_axial_corrected"][index]/maxStressVal)
            else:
                raise ValueError("not implemented: normalizeBy: {}".format(normalizeBy))
        else:
            if normalizeBy=="StrainAtFullLoad":
                data["Strain_axial_normalized"] .append(data["Strain_axial"][index]/data["Strain_axial_atFullLoad"][index])
            elif normalizeBy=="maxStrain":
                data["Strain_axial_normalized"] .append(data["Strain_axial"][index]/data["Strain_axial_atMaxStrain"][index])
            elif normalizeBy=="maxStress":
                data["Strain_axial_normalized"] .append(data["Strain_axial"][index]/maxStressVal)
            else:
                raise ValueError("not implemented: normalizeBy: {}".format(normalizeBy))

        data["Modulus_axial_atFullLoad"].append(data["StressLabel"][index]/data["Strain_axial_atFullLoad"][-1])
        data["Modulus_axial_atMaxStrain"].append(data["StressLabel"][index]/data["Strain_axial_atMaxStrain"][-1])


        Strain_trans_offset=Strain_trans[max(iStart-includeTimeIntervalBeforeAppliedLoad,0)]
        data["Strain_trans_offset"].append(Strain_trans_offset)
        data["Strain_trans"].append(np.array(Strain_trans[max(iStart-includeTimeIntervalBeforeAppliedLoad,0):iEnd-includeTimeIntervalBeforeAppliedLoad])-Strain_trans_offset)
        data["Strain_trans_atFullLoad"].append(Strain_trans[iStart+timeOffsetAfterLoadApplication]-Strain_trans_offset)
        data["Strain_trans_max"].append(min(data["Strain_trans"][index]))
        if normalizeBy=="StrainAtFullLoad":
            data["Strain_trans_normalized"] .append(data["Strain_trans"][index]/data["Strain_trans_atFullLoad"][index])
        elif normalizeBy=="maxStrain":
            data["Strain_trans_normalized"] .append(data["Strain_trans"][index]/data["Strain_trans_max"][index])
        elif normalizeBy=="maxStress":
            data["Strain_trans_normalized"] .append(data["Strain_trans"][index]/maxStressVal)
        else:
            raise ValueError("not implemented: normalizeBy: {}".format(normalizeBy))

        data["Strain_trans_normalized"].append(data["Strain_trans"][index]/data["Strain_trans_atFullLoad"][index])

        data["Poisson_axial_atFullLoad"].append(-data["Strain_trans_atFullLoad"][-1]/data["Strain_axial_atFullLoad"][-1])

        if plotSeparateLoadings:

            plt.figure(figsize=[12,8],num='Stress={:8.4f} MPa_{}'.format(data["StressLabel"][index],titleStr))
            plt.plot(data["Time"][index]/60,data["Strain_axial"][index],label="Strain_axial")
            if T_correction is not None:
                plt.plot(data["Time"][index]/60,data["Strain_axial_corrected"][index],label="Strain_axial_corrected")
            plt.plot(data["Time"][index]/60,data["Strain_trans"][index],label="Strain_trans")

            plt.scatter(data["t1"][index]/60,data["Strain_axial"][index][includeTimeIntervalBeforeAppliedLoad+timeOffsetAfterLoadApplication],s=40,c="r")
            plt.scatter(data["t2"][index]/60,data["Strain_axial"][index][includeTimeIntervalBeforeAppliedLoad+iEndLoad-iStart],s=40,c="r")
            plt.scatter(data["t3"][index]/60,data["Strain_axial"][index][includeTimeIntervalBeforeAppliedLoad+iDrop-iStart+timeOffsetAfterLoadDrop],s=40,c="r")

            plt.legend()
            plt.xlabel("Time (min)")

        if exportPath is not None and round(data["StressLabel"][index]) not in excludeLoading:
            dataExport={
                "Time"  :data["Time"][index],
                "Stress":data["Stress"][index],
                "Strain_axial":data["Strain_axial"][index],
                "Strain_trans":data["Strain_trans"][index]
            }

            # Create a Pandas dataframe from the data.
            myDataFrame = DataFrame(dataExport)

            filenameOutput="CreepData-{:3.1f}MPa.xlsx".format(data["StressLabel"][index])

            # Create a Pandas Excel writer using XlsxWriter as the engine.
            writer = ExcelWriter(os.path.join(exportPath,filenameOutput), engine='xlsxwriter')

            # Convert the dataframe to an XlsxWriter Excel object.
            myDataFrame.to_excel(writer, sheet_name='Sheet1')

            # Close the Pandas Excel writer and output the Excel file.
            writer.save()

            E=data["Modulus_axial_atFullLoad"][index]
            nu=0.373#data["Poisson_axial_atFullLoad"][index]

            kappa,mu=convert_E_nu_to_kappa_mu(E, nu)

            #make separate file containing HeavisideTimes and Moduli
            dataHeaviside={
                "Heaviside_times":[
                    data["t1"][index],
                    data["t2"][index],
                    data["t3"][index]
                    ],
                "Modulus_axial_atFullLoad"  :[E,    0,0],# columns are 3 values high
                "Poisson_axial_atFullLoad"  :[nu,   0,0],
                "kappa"                     :[kappa,0,0],
                "mu"                        :[mu,   0,0]
            }
            
            # Create a Pandas dataframe from the data.
            myDataFrame = DataFrame(dataHeaviside)

            filenameOutput="CreepData-{:3.1f}MPa_HeavisideTimes.xlsx".format(data["StressLabel"][index])

            # Create a Pandas Excel writer using XlsxWriter as the engine.
            writer = ExcelWriter(os.path.join(exportPath,filenameOutput), engine='xlsxwriter')

            # Convert the dataframe to an XlsxWriter Excel object.
            myDataFrame.to_excel(writer, sheet_name='Sheet1')

            # Close the Pandas Excel writer and output the Excel file.
            writer.save()

        index+=1

    if exportPath is not None:

        # add single entry for last recovery, 40 min after last entry

        lastRecoveryTime=40

        StressTimeSeries.append( (StressTimeSeries[-1][0]+lastRecoveryTime,minStressVal ))

        with open(os.path.join(exportPath,"StressTimeSeries.txt"),"w") as f:
            f.write("StressTimeSeries=[\n")
            for t,s in StressTimeSeries:
                f.write("\t({: >8.4f}\t,{: >8.4f}),\n".format(t,s))

            f.write("]")

        dataJSON={
            "Time"          :list(np.array(Time)/60),
            "Stress"        :list(np.array(Force)/areaCrossSection),
            "Strain_axial"  :list(Strain_axial),
            "Strain_trans"  :list(Strain_trans),
        }

        with open(os.path.join(exportPath,"StressStrainProcessed.json"),"w") as f:
            json.dump(dataJSON,f,indent=4)

    print("\n Force(N) \tStress (MPa),\tE_l (first loading) (GPa),\tE_l(t) (max strain) (GPa),\tPoisson's ratio")
    for F,s,E,Ecreep,v in zip(
        [Force[p] for p in startPos[:-1]],
        Stress,
        data["Modulus_axial_atFullLoad"],
        data["Modulus_axial_atMaxStrain"],
        data["Poisson_axial_atFullLoad"]):

        print("{: >8.2f},\t{: >8.2f},\t\t{: >4.2f},\t\t\t\t{: >4.2f},\t\t\t\t{:>4.2f}".format(F,s,E/1000,Ecreep/1000,v))

    fig, (axSequence, axNorm) = plt.subplots(1, 2,figsize=[18,8],num="CreepTest_rawData_{}".format(titleStr))

    positionsLeft=[0.05, 0.1, 0.45, 0.82]

    axSequence.set_position(positionsLeft)
    axNorm.    set_position([0.65,0.1, 0.3,  0.82])

    axSequence.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))

    axSequence.yaxis.major.formatter._useMathText = True
    axNorm    .yaxis.major.formatter._useMathText = True


    Time=np.array(Time)/60

    lines=[
        axSequence.plot(Time,np.array(Strain_axial),     label=r"$\epsilon_{axial}$"),
        axSequence.plot(Time,np.array(Strain_trans),     label=r"$\epsilon_{transverse}$"),
    ]

    if plotTempCorrection:
        lines.extend(
            [
                axSequence.plot(Time,T_correction,label="T_correction"),
                axSequence.plot(Time,np.array(Strain_axial)+baseLineVerticalOffset_axial-T_correction,label="Strain_axial_corrected")
            ]
        )

    # axSequence=plt.gca()

    if textAnnotations:
        scatter=axSequence.scatter(
            Time[startPos[:-1]],
            [a+b for a,b in zip(data["Strain_axial_atFullLoad"],data["Strain_axial_offset"]) ],
            s=50,
            c="r",
            label="at first loading"
        )

        lines.append(scatter.legend_elements(prop="sizes", alpha=0.6)[0])

        lines[-1][0]._color="r" #HACK


    if textAnnotations:
        for x,y,s in zip(     
            Time[startPos[:-1]],
            [a+b for a,b in zip(data["Strain_axial_atFullLoad"],data["Strain_axial_offset"]) ],
            data["Modulus_axial_atFullLoad"]
            ):

            axSequence.text(x-0.2,y*1.1,r"$E_l$={:4.2f} GPa".format(s/1000),c="r")

        for x,y,s in zip(     
            Time[startPos[:-1]],
            [a+b for a,b in zip(data["Strain_trans_atFullLoad"],data["Strain_trans_offset"]) ],
            data["Poisson_axial_atFullLoad"]
            ):

            axSequence.text(x-0.2,y*1.2,r"$\nu$={:4.2f}".format(s),c="C2")

    if plotStress:
        axRight = axSequence.twinx()
        axRight.set_position(positionsLeft)


        Stress=np.array(Force)/areaCrossSection
        lines.append(axRight.plot(Time,Stress,"--",color="C2",label=r"$\sigma$"))

        ax1_ylims = axSequence.axes.get_ylim()           # Find y-axis limits set by the plotter
        ax1_yratio = ax1_ylims[0] / ax1_ylims[1]  # Calculate ratio of lowest limit to highest limit

        ax2_ylims = axRight.axes.get_ylim()           # Find y-axis limits set by the plotter
        ax2_yratio = ax2_ylims[0] / ax2_ylims[1]  # Calculate ratio of lowest limit to highest limit


        # If the plot limits ratio of plot 1 is smaller than plot 2, the first data set has
        # a wider range range than the second data set. Calculate a new low limit for the
        # second data set to obtain a similar ratio to the first data set.
        # Else, do it the other way around

        if ax1_yratio < ax2_yratio: 
            axRight.set_ylim(bottom = ax2_ylims[1]*ax1_yratio)
        else:
            axSequence.set_ylim(bottom = ax1_ylims[1]*ax2_yratio)


    if textAnnotations and textProperiesAtMaxStrain:
        scatter=axSequence.scatter(
            Time[startPos[:-1]],
            [a+b for a,b in zip(data["Strain_axial_atMaxStrain"],data["Strain_axial_offset"]) ],
            s=50,
            c="r",
            label="at max strain"
        )

        for x,y,s in zip(     
            Time[startPos[:-1]],
            [a+b for a,b in zip(data["Strain_axial_atMaxStrain"],data["Strain_axial_offset"]) ],
            data["Modulus_axial_atMaxStrain"]
            ):

            axSequence.text(x-1.5,y*1.2,r"$E_l$={:4.2f} GPa".format(s/1000),c="r")

        for x,y,s in zip(     
            Time[startPos[:-1]],
            [a+b for a,b in zip(data["Strain_trans_max"],data["Strain_trans_offset"]) ],
            data["Poisson_axial_atFullLoad"]
            ):

            axSequence.text(x-1.5,y*1.2,r"$\nu$={:4.2f}".format(s),c="C1")

        lines.append(scatter.legend_elements(prop="sizes", alpha=0.6)[0])

        lines[-1][0]._color="r" #HACK

    # manually create legend (from two twin axes)

    lines=[l[0] for l in lines]

    labels = [l.get_label() for l in lines]

    if textAnnotations:
        if plotStress:
            labels[-2]=scatter.get_label()
        else:
            labels[-1]=scatter.get_label()


    leg=plt.legend(lines, labels, loc=0)

    # get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(4)

    axSequence.set_xlabel("Time (min)")
    axSequence.xaxis.set_minor_locator(AutoMinorLocator())

    axSequence.set_ylabel("Strain (m/m)")

    if plotStress:
        axRight.set_ylabel("Stress (MPa)")

    # plt.grid()

    # plt.figure(figsize=[12,8],num="stacked curves_{}".format(titleStr))

    for index,label in enumerate(data["StressLabel"]):
        loadingLevel=round(label)
        axNorm.plot(
            data["Time"][index]/60-data["Time"][index][0]/60,
            data["Strain_axial_normalized"][index],
            label="{:4.1f} MPa".format(label) if loadingLevel not in excludeLoading else "{:4.0f} MPa (excluded)".format(label),
            lineStyle="-" if loadingLevel not in excludeLoading else ":"
        )

    axNorm.set_xlabel("Time (min)")
    axNorm.xaxis.set_minor_locator(AutoMinorLocator())
    axNorm.set_ylabel("Strain, normalized (a.u.)")

    plt.figtext(0.015,0.95,"a)",fontsize=28)
    plt.figtext(0.595,0.95,"b)",fontsize=28)

    leg = axNorm.legend()
    # get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(4)

def readAmitexResults(AmitexResultsFile):

    with open(AmitexResultsFile,"r") as f:
        lines=f.readlines()
        
        fields=[ff.replace("\n","").replace(" ","") for ff in lines[0].split(", ")]

        dataAmitex={}
        LUT={}

        for ii,field in enumerate(fields):
            dataAmitex[field]=[]
            LUT[ii]=field

        for line in lines[1:]:
            values=[float(vv.replace("\n","")) for vv in line.split(", ")]

            for ii,vv in enumerate(values):
                dataAmitex[LUT[ii]].append(vv)

        for key,values in dataAmitex.items():
            dataAmitex[key]=np.array(values)

    return dataAmitex

def readAmitexResultsRelaxation(AmitexResultsFile):

    with open(AmitexResultsFile,"r") as f:
        lines=f.readlines()
        
        dataAmitex={
            "Time" :[],
            "C_hom":[],
            "alpha":[],
            "beta" :[]
        }

        for line in lines:
            values=[float(s) for s in [vv.replace("\n","") for vv in line.split(", ")] if s]

            j=-1

            dataAmitex["Time" ].append(values[0])
            dataAmitex["C_hom"].append(np.zeros((6,6)))
            dataAmitex["alpha"].append(values[37])
            dataAmitex["beta" ].append(values[38])


            for i in range(36):
                ii=i%6
                if ii ==0:
                    j+=1

                dataAmitex["C_hom"][-1][ii,j]=values[i+1]       

    return dataAmitex

def find(dataPath,searchStr,verbose=False,returnFilenames=False):

    pathList=[]

    for dirPath, dirNames, filenames in os.walk(dataPath):
        if verbose:
            print("\ndirPath:\t{},\ndirNames:\t{},\nfilenames:\t{}\n".format(dirPath, dirNames, filenames))

        for filename in filenames:
            if searchStr in filename:
                if returnFilenames:
                    pathList.append(os.path.join(dirPath,filename))
                else:
                    pathList.append(dirPath)

    return pathList
