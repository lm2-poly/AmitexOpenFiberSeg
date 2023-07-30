import os
from tracemalloc import start
from matplotlib.cbook import flatten
import numpy as np
from matplotlib import pyplot as plt
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
    excludeLoading=[]
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
    factorLengthOfLoadingAfterRelease=3


    for i,s in enumerate(startPos[:-1]):
        if s<endOfLoadPos[i]<dropPos[i]<startPos[i+1]:
            intervalLoadApplied=dropPos[i]-s
            endOfRecovery.append(s+intervalLoadApplied*factorLengthOfLoadingAfterRelease)
        else:
            raise ValueError("square wave mis-identified")


    for iStart,iEndLoad,iDrop,iEnd in zip(startPos[:-1],endOfLoadPos,dropPos,endOfRecovery):
        data["Time"]        .append(np.array(Time        [max(iStart-includeTimeIntervalBeforeAppliedLoad,0):iEnd-includeTimeIntervalBeforeAppliedLoad]))
        data["t1"].append(Time[iStart])
        data["t2"].append(Time[iEndLoad])
        data["t3"].append(Time[iDrop+timeOffsetAfterLoadDrop])
        
        timeOffset_zeroThisSeries=data["Time"][-1][0]

        data["Time"][-1]-=timeOffset_zeroThisSeries#bring start time to 0 for each loading step
        data["t1"]  [-1]-=timeOffset_zeroThisSeries
        data["t2"]  [-1]-=timeOffset_zeroThisSeries
        data["t3"]  [-1]-=timeOffset_zeroThisSeries

        data["Force"]       .append(np.array(Force[max(iStart-includeTimeIntervalBeforeAppliedLoad,0):iEnd-includeTimeIntervalBeforeAppliedLoad]))
        data["Stress"]      .append(data["Force"][-1] /areaCrossSection)
        maxStressVal=np.max(data["Stress"][index])

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

            filenameOutput="CreepData-{:3.0f}MPa.xlsx".format(data["StressLabel"][index])

            # Create a Pandas Excel writer using XlsxWriter as the engine.
            writer = ExcelWriter(os.path.join(exportPath,filenameOutput), engine='xlsxwriter')

            # Convert the dataframe to an XlsxWriter Excel object.
            myDataFrame.to_excel(writer, sheet_name='Sheet1')

            # Close the Pandas Excel writer and output the Excel file.
            writer.save()

            E=data["Modulus_axial_atFullLoad"][index]
            nu=0.37#data["Poisson_axial_atFullLoad"][index]

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

            filenameOutput="CreepData-{:3.0f}MPa_HeavisideTimes.xlsx".format(data["StressLabel"][index])

            # Create a Pandas Excel writer using XlsxWriter as the engine.
            writer = ExcelWriter(os.path.join(exportPath,filenameOutput), engine='xlsxwriter')

            # Convert the dataframe to an XlsxWriter Excel object.
            myDataFrame.to_excel(writer, sheet_name='Sheet1')

            # Close the Pandas Excel writer and output the Excel file.
            writer.save()

        index+=1

    print("\n Force(N) \tStress (MPa),\tE_l (first loading) (GPa),\tE_l(t) (max strain) (GPa),\tPoisson's ratio")
    for F,s,E,Ecreep,v in zip(
        [Force[p] for p in startPos[:-1]],
        Stress,
        data["Modulus_axial_atFullLoad"],
        data["Modulus_axial_atMaxStrain"],
        data["Poisson_axial_atFullLoad"]):

        print("{: >8.2f},\t{: >8.2f},\t\t{: >4.2f},\t\t\t\t{: >4.2f},\t\t\t\t{:>4.2f}".format(F,s,E/1000,Ecreep/1000,v))

    plt.figure(figsize=[12,8],num="CreepTest_rawData_{}".format(titleStr))

    Time=np.array(Time)/60

    lines=[
        plt.plot(Time,np.array(Strain_axial),     label="Strain_axial"),
        plt.plot(Time,np.array(Strain_trans),     label="Strain_trans"),
    ]

    if plotTempCorrection:
        lines.extend(
            [
                plt.plot(Time,T_correction,label="T_correction"),
                plt.plot(Time,np.array(Strain_axial)+baseLineVerticalOffset_axial-T_correction,label="Strain_axial_corrected")
            ]
        )

    axLeft=plt.gca()

    scatter=axLeft.scatter(
        Time[startPos[:-1]],
        [a+b for a,b in zip(data["Strain_axial_atFullLoad"],data["Strain_axial_offset"]) ],
        s=50,
        c="r",
        label="at first loading"
    )

    lines.append(scatter.legend_elements(prop="sizes", alpha=0.6)[0])

    lines[-1][0]._color="r" #HACK


    for x,y,s in zip(     
        Time[startPos[:-1]],
        [a+b for a,b in zip(data["Strain_axial_atFullLoad"],data["Strain_axial_offset"]) ],
        data["Modulus_axial_atFullLoad"]
        ):

        axLeft.text(x-0.2,y*1.1,r"$E_l$={:4.2f} GPa".format(s/1000),c="r")

    for x,y,s in zip(     
        Time[startPos[:-1]],
        [a+b for a,b in zip(data["Strain_trans_atFullLoad"],data["Strain_trans_offset"]) ],
        data["Poisson_axial_atFullLoad"]
        ):

        axLeft.text(x-0.2,y*1.2,r"$\nu$={:4.2f}".format(s),c="C2")

    if plotStress:
        axRight = axLeft.twinx()

        Stress=np.array(Force)/areaCrossSection
        lines.append(axRight.plot(Time,Stress,"--",color="C2",label="Stress"))

        ax1_ylims = axLeft.axes.get_ylim()           # Find y-axis limits set by the plotter
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
            axLeft.set_ylim(bottom = ax1_ylims[1]*ax2_yratio)


    if textProperiesAtMaxStrain:
        scatter=axLeft.scatter(
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

            axLeft.text(x-1.5,y*1.2,r"$E_l$={:4.2f} GPa".format(s/1000),c="r")

        for x,y,s in zip(     
            Time[startPos[:-1]],
            [a+b for a,b in zip(data["Strain_trans_max"],data["Strain_trans_offset"]) ],
            data["Poisson_axial_atFullLoad"]
            ):

            axLeft.text(x-1.5,y*1.2,r"$\nu$={:4.2f}".format(s),c="C1")

        lines.append(scatter.legend_elements(prop="sizes", alpha=0.6)[0])

        lines[-1][0]._color="r" #HACK

    # manually create legend (from two twin axes)

    lines=[l[0] for l in lines]

    labels = [l.get_label() for l in lines]

    if plotStress:
        labels[-2]=scatter.get_label()
    else:
        labels[-1]=scatter.get_label()


    leg=plt.legend(lines, labels, loc=0)

    # get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(4)


    axLeft.set_xlabel("Time (min)")
    axLeft.set_ylabel("Strain")
    if plotStress:
        axRight.set_ylabel("Stress (MPa)")


    plt.grid()

    plt.figure(figsize=[12,8],num="stacked curves_{}".format(titleStr))

    for index,label in enumerate(data["StressLabel"]):
        loadingLevel=round(label)
        plt.plot(
            data["Time"][index]/60-data["Time"][index][0]/60,
            data["Strain_axial_normalized"][index],
            label="{:4.0f} MPa".format(label) if loadingLevel not in excludeLoading else "{:4.0f} MPa (excluded)".format(label),
            lineStyle="-" if loadingLevel not in excludeLoading else ":"
        )

    plt.xlabel("Time min)")
    plt.ylabel("Strain, normalized")


    leg = plt.legend()
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

def isotropic_projector_Facu(M):
    _alpha,_beta,E,nu=extract_isotropic_parameters( M )
    J_tensor4 = generate_J_tensor4()
    J_matrix4 = tensor4_to_voigt4( J_tensor4 )
    
    K_tensor4 = generate_K_tensor4()
    K_matrix4 = tensor4_to_voigt4( K_tensor4 )

    isotropic_M=dot(_alpha,J_matrix4)+dot(_beta,K_matrix4)

    return isotropic_M, _alpha,_beta, E, nu

def modifyConvention(C):
    """Amitex and Abaqus output tensors are in notation 11 22 33 12 13 23, 
    this converts into notation 11 22 33 23 13 12"""

    temp=C.copy()
    temp[5]=C[3]
    temp[3]=C[5]

    C_mod=temp.copy()

    for i in range(6):
        C_mod[i][5]=temp[i][3]
        C_mod[i][3]=temp[i][5]       

    return C_mod

def write_to_file_Voigt4(
    filename,
    tensor4,
    prec_total=8,
    prec_decimal=4,
    material=None,
    material_tag=None,
    modifyConventionBool=True,
    loadingType=False,
    time=None,
    units=None
    ):

    """tensor4 is a list[6][6] (or array[numTimeSteps,6,6] for viscoelasticity), 
    prec_total is the total number of digits, prec_decimal is the number of decimal places"""
    prec="{{: >{}.{}f}}, ".format(prec_total,prec_decimal)
    precString="["+prec*6+"],\n"
    
    if loadingType == "relaxation":
        
        with open(filename,'w') as f:

            outputString=prec*39+"\n"
            for iTimeStep in range(len(tensor4)):
                
                print("\n timeStep:{}, time: {: >12.4f}".format(iTimeStep,time[iTimeStep]))

                # printVoigt4(tensor4[iTimeStep])

                C_iso,alphaIso,betaIso,E,nu = isotropic_projector_Facu(modifyConvention(tensor4[iTimeStep]))
                
                outputStringTerminal="\nalpha\t= \t"+prec+"\nbeta\t= \t"+prec+"\nE\t\t= \t"+prec+"\nnu\t\t= \t"+prec+"\n"
                print(outputStringTerminal.format(alphaIso,betaIso,E,nu))

                f.write(outputString.format(time[iTimeStep],*(ravel(tensor4[iTimeStep])),alphaIso/3,betaIso/2))

                if iTimeStep==0:
                    E_l,E_t,nu_l,nu_t,G_l=transverse_isotropic_projector(
                        modifyConvention(tensor4[iTimeStep]),
                        axis=2
                    )[-5:]
            
            return (E_l,E_t,nu_l,nu_t,G_l)


    if loadingType in ["creep","StressTimeSeries"]:
        with open(filename,'w') as f:

            headerFormater="{{: <{}}}, ".format(prec_total)
            headerString=(headerFormater*13)[:-2]+"\n"

            unitsTime="m" if units["time"]=="minutes" else "s"
            unitsStress="MPa" if units["stress"]=="MPa" else "GPa"
            
            f.write(headerString.format(
                "Time ({})".format(unitsTime),
                "Stress_xx ({})".format(unitsStress),
                "Stress_yy ({})".format(unitsStress),
                "Stress_zz ({})".format(unitsStress),
                "Stress_xy ({})".format(unitsStress),
                "Stress_xz ({})".format(unitsStress),
                "Stress_yz ({})".format(unitsStress),
                
                "Strain_xx".format(unitsStress),
                "Strain_yy".format(unitsStress),
                "Strain_zz".format(unitsStress),
                "Strain_xy".format(unitsStress),
                "Strain_xz".format(unitsStress),
                "Strain_yz".format(unitsStress),
                )
            )

            outputString=(prec*13)[:-2]+"\n"
            for iTimeStep in range(len(tensor4[0])):
                stress=tensor4[0][iTimeStep]
                strain=tensor4[1][iTimeStep]
                                
                f.write(outputString.format(time[iTimeStep],*stress,*strain))
            
            return (False,False,False,False,False)

    elif loadingType == "elasticity":
        if modifyConventionBool:
            C=modifyConvention(tensor4) #Amitex suit la convention 11 22 33 12 13 23, remettre en 11 22 33 23 13 12
        else: 
            C=tensor4

        with open(filename,'w') as f:
            f.write("Resulting homogenized stiffness tensor\n\n")

            for i in range(6):
                f.write(precString.format(*tensor4[i]))
            if modifyConventionBool:
                f.write('\n/!\ This tensor is in notation 11 22 33 12 13 23 /!\ ')
            else:
                f.write('\n Usual convention')

            f.write("\n\nIsotropic parameters: \n")
            C_iso,alphaIso,betaIso,E,nu = isotropic_projector_Facu(C)
            outputString="\nalpha\t= \t"+prec+"\nbeta\t= \t"+prec+"\nE\t\t= \t"+prec+"\nnu\t\t= \t"+prec+"\n"
            f.write(outputString.format(alphaIso,betaIso,E,nu))
            f.write("\nthe following analysis is considering an alignment of fibers alows z axis (axis=2)\n\n")
            axis=2

            alpha, beta, gamma, gamma_prime, delta, delta_prime=extract_trans_iso_parameters_from_S(C ,axis)		

            f.write("alpha\t\t={: >10.3f}\nbeta\t\t={: >10.3f}\ngamma\t\t={: >10.3f}\ndelta\t\t={: >10.3f}\ndelta_prime\t={: >10.3f}\nalpha*beta-gamma^2\t={: >10.3f}\n".\
                format(alpha,beta,gamma,delta,delta_prime,alpha*beta-gamma**2))
            f.write('\n')
            E_l,E_t,nu_l,nu_t,G_l=transverse_isotropic_projector(C,axis)[-5:]
            f.write(
                "E_l\t\t={: >10.4f}\nE_t\t\t={: >10.4f}\nnu_l\t={: >10.4f}\nnu_t\t={: >10.4f}\nG_l\t\t={: >10.6f}\n".\
                format(E_l,E_t,nu_l,nu_t,G_l)
            )

            C_trans_iso=generate_trans_isoC_from_E_nu_G(E_l,E_t,nu_l,nu_t,G_l,axis)

            f.write("\nOriginal tensor in 11 22 33 23 13 12 notation: \n")
            for iRows in range(6):
                f.write(precString.format(*C[iRows]))

            f.write("\nTransverse isotropic projection: \n")
            for iRows in range(6):
                f.write(precString.format(*C_trans_iso[iRows]))

            distance_trans=matrix_distance(C,C_trans_iso)
            f.write("\nDistance (Frobenius norm), to transverse isotropic space:\n{: >10.3f}\n".\
                format(distance_trans))

            f.write("\nIsotropic projection: \n")
            for iRows in range(6):
                f.write(precString.format(*C_iso[iRows]))

            distance_iso=matrix_distance(C,C_iso)
            f.write("\nDistance (Frobenius norm), to isotropic space:\n{: >10.3f}\n".format(distance_iso))

            if material is not None:
                f.write("\n ############################################################")
                f.write("\n\n\tInput data\n")

                f.write("\nMaterial tag: {:>15}\n".format(material_tag))

                ### matrix

                f.write("\nMatrix:\n")

                if material["matrix"]["behavior"]=="iso":
                    for key,val in material["matrix"].items():
                        f.write("\n{:<10}:\t{:>6}".format(key,val))
                elif material["matrix"]["behavior"]=="orthotropic":
                    for iRows in range(6):
                        f.write(precString.format(*material["matrix"]["C"][iRows]))
                
                elif material["matrix"]["behavior"]=="viscoelas_maxwell":
                    f.write("\nviscoelastic properties:")
                    f.write("\nkappa_0:\t{:>6}".format(material["matrix"]["kappa_0"]))
                    f.write("\nmu_0   :\t{:>6}".format(material["matrix"]["mu_0"]))
                    f.write("\n chains: (kappa_i,nu_i,tau_i):")
                    
                    for chain in material["matrix"]["chains"]:
                        f.write("\n\t{:>6}\t{:>6}\t{:>6}".format(*chain))


                f.write("\n\nFiber:")
                if material["fiber"]["behavior"] in ["iso","trans_iso"]:
                    for key,val in material["fiber"].items():
                        f.write("\n{:<10}:\t{:>6}".format(key,val))
                elif material["fiber"]["behavior"]=="none":
                    f.write("\nNo fibers present")

        return (E_l,E_t,nu_l,nu_t,G_l)