# -*- coding: utf-8 -*-

from cfrph_interface import CraftInterface as CrI
from cfrph_interface import AmitexInterface as AmI
from cfrph_interface import MoriTanakaInterface as MT_I
from cfrph_utils import getCommonPaths, secondsToText,hashDict
import os
import time
import json
from socket import gethostname

from materialProperties import dict_material_properties_all,hostnameStr,openMP_threads
from tensor_personal_functions import modifyConvention, printVoigt4

# microstructure_directory="sampleF1/processed_x56-1436_y80-1422_z1-970/2022-06-27_13h16m01/AmitexFiles/downSampled/"

# material_tagList=[
#     "UMAT_Vx_neat_dualTemp_G4_G6_only_lowTemp",
#     "UMAT_Vx_neat_dualTemp_G4_G6_only_highTemp"
# ]
# origin=[0,0,0]
# OpenFiberSeg=True
# convergence_acceleration=True
# exclusiveRVEstr=None

## sampleF1
# microstructure_directory="sampleF1/processed_x56-1436_y80-1422_z1-970/2022-06-27_13h16m01/AmitexFiles/downSampled/"

# material_tagList=[
#     "UMAT_Vx_neat_dualTemp_G4_G6_only_noPorosity_lowTemp",
#     "UMAT_Vx_neat_dualTemp_G4_G6_only_noPorosity_highTemp",
#     ]
# origin=[0,0,0]
# OpenFiberSeg=True
# convergence_acceleration=True

## sampleB1

# microstructure_directory="sampleB1/processed_x1-2009_y1-1722_z1-971/2022-06-15_09h31m26/AmitexFiles/downSampled/"

# material_tagList=[
#     "UMAT_Vx_neat_dualTemp_G4_G6_only_noPorosity_lowTemp",
#     "UMAT_Vx_neat_dualTemp_G4_G6_only_noPorosity_highTemp",
#     ]
# origin=[0,0,0]
# OpenFiberSeg=True
# convergence_acceleration=True

# sampleF1 meso
# meso structure homogenization for sampleF1, umat from microscale homogenization

# microstructure_directory="sampleF1/sampleF1_2022-06-22_185257_meso_large/processed_rotated/AmitexFiles/downSampled/"
# material_tagList=[
#     "UMAT_microStruct_rightMatrix_Miyagawa_lowTemp",
#     "UMAT_microStruct_rightMatrix_Miyagawa_highTemp",
# ]
# origin=[0,0,0]
# OpenFiberSeg=True
# convergence_acceleration=True


for material_tag in material_tagList:

    dict_material_properties=dict_material_properties_all[material_tag]
    resolution_vtk_list = [
        "all",
        # {"x":50,"y":50,"z":17}, # for debugging small subvolume
        ]

    if "loadingSpecs" in dict_material_properties:
        if "porosityThreshold" in dict_material_properties["loadingSpecs"]:
            porosityThreshold=dict_material_properties["loadingSpecs"]["porosityThreshold"]
        else:
            porosityThreshold=None
    else:
        porosityThreshold=None

    if useInterface:
        dict_material_properties["matrix"]["interfaceVal"]=interfaceVal

    print('\nMain microstructure directory:\n\n {}\n'.format(microstructure_directory))

    run_amitex_bool     = True # Run Amitex on newly processed files
    run_MoriTanaka_bool = False
    overwrite           = False
    makeVTK_stress_strain=True # Applies to Elasticity: makes detailled output of Amitex, not only averaged values 

    if overwrite:
        input("\n\n\t\tWarning, overwrite is active, press 'enter' to continue \n\n")

    # Fibres symmetry
    symmetry_fibres = dict_material_properties['fiber']['behavior']

    directories=getCommonPaths(
        microstructure_directory,
        resolution_vtk_list,
        origin,
        dict_material_properties,
        OpenFiberSeg=OpenFiberSeg,
        exclusiveRVEstr=exclusiveRVEstr
        )


    resolution_vtk_list=list(directories.keys())

    print('List of directories identified for processing:\n')
    for resolution,paths in directories.items():
        print("resolutions: {}".format(resolution))
        print("\tpaths:")
        for p in paths:
            print("\t\t{}".format(p))
            # HACK to avoid following error when printing many lines:
            # "BlockingIOError: [Errno 11] write could not complete without blocking"
            time.sleep(0.001) 
    print('\n')

    time_stamp_string = time.strftime("%Y-%m-%d_%H:%M:%S")

    for resolution in resolution_vtk_list:
        for current_path in directories[hashDict(resolution)]:

            ##################################################################################################
            ##################################################################################################

            ##### AMITEX database
            
            ##################################################################################################
            ##################################################################################################

            if "loadingSpecs" in dict_material_properties_all[material_tag]:
                loadingType=dict_material_properties_all[material_tag]["loadingSpecs"]["loadingType"].replace(" ","_")
                if porosityThreshold is None:
                    if useInterface:
                        amitex_log_file_name    = '{}log_mat_{}_interfaceVal_({:3.2f},{:3.2f})_res={}_origin_{}_loading={}_amitex.txt'.format(
                            current_path,material_tag,*interfaceVal,resolution,origin,loadingType).replace("\'","")
                    else:
                        amitex_log_file_name    = '{}log_mat_{}_res={}_origin_{}_loading={}_amitex.txt'.format(
                            current_path,material_tag,resolution,origin,loadingType).replace("\'","")
                else:
                    if useInterface:
                        amitex_log_file_name    = '{}log_mat_{}_interfaceVal_({:3.2f},{:3.2f})_res={}_origin_{}_loading={}_poroThresh={:<3.2f}_amitex.txt'.format(
                            current_path,material_tag,*interfaceVal,resolution,origin,loadingType,porosityThreshold).replace("\'","")
                    else:
                        amitex_log_file_name    = '{}log_mat_{}_res={}_origin_{}_loading={}_poroThresh={:<3.2f}_amitex.txt'.format(
                            current_path,material_tag,resolution,origin,loadingType,porosityThreshold).replace("\'","")
            else:
                if useInterface:
                    amitex_log_file_name    = '{}log_mat_{}_interfaceVal_({:3.2f},{:3.2f})_res={}_origin_{}_sym={}_amitex.txt'.format(
                        current_path,material_tag,*interfaceVal,resolution,origin,symmetry_fibres).replace("\'","")
                else:
                    amitex_log_file_name    = '{}log_mat_{}_res={}_origin_{}_sym={}_amitex.txt'.format(
                        current_path,material_tag,resolution,origin,symmetry_fibres).replace("\'","")

            process_now_bool = False
            if run_amitex_bool:
                if overwrite:
                    print("\n\n\t\tWarning, overwrite is active")
                    process_now_bool=True
                else:
                    if porosityThreshold is not None:
                        pathtoFiberVolumeFractions=os.path.join(current_path,"fiberVolumeFractions.json")

                        if not os.path.exists(pathtoFiberVolumeFractions):
                            pathtoFiberVolumeFractions=os.path.join(current_path,"fiberVolumeFractions_adjusted.json")
                            if not os.path.exists(pathtoFiberVolumeFractions):
                                raise IOError("json file to fiberVolumeFractions not in subdirectory")

                        with open(pathtoFiberVolumeFractions,"r") as f:
                            fiberData=json.load(f)

                            if fiberData["meanPoresFrac"]>porosityThreshold:
                                #skip this microstructure
                                print("This microstructure has {:6.3%} porosity, which is above the porosity threshold of {:>6.3%}. Skipping...\n".\
                                    format(fiberData["meanPoresFrac"],porosityThreshold))
                                continue


                    # Check if this microstructure has been processed
                    if os.path.isfile(amitex_log_file_name):
                        print(amitex_log_file_name,'found...\n')

                        with open(amitex_log_file_name, 'r') as f:
                            lines=f.readlines()

                        if len(lines)>1 and 'Amitex executed casename' in lines[1]:
                            print('Already fully processed: {}\n'.format(current_path.split('/')[-1]))
                        #if processing has not been completed, attempt again
                        else:
                            with open(amitex_log_file_name, 'w') as f:
                                lines[0]='Attemping to process...\n'
                                print("Attemping to process...\n")
                                process_now_bool=True               
                                f.writelines(lines) 
                    else:
                        if not os.path.exists(current_path):
                            print("creating directory at: \n{}".format(current_path))
                            os.mkdir(current_path)

                        #first attempt
                        with open(amitex_log_file_name, 'w') as f:
                            f.write('Attemping to process...\n')
                            print('Attemping to process...\n')
                            process_now_bool = True

                if process_now_bool:

                    if OpenFiberSeg:
                        casename="{}_{}".format(current_path.split("/")[-3],current_path.split("/")[-2])
                    else:
                        casename="{}".format(current_path.split("/")[-2])

                    amitex_instance = AmI(
                        current_path, 
                        casename,
                        dict_material_properties,
                        material_tag,
                        resolution,
                        OpenFiberSeg=OpenFiberSeg,
                        origin=origin,
                        convergence_acceleration=convergence_acceleration,
                        openMP_threads=openMP_threads,
                        makeVTK_stress_strain=makeVTK_stress_strain
                    )
                    
                    # Create folders for input and output files relating to this caseMicro
                    case_working_path = amitex_instance.get_workspace_path()

                    if not os.path.isdir(case_working_path):
                        os.mkdir(case_working_path)
                    print('\n\n#####################################\n')
                    print('\nCurrently processing:\n    {}\n'.format(case_working_path.split(microstructure_directory)[-1]))
                    

                    if run_amitex_bool:
                        cpu_time_start=time.time()
                        amitex_instance.preprocessing() 
                        processingCompleted,errorMsg=amitex_instance.processing()

                        if processingCompleted:
                            with open(amitex_log_file_name, 'r+') as f:

                                E_l,E_t,nu_l,nu_t,G_l=amitex_instance.postprocessing()
                                
                                cpu_time_end=time.time()
                                cpu_time_str=secondsToText(cpu_time_end-cpu_time_start)
                                f.write('\nAmitex executed casename: \t{}, \ton: {}, \t CPU time: {}, \t on host machine: {}'\
                                    .format(amitex_instance.get_generic_name(), time_stamp_string, cpu_time_str,gethostname()))
                                
                                f.write("\nCalculated homogeneous properties (with axis being z) :\n\nE_l\t\t={: >10.4f}\nE_t\t\t={: >10.4f}\nnu_l\t\t={: >10.4f}\nnu_t\t\t={: >10.4f}\nG_l\t\t={: >10.4f}\n".\
                                    format(E_l,E_t,nu_l,nu_t,G_l))
                        else:
                            with open(amitex_log_file_name, 'r+') as f:

                                f.write('\nAmitex FAILED at executing casename: \t{}, \ton: {}, \t on host machine: {}'\
                                    .format(amitex_instance.get_generic_name(), time_stamp_string, gethostname()))
                                f.write("\n returned error message: {}".format(errorMsg))
                                print()

                    if processingCompleted:
                        with open(amitex_log_file_name, 'r') as f:  
                            lines=f.readlines()  
                            #replace first line in text file: change lines[0]
                            if dict_material_properties['fiber']['behavior']=="iso":
                                lines[0]=\
                                "processing complete for this microstructure, with material properties= [E_m={: >8.4f} ,nu_m={: >8.4f}, E_f={: >8.4f}, nu_f={: >8.4f}] \n".\
                                format(
                                    amitex_instance.get_E_m(), 
                                    amitex_instance.get_nu_m(),
                                    amitex_instance.get_E_f(),
                                    amitex_instance.get_nu_f()
                                )
                            elif dict_material_properties['fiber']['behavior']=="trans_iso":
                                if dict_material_properties['matrix']['behavior']=="iso":
                                    lines[0]=\
                                    "processing complete for this microstructure, with material properties= [E_m={: >8.4f} ,nu_m={: >8.4f}, E_t={: >8.4f}, E_l={: >8.4f},nu_t={: >8.4f}, nu_l={: >8.4f}, G_l={: >8.4f}] \n".\
                                    format(
                                        amitex_instance.get_E_m(), 
                                        amitex_instance.get_nu_m(), 
                                        amitex_instance.get_E_t(), 
                                        amitex_instance.get_E_l(), 
                                        amitex_instance.get_nu_t(),
                                        amitex_instance.get_nu_l(),
                                        amitex_instance.get_G_l()
                                        )

                                else:
                                    lines[0]="processing complete for this microstructure"



                        print("Processed microstructure:")

                        print(case_working_path)

                        print(lines[0])

                        print("\nCalculated homogeneous properties (with axis being z) :\n\nE_l\t\t={: >10.4f}\nE_t\t\t={: >10.4f}\nnu_l\t\t={: >10.4f}\nnu_t\t\t={: >10.4f}\nG_l\t\t={: >10.4f}\n".\
                                    format(E_l,E_t,nu_l,nu_t,G_l))

                        if OpenFiberSeg:
                            print("Calculated fiber volume fraction:       \t{: >8.3%}".format(amitex_instance.fiberVolumeFraction      ))
                            print("Calculated fiber/matrix volume fraction:\t{: >8.3%}".format(amitex_instance.fiberMatrixVolumeFraction))
                            print("Calculated pores volume fraction:       \t{: >8.3%}".format(amitex_instance.poresVolumeFraction      ))

                        print("Homogenized stiffness tensor:\n")
                        if amitex_instance.loadingType =="elasticity":
                            printVoigt4(modifyConvention(amitex_instance.C_hom))
                        elif amitex_instance.loadingType =="relaxation":
                            print("(initial time step only, Abaqus convention)")
                            printVoigt4(modifyConvention(amitex_instance.C_hom[0]))


            MT_log_file_name    = '{}log_mat_{}_res={}_origin_{}_sym={}_MT.txt'.format(
                current_path,material_tag,resolution,origin,symmetry_fibres).replace("\'","")
            process_now_bool = False
            if run_MoriTanaka_bool:
                # Check if this microstructure has been processed
                if os.path.isfile(MT_log_file_name):
                    print(MT_log_file_name,'found...\n')

                    with open(MT_log_file_name, 'r') as f:
                        lines=f.readlines()

                    if "Attemping to process..." in lines[0]:
                        #if processing has not been completed, attempt again
                        process_now_bool=True               
                    else:
                        print('Already fully processed: {}\n'.format(current_path.split('/')[-1]))
                else:
                    if not os.path.exists(current_path):
                        print("creating directory at: \n{}".format(current_path))
                        os.mkdir(current_path)

                    #first attempt
                    with open(MT_log_file_name, 'w') as f:
                        f.write('Attemping to process...\n')
                        print('Attemping to process...\n')
                        process_now_bool = True

                if process_now_bool:

                    if OpenFiberSeg:
                        casename="{}_{}".format(current_path.split("/")[-3],current_path.split("/")[-2])
                    else:
                        casename="{}".format(current_path.split("/")[-2])

                    MoriTanaka_instance = MT_I(
                        current_path, 
                        casename,
                        dict_material_properties,
                        material_tag,
                        resolution,
                        OpenFiberSeg=OpenFiberSeg,
                        origin=origin,
                        convergence_acceleration=convergence_acceleration,
                        openMP_threads=openMP_threads)


                    cpu_time_start=time.time()

                    MoriTanaka_instance.preprocessing()
                    MoriTanaka_instance.processing()

                    with open(MT_log_file_name, 'r+') as f:

                        E_l,E_t,nu_l,nu_t,G_l=MoriTanaka_instance.postprocessing()
                        
                        cpu_time_end=time.time()
                        cpu_time_str=secondsToText(cpu_time_end-cpu_time_start)

                        f.write('\nMoriTanaka executed casename: \t{}, \ton: {}, \t CPU time: {}, \t on host machine: {}'\
                            .format(MoriTanaka_instance.get_generic_name(), time_stamp_string, cpu_time_str,gethostname()))
                        
                        f.write("\nCalculated homogeneous properties (with axis being z) :\n\nE_l\t\t={: >10.4f}\nE_t\t\t={: >10.4f}\nnu_l\t\t={: >10.4f}\nnu_t\t\t={: >10.4f}\nG_l\t\t={: >10.4f}\n".\
                            format(E_l,E_t,nu_l,nu_t,G_l))

                    print("Processed microstructure:")

                    print(case_working_path)

                    print("material_tag={}".format(material_tag))

                    if dict_material_properties['fiber']['behavior']=="iso":
                        print(
                        "processing complete for this microstructure, with material properties= [E_m={: >8.4f} ,nu_m={: >8.4f}, E_f={: >8.4f}, nu_f={: >8.4f}] \n".\
                            format(
                                MoriTanaka_instance.get_E_m(), 
                                MoriTanaka_instance.get_nu_m(),
                                MoriTanaka_instance.get_E_f(),
                                MoriTanaka_instance.get_nu_f()
                            )
                        )
                    elif dict_material_properties['fiber']['behavior']=="trans_iso":
                        print(
                        "processing complete for this microstructure, with material properties= [E_m={: >8.4f} ,nu_m={: >8.4f}, E_t={: >8.4f}, E_l={: >8.4f},nu_t={: >8.4f}, nu_l={: >8.4f}, G_l={: >8.4f}] \n".\
                        format(
                            MoriTanaka_instance.get_E_m(), 
                            MoriTanaka_instance.get_nu_m(), 
                            MoriTanaka_instance.get_E_t(), 
                            MoriTanaka_instance.get_E_l(), 
                            MoriTanaka_instance.get_nu_t(),
                            MoriTanaka_instance.get_nu_l(),
                            MoriTanaka_instance.get_G_l()
                            )
                        )

                    print("\nCalculated homogeneous properties (with axis being z) :\n\nE_l\t\t={: >10.4f}\nE_t\t\t={: >10.4f}\nnu_l\t\t={: >10.4f}\nnu_t\t\t={: >10.4f}\nG_l\t\t={: >10.4f}\n".\
                                format(E_l,E_t,nu_l,nu_t,G_l))

                    print("Calculated fiber volume fraction:\t{: >8.3%}".format(MoriTanaka_instance.fiberVolumeFraction))
                    if OpenFiberSeg:
                        "pores not implemented in legacy version of microstructures"
                        print("Calculated pores volume fraction:\t{: >8.3%}".format(MoriTanaka_instance.poresVolumeFraction))


                    print("Homogenized stiffness tensor:\n")
                    printVoigt4(MoriTanaka_instance.C_hom)


__author__     = "Facundo Sosa-Rey"
__copyright__  = "all rights reserved"
__credits__    = ["Clément Vella"]
__license__    = "None"
__versionOf__  = "1.0.1"
__maintainer__ = ""
__email__      = ""
__status__     = "Development"

