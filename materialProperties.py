

import subprocess
import numpy as np

cmd = "hostname"
# returns output as byte string
hostnameStr = subprocess.check_output(cmd).decode("utf-8").split("\n")[0]
# using decode() function to convert byte string to string
print('Current hostname is :', hostnameStr)

# nIterMaxAmitex='4000'
nIterMaxAmitex='8000' # for mesoscale homogenization

if hostnameStr == "atlas.meca.polymtl.ca":
    mat_lib_path="/home/monet/fasos/Git_repos/Amitex_UMATs/libUmatAmitex.so"
    max_vtk_size=1000
    openMP_threads=32
    showPlotsViscoelasticity=False

else:
    raise EnvironmentError("The hostname: {} is not recognized, unknown path do data".format(hostnameStr))

dict_material_properties_all={}

dict_material_properties_all["UMAT_Vx_neat_dualTemp_G4_G6_only_lowTemp"]={

    'matrix': {
        'behavior': 'UMAT_viscoelastic',
        'units':{'stress':'MPa','time':'minutes'},
        'subroutine': 'UMAT_Vx_neat_dualTemp_G4_G6_only',
        'kappa_0' :5314.1345,
        'mu_0'    :1103.5085,
    },

    'fiber': {
        'behavior': 'trans_iso',
        "units":"MPa",
        'young_l'  : 450e3, #units MPa
        'young_t'  : 10e3,  #units MPa
        'poisson_l': 0.28,
        'poisson_t': 0.4,
        'shear_l'  : 7e3,   #units MPa
        'axis'     : 2
    },

    "mat_lib_path":mat_lib_path,

    # "loadingSpecs":{
    #     "loadingType":"Creep recovery",
    #     "units":{"stress":"MPa","time":"minutes"},
    #     # "StressMax" :9.6807, #compare with G4
    #     # "StressMax" :19.3195, #compare with G4
    #     # "Temperature":21,
    #     # "StressMax" :9.4894, #compare with G1
    #     # "StressMax" :17.0330, #compare with G1

    #     "Temperature":21,   
    #     "t1"        :0.01,
    #     "t2"        :8,
    #     "t3"        :8.01,
    #     "tEnd"      :16,
    #     "nIter_t1"  :30,
    #     "nIter_t2"  :60,
    #     "nIter_t3"  :30,
    #     "nIter_tEnd":40,
    #     "direction"  :"zz"
    # }

    "loadingSpecs":{
        "loadingType":"Relaxation",
        "Temperature":21,
        'units':{'stress':'MPa','time':'minutes'},
        "StrainVal":1.,
        "t1":1,     #loadApplication
        "t2":44,    #relaxation time
        "t3":1000000,   #recovery
        "nIter_t1":1,
        "nIter_t2":44,
        "nIter_t3":3,
        "porosityThreshold":0.01
    }

    # "loadingSpecs":{
    #     "loadingType":"StressTimeSeries",
    #     'units':{'stress':'MPa','time':'minutes'},
    #     "Temperature":21,
    #     "TimeStressPairs":[
    #         (  0.0100	,  4.8422),
    #         (  7.9933	,  4.8422),
    #         (  8.0267	,  0.0001),
    #         ( 40.0500	,  0.0001),
    #         ( 40.0600	,  9.6807),
    #         ( 48.0517	,  9.6807),
    #         ( 48.1017	,  0.0001),
    #         ( 80.1417	,  0.0001),
    #         ( 80.1517	, 14.5137),
    #         ( 88.1350	, 14.5137),
    #         ( 88.1933	,  0.0001),
    #         (120.2333	,  0.0001),
    #         (120.2433	, 19.3195),
    #         (128.2350	, 19.3195),
    #         (128.3183	,  0.0001),
    #         (168.3183	,  0.0001),
    #     ],
    #     "direction"  :"zz",
    #     "timeIntervals":1/6,
    # }

}

dict_material_properties_all["UMAT_Vx_neat_dualTemp_G4_G6_only_highTemp"]={

    'matrix': {
        'behavior': 'UMAT_viscoelastic',
        'units':{'stress':'MPa','time':'minutes'},
        'subroutine': 'UMAT_Vx_neat_dualTemp_G4_G6_only',
        'kappa_0' :5314.1345,
        'mu_0'    :1103.5085,
    },


    'fiber': {
        'behavior': 'trans_iso',
        "units":"MPa",
        'young_l'  : 450e3, #units MPa
        'young_t'  : 10e3,  #units MPa
        'poisson_l': 0.28,
        'poisson_t': 0.4,
        'shear_l'  : 7e3,   #units MPa
        'axis'     : 2
    },

    "mat_lib_path":mat_lib_path,

    # "loadingSpecs":{
    #     "loadingType":"Creep recovery",
    #     "units":{"stress":"MPa","time":"minutes"},
    #     # "StressMax" :9.6807, #compare with G4
    #     # "StressMax" :19.3195, #compare with G4
    #     # "Temperature":21,
    #     # "StressMax" :9.4894, #compare with G1
    #     # "StressMax" :17.0330, #compare with G1

    #     "StressMax" :19.72173, #compare with F4 HIGH temp
    #     "Temperature":120,

    #     "t1"        :0.01,
    #     "t2"        :20,
    #     "t3"        :20.01,
    #     "tEnd"      :32,
    #     "nIter_t1"  :1,
    #     "nIter_t2"  :40,
    #     "nIter_t3"  :3,
    #     "nIter_tEnd":5,
    #     "direction"  :"zz"
    # }

    "loadingSpecs":{
        "loadingType":"Relaxation",
        "Temperature":120,
        'units':{'stress':'MPa','time':'minutes'},
        "StrainVal":1.,
        "t1":1,     #loadApplication
        "t2":44,    #relaxation time
        "t3":1000000,   #recovery
        "nIter_t1":1,
        "nIter_t2":44,
        "nIter_t3":3,
        "porosityThreshold":0.01
    }
}

dict_material_properties_all["UMAT_Vx_neat_dualTemp_G4_G6_only_noPorosity_lowTemp"]={

    'matrix': {
        'behavior': 'UMAT_viscoelastic',
        'units':{'stress':'MPa','time':'minutes'},
        'subroutine': 'UMAT_Vx_neat_dualTemp_G4_G6_only_noPorosity',
        'kappa_0' :5314.1345, #only an estimate, for reference material
        'mu_0'    :1103.5085, #only an estimate, for reference material
    },

    'fiber': {
        'behavior': 'trans_iso',
        "units":"MPa",
        'young_l'  : 450e3, #units MPa
        'young_t'  : 10e3,  #units MPa
        'poisson_l': 0.28,
        'poisson_t': 0.4,
        'shear_l'  : 7e3,   #units MPa
        'axis'     : 2
    },

    "mat_lib_path":mat_lib_path,

    "loadingSpecs":{
        "loadingType":"Relaxation",
        "Temperature":21,
        'units':{'stress':'MPa','time':'minutes'},
        "StrainVal":1.,
        "t1":1,     #loadApplication
        "t2":44,    #relaxation time
        "t3":1000000,   #recovery
        "nIter_t1":1,
        "nIter_t2":44,
        "nIter_t3":3,
        "porosityThreshold":0.01
    }
}

dict_material_properties_all["UMAT_Vx_neat_dualTemp_G4_G6_only_noPorosity_highTemp"]={

    'matrix': {
        'behavior': 'UMAT_viscoelastic',
        'units':{'stress':'MPa','time':'minutes'},
        'subroutine': 'UMAT_Vx_neat_dualTemp_G4_G6_only_noPorosity',
        'kappa_0' :5314.1345, #only an estimate, for reference material
        'mu_0'    :1103.5085, #only an estimate, for reference material
    },

    'fiber': {
        'behavior': 'trans_iso',
        "units":"MPa",
        'young_l'  : 450e3, #units MPa
        'young_t'  : 10e3,  #units MPa
        'poisson_l': 0.28,
        'poisson_t': 0.4,
        'shear_l'  : 7e3,   #units MPa
        'axis'     : 2
    },

    "mat_lib_path":mat_lib_path,

    "loadingSpecs":{
        "loadingType":"Relaxation",
        "Temperature":120,
        'units':{'stress':'MPa','time':'minutes'},
        "StrainVal":1.,
        "t1":1,     #loadApplication
        "t2":44,    #relaxation time
        "t3":1000000,   #recovery
        "nIter_t1":1,
        "nIter_t2":44,
        "nIter_t3":3,
        "porosityThreshold":0.01
    }
}

