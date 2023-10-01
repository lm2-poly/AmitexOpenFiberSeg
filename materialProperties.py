

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

dict_material_properties_all["Victrex450G_T300_Miyagawa_nanoIndentation"]={
    'matrix': {
        'behavior':"iso",
        'young'  :  4.000,
        'poisson':  0.3855
    },
    'fiber': {
        'behavior': 'trans_iso',
        'young_l'  : 230,
        'young_t'  : 13.4,
        'poisson_l': 0.256,
        'poisson_t': 0.3,
        'shear_l'  : 27.3,
        'axis'     : 2
    },
    "mat_lib_path":mat_lib_path
}

# The so called anisotropic behavior law is implemented for orthotropic behavior. 
# Only the coefficients at the non-zero positions below are accounted for, 
# and symmetry is enforced: (see file <amitex path>/libAmitex/src/materiaux/elasaniso.f90) 
# !!                          9 coefficients rangés par ligne
# !!                          1 2 3 0 0 0
# !!                          2 4 5 0 0 0
# !!                          3 5 6 0 0 0
# !!                          0 0 0 7 0 0
# !!                          0 0 0 0 8 0
# !!                          0 0 0 0 0 9

dict_material_properties_all["D3_fourth_microStruc_outOfPlanePass_wrongName_vertical_averaged"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [ 14.5228,   4.9617,   4.7737,  -0.0590,   0.0413,   0.7239, ],
                [  4.9618,   8.7931,   4.9618,   0.0211,   0.0000,   0.1517, ],
                [  4.7737,   4.9617,  14.5228,   0.6619,  -0.0412,   0.0971, ],
                [ -0.0590,   0.0212,   0.6619,   4.5480,   0.1701,   0.0000, ],
                [  0.0413,   0.0000,  -0.0412,   0.1702,   4.3293,  -0.0897, ],
                [  0.7240,   0.1517,   0.0972,   0.0000,  -0.0896,   4.5480, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["N0.8_C2_juxtaposed_0_T300_nanoIndent"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  6.25086050,   3.09376570,   3.31509850,  -0.04882133,  -0.05375654,   0.05341430, ],
                [  3.09353500,   6.69329640,   3.61480610,  -0.23356716,  -0.01006164,   0.04457562, ],
                [  3.31469750,   3.61422290,  24.07620300,  -1.54341541,  -0.25938050,   0.03069533, ],
                [ -0.04885639,  -0.23363743,  -1.54331825,   5.03395520,   0.04736892,  -0.02695752, ],
                [ -0.05374503,  -0.01004050,  -0.25945045,   0.04746944,   4.65745940,  -0.11760305, ],
                [  0.05336555,   0.04449516,   0.03071204,  -0.02697618,  -0.11758055,   3.27553040, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["N0.8_C2_juxtaposed_1_T300_nanoIndent"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  6.96855420,   3.53761930,   3.64890640,  -0.01942356,  -0.03496586,   0.02033887, ],
                [  3.53742080,   7.30143980,   4.04355620,  -0.32990764,  -0.00382285,   0.04840180, ],
                [  3.64910740,   4.04369900,  23.16822400,  -1.66489253,  -0.21015371,   0.04785104, ],
                [ -0.01946521,  -0.32993115,  -1.66504004,   5.25153520,   0.06938761,  -0.03536723, ],
                [ -0.03502826,  -0.00386908,  -0.21023871,   0.06955431,   4.75809480,  -0.13934665, ],
                [  0.02038838,   0.04842597,   0.04783325,  -0.03536385,  -0.13929943,   3.47087660, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["N0.8_C2_juxtaposed_2_T300_nanoIndent"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  7.16544760,   3.72083810,   3.88285800,  -0.03910387,  -0.11201336,  -0.00626386, ],
                [  3.72040620,   7.66084260,   4.30838010,  -0.33120964,  -0.04624726,  -0.00235565, ],
                [  3.88238060,   4.30803730,  24.18087400,  -2.11838613,  -0.85385722,   0.10735234, ],
                [ -0.03910309,  -0.33118977,  -2.11831882,   5.43324540,   0.11848164,  -0.08538772, ],
                [ -0.11203715,  -0.04624977,  -0.85380167,   0.11845491,   4.92207040,  -0.14685281, ],
                [ -0.00630647,  -0.00235138,   0.10738145,  -0.08538730,  -0.14686737,   3.56175600, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["N0.8_C2_juxtaposed_3_T300_nanoIndent"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  6.96018140,   3.51879720,   3.68372640,  -0.02231061,  -0.00649245,   0.08606819, ],
                [  3.51832070,   7.31078260,   3.97372400,  -0.16918162,  -0.00542938,   0.08915416, ],
                [  3.68340870,   3.97390340,  26.13092500,  -1.21139707,   0.01812057,   0.00556122, ],
                [ -0.02233117,  -0.16921208,  -1.21142415,   5.28105260,   0.04176776,  -0.01308614, ],
                [ -0.00646717,  -0.00542786,   0.01820938,   0.04181259,   4.97834120,  -0.08637446, ],
                [  0.08602295,   0.08907934,   0.00548322,  -0.01311826,  -0.08639146,   3.53099940, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["N0.8_C2_juxtaposed_4_T300_nanoIndent"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  7.39335290,   3.77751470,   3.95362630,  -0.01410704,  -0.11689822,   0.07496234, ],
                [  3.77753030,   7.60368040,   4.12800300,  -0.06073188,  -0.01597787,   0.09351266, ],
                [  3.95349740,   4.12866690,  25.31364300,  -0.50223020,  -0.91196154,   0.04224493, ],
                [ -0.01409578,  -0.06073022,  -0.50225171,   5.24004860,   0.13435068,  -0.07245378, ],
                [ -0.11690525,  -0.01597716,  -0.91198961,   0.13437094,   5.05889200,  -0.05190312, ],
                [  0.07494860,   0.09349166,   0.04217829,  -0.07248562,  -0.05188823,   3.63131360, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["Averaged_C2_microstructure_T300_nanoIndent"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  6.9455,   3.5257,   3.6858,   0.0248,   0.0625,   0.0469, ],
                [  3.5255,   7.2952,   3.9972,   0.2095,   0.0157,   0.0550, ],
                [  3.6855,   3.9971,  24.4621,   1.3079,   0.4235,   0.0467, ],
                [  0.0248,   0.2095,   1.3079,   5.2410,   0.0807,   0.0453, ],
                [  0.0626,   0.0157,   0.4235,   0.0807,   4.8735,   0.1010, ],
                [  0.0469,   0.0550,   0.0467,   0.0454,   0.1010,   3.4894, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["B3_first_microstructure0_T300_nanoIndent"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  8.22568400,   4.58216440,   4.62548010,  -0.04029686,  -0.09321788,   0.05070894, ],
                [  4.58198970,   9.91864580,   5.37814310,  -0.38404960,  -0.08433120,   0.13934489, ],
                [  4.62559680,   5.37859620,  20.17455400,  -0.71959547,  -0.55487833,   0.09287116, ],
                [ -0.04027293,  -0.38402252,  -0.71955149,   6.43357640,   0.14056308,  -0.14151186, ],
                [ -0.09327644,  -0.08435508,  -0.55491274,   0.14049283,   4.71484220,  -0.11286107, ],
                [  0.05071887,   0.13934056,   0.09282295,  -0.14145141,  -0.11283847,   4.05342420, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["B3_first_microstructure1_T300_nanoIndent"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  8.17186810,   4.58563940,   4.64213140,  -0.01648812,  -0.00648199,  -0.04355343, ],
                [  4.58566940,   9.48159060,   5.29616370,  -0.13568718,   0.02239759,  -0.10586094, ],
                [  4.64204370,   5.29634740,  21.29727900,  -0.75115330,  -0.19060063,  -0.05783480, ],
                [ -0.01651781,  -0.13572631,  -0.75118315,   6.15880900,  -0.09119630,   0.03140004, ],
                [ -0.00655177,   0.02232942,  -0.19072839,  -0.09120067,   4.69533440,  -0.05859932, ],
                [ -0.04366292,  -0.10595510,  -0.05796106,   0.03137655,  -0.05863848,   3.93577020, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["B3_first_microstructure2_T300_nanoIndent"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  8.25285090,   4.59305300,   4.69785080,  -0.03210116,  -0.12061665,   0.03895176, ],
                [  4.59271930,   9.84246630,   5.53702280,  -0.49825821,  -0.08921178,   0.14871126, ],
                [  4.69774760,   5.53684500,  20.25439600,  -1.60270744,  -1.05554151,   0.12985094, ],
                [ -0.03207636,  -0.49822990,  -1.60260647,   6.74457560,   0.22519826,  -0.17903157, ],
                [ -0.12062556,  -0.08919999,  -1.05551584,   0.22530264,   4.80531260,  -0.12683853, ],
                [  0.03896235,   0.14869997,   0.12984474,  -0.17903970,  -0.12682900,   3.99595200, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["B3_first_microstructure3_T300_nanoIndent"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  8.38658500,   4.64148820,   4.66004590,  -0.00746083,  -0.06281197,  -0.07642202, ],
                [  4.64183810,  10.62818900,   5.69643620,  -0.08323579,   0.01096300,  -0.30960195, ],
                [  4.66014370,   5.69639170,  19.24471700,   0.22896051,  -0.51669475,  -0.06876225, ],
                [ -0.00743379,  -0.08321660,   0.22904481,   7.29697820,  -0.16224970,   0.00480450, ],
                [ -0.06285088,   0.01094671,  -0.51680626,  -0.16222230,   4.78696300,   0.00038183, ],
                [ -0.07648067,  -0.30968145,  -0.06875215,   0.00482407,   0.00036768,   4.17295480, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["B3_first_microstructure4_T300_nanoIndent"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  8.20436260,   4.54286650,   4.61220880,  -0.01978385,  -0.09224770,   0.01953271, ],
                [  4.54307070,  10.03406800,   5.48202340,  -0.30552350,  -0.04198324,   0.02479550, ],
                [  4.61177880,   5.48174040,  20.01743300,  -0.82304480,  -0.75790092,   0.06724367, ],
                [ -0.01982943,  -0.30552014,  -0.82312049,   6.83726980,   0.10874764,  -0.08646191, ],
                [ -0.09224950,  -0.04201550,  -0.75786209,   0.10866492,   4.78446580,  -0.06836998, ],
                [  0.01950543,   0.02479269,   0.06720164,  -0.08644907,  -0.06835130,   4.04455140, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["D3_meso_averaged"]={ # this is D3_micro actually
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                # Averaging vertical and out-of-plane:
                [  8.3726,   4.5677,   4.5365,   0.0370,   0.1272,   0.0854, ],
                [  4.5682,  18.6165,   5.3381,   0.8751,   0.1676,   0.3508, ],
                [  4.5368,   5.3379,  13.0631,   0.4295,   0.3976,   0.0754, ],
                [  0.0370,   0.8751,   0.4295,   7.3480,   0.2267,   0.3102, ],
                [  0.1271,   0.1676,   0.3975,   0.2267,   4.6303,   0.1162, ],
                [  0.0854,   0.3508,   0.0755,   0.3102,   0.1162,   5.0073, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["D3_meso_wholePeriod"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                # Averaged C tensor, case: N0.4_0_90_all
                [  7.5682,   4.0969,   4.0084,   0.0109,   0.0346,   0.0183, ],
                [  4.0963,  16.6153,   4.7316,   0.2086,   0.0234,   0.1022, ],
                [  4.0081,   4.7316,  12.7900,   0.2684,   0.3640,   0.0244, ],
                [  0.0108,   0.2086,   0.2685,   6.0733,   0.0235,   0.0485, ],
                [  0.0346,   0.0233,   0.3640,   0.0236,   4.1538,   0.0251, ],
                [  0.0182,   0.1021,   0.0243,   0.0485,   0.0251,   4.3636, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["B6_fourth_2ndAttempt"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  8.77480360,   4.51328020,   4.73715850,   0.01788955,   0.27197577,   0.01908784, ],
                [  4.51337400,  10.35439500,   5.46313640,   0.04779645,   0.08789205,   0.01196147, ],
                [  4.73699600,   5.46353660,  24.97557500,  -0.56291200,   1.41217355,  -0.04604223, ],
                [  0.01789703,   0.04783335,  -0.56272412,   9.01007960,   0.04304884,   0.31940302, ],
                [  0.27192992,   0.08784881,   1.41205420,   0.04306810,   6.34299100,  -0.02407919, ],
                [  0.01897536,   0.01187777,  -0.04625663,   0.31947368,  -0.02404983,   4.61499140, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["B6_fourth_microstructure0"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  8.71805700,   4.54976090,   4.72626110,   0.01233113,   0.08700828,   0.04120264, ],
                [  4.54968270,  10.95215400,   5.76293470,   0.68736556,   0.05612410,   0.13939220, ],
                [  4.72629300,   5.76338770,  22.57113700,   0.81993742,   0.50225853,   0.13062620, ],
                [  0.01232819,   0.68733011,   0.81996223,   9.26455660,   0.29277144,   0.10970643, ],
                [  0.08698245,   0.05612995,   0.50225257,   0.29292936,   6.05550120,   0.12737929, ],
                [  0.04122427,   0.13935040,   0.13071878,   0.10959423,   0.12739464,   4.65460420, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}


dict_material_properties_all["B6_fourth_microstructure1"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  8.73185270,   4.55634310,   4.74552230,   0.05555676,   0.19791281,   0.13089294, ],
                [  4.55613010,  10.88036200,   5.69169270,   0.71785943,   0.11814968,   0.27913088, ],
                [  4.74553260,   5.69163380,  21.99906700,   0.98960520,   0.84899259,   0.17913686, ],
                [  0.05555730,   0.71787936,   0.98960454,   8.95823180,   0.45403446,   0.25605478, ],
                [  0.19793547,   0.11822779,   0.84897105,   0.45402996,   6.10688940,   0.17929304, ],
                [  0.13084412,   0.27908055,   0.17912785,   0.25612620,   0.17933032,   4.71593540, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["B6_fourth_microstructure2"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  8.49650000,   4.40467100,   4.59764290,  -0.00888541,  -0.06409883,   0.02678748, ],
                [  4.40506840,   9.70403010,   5.72916210,  -0.35500073,  -0.01001951,   0.00888280, ],
                [  4.59745740,   5.72911300,  27.64418700,  -1.72026140,  -0.19307224,  -0.02765633, ],
                [ -0.00894625,  -0.35503068,  -1.72038783,   8.65594760,  -0.05274652,  -0.00879993, ],
                [ -0.06408934,  -0.01000546,  -0.19289285,  -0.05294788,   6.18761680,  -0.14055750, ],
                [  0.02678287,   0.00889562,  -0.02766875,  -0.00878969,  -0.14053730,   4.37113300, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["B6_fourth_microstructure3"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  7.74851120,   4.03458150,   4.19905290,  -0.05204971,   0.00462550,  -0.08723774, ],
                [  4.03455250,   9.96574550,   5.46457710,  -0.63565597,   0.08698311,  -0.17828951, ],
                [  4.19875500,   5.46488160,  24.22621700,  -2.16939554,   0.14492056,  -0.13093874, ],
                [ -0.05199869,  -0.63561635,  -2.16925426,   8.51958780,  -0.22958764,   0.11864906, ],
                [  0.00451782,   0.08693209,   0.14478406,  -0.22960658,   5.78048220,  -0.19326150, ],
                [ -0.08750794,  -0.17849873,  -0.13112617,   0.11865261,  -0.19322821,   4.32005820, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["B_averaged_B3_first_B6_fourth"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  8.3711,   4.5004,   4.6243,   0.0263,   0.1001,   0.0534, ],
                [  4.5004,  10.1762,   5.5501,   0.3850,   0.0608,   0.1346, ],
                [  4.6242,   5.5502,  22.2405,   1.0388,   0.6177,   0.0931, ],
                [  0.0263,   0.3850,   1.0387,   7.7880,   0.1800,   0.1256, ],
                [  0.1001,   0.0608,   0.6177,   0.1800,   5.4260,   0.1032, ],
                [  0.0535,   0.1346,   0.0931,   0.1256,   0.1032,   4.2879, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["N0.25_averaged_first_second"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  7.7154,   4.1587,   4.2752,   0.0098,   0.0659,   0.0544, ],
                [  4.1588,   7.9959,   4.4725,   0.0850,   0.0178,   0.0603, ],
                [  4.2752,   4.4725,  23.5791,   0.5743,   0.4574,   0.0304, ],
                [  0.0098,   0.0850,   0.5743,   5.2488,   0.0750,   0.0457, ],
                [  0.0659,   0.0178,   0.4575,   0.0751,   4.7997,   0.0427, ],
                [  0.0544,   0.0603,   0.0304,   0.0457,   0.0427,   3.6098, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["Averaged_alignedFFT_artificialRVE"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  8.6896,   4.9259,   5.0267,   0.0087,   0.0188,   0.0180, ],
                [  4.9259,   8.6742,   5.0162,   0.0097,   0.0041,   0.0113, ],
                [  5.0267,   5.0162,  30.5544,   0.0670,   0.0475,   0.0147, ],
                [  0.0087,   0.0097,   0.0670,   4.6278,   0.0306,   0.0059, ],
                [  0.0188,   0.0041,   0.0475,   0.0306,   4.6287,   0.0107, ],
                [  0.0180,   0.0113,   0.0147,   0.0059,   0.0107,   3.6885, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}

dict_material_properties_all["Averaged_alignedFFT_artificialRVE_full"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  7.7604,   4.2659,   4.3283,   0.0067,   0.0144,   0.0137, ],
                [  4.2659,   7.7602,   4.3295,   0.0167,   0.0074,   0.0138, ],
                [  4.3283,   4.3294,  29.8857,   0.0801,   0.0848,   0.0139, ],
                [  0.0067,   0.0167,   0.0801,   4.4995,   0.0220,   0.0102, ],
                [  0.0144,   0.0074,   0.0848,   0.0220,   4.4989,   0.0071, ],
                [  0.0137,   0.0138,   0.0139,   0.0102,   0.0071,   3.4709, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}



dict_material_properties_all["Averaged_alignedFFT_artificialRVE_0_90"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  8.6896,   4.9763,   4.9763,   0.0000,   0.0004,   0.0184, ],
                [  4.9763,  19.6143,   5.0162,  -0.0287,  -0.0053,   0.0294, ],
                [  4.9763,   5.0162,  19.6143,   0.0287,   0.0181,   0.0094, ],
                [  0.0000,  -0.0286,   0.0286,   4.6278,   0.0182,  -0.0123, ],
                [  0.0004,  -0.0053,   0.0181,   0.0182,   4.1586,   0.0000, ],
                [  0.0184,   0.0294,   0.0094,  -0.0123,  -0.0000,   4.1586, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}


dict_material_properties_all["Averaged_alignedFFT_artificialRVE_0_90_full"]={
    'matrix': {
        'behavior':"orthotropic",
        'C'  : np.array(
            [
                [  7.7604,   4.2971,   4.2971,   0.0000,   0.0003,   0.0141, ],
                [  4.2971,  18.8230,   4.3294,  -0.0317,  -0.0033,   0.0493, ],
                [  4.2971,   4.3294,  18.8230,   0.0317,   0.0355,   0.0106, ],
                [  0.0000,  -0.0317,   0.0317,   3.3746,   0.0136,  -0.0004, ],
                [  0.0003,  -0.0033,   0.0355,   0.0136,   3.1172,   0.0018, ],
                [  0.0141,   0.0493,   0.0106,  -0.0004,   0.0018,   2.8602, ],
            ]
        )
    },
    'fiber': {
        'behavior': 'none',
    },
    "mat_lib_path":mat_lib_path
}





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

