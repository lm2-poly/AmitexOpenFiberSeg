import numpy as np
import viscid as vs
import tensor_personal_functions as tensF

from scipy.optimize import curve_fit,fmin

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams["font.family"] = "Times New Roman"

def rotateAndProject(params,C):
    theta,phi=params
    P=vs.eul2dcm( [-phi, theta,0.], 'zyz', unit='deg')

    C_rotated   =tensF.tensorial_base_change(P,tensF.voigt4_to_tensor4_Facu(C))
    C_rotated   =tensF.tensor4_to_voigt4_Facu(C_rotated)

    S=np.linalg.inv(C_rotated)

    return 1/S[0,0],1/S[1,1],1/S[2,2]

C_perType={
    "testInput_iso":tensF.generate_isoC_from_E_nu(4., 0.3855),
    "testInput_trans_iso":tensF.generate_trans_isoC_from_E_nu_G(230,13.4,0.256,0.3,27.3,2),

    "N0.25_all":np.array(
                [
                    [  6.2851,   3.2602,   3.4416,   0.0040,   0.0187,   0.0231, ],
                    [  3.2601,   6.9032,   3.7252,   0.0309,   0.0039,   0.0224, ],
                    [  3.4414,   3.7250,  21.3773,   0.2181,   0.1813,   0.0397, ],
                    [  0.0040,   0.0309,   0.2182,   4.8124,   0.0239,   0.0131, ],
                    [  0.0188,   0.0039,   0.1814,   0.0239,   4.4069,   0.0154, ],
                    [  0.0231,   0.0225,   0.0398,   0.0131,   0.0153,   3.2115, ],
                ]
            ),
    "N0.25_dual":np.array( #"N0.25_meso with N0.25_averaged_first_second
                [
                    [  5.55826680,   2.80580810,   2.96074060,   0.00002673,   0.00624373,  -0.02973811, ],
                    [  2.80561770,   6.22786200,   3.26150440,  -0.00235489,   0.00263223,  -0.03835737, ],
                    [  2.96071390,   3.26142650,  19.89251600,  -0.00285557,   0.00982625,  -0.02404902, ],
                    [  0.00001416,  -0.00236668,  -0.00286705,   4.57402520,  -0.02074333,   0.00162027, ],
                    [  0.00622126,   0.00261498,   0.00980849,  -0.02067349,   4.03231280,   0.00003948, ],
                    [ -0.02979023,  -0.03840783,  -0.02412123,   0.00161354,   0.00007526,   2.95846080, ],
                ]
            ),
            

    "N0.25_micro":np.array( 
            [
                [  7.7154,   4.1587,   4.2752,   0.0098,   0.0659,   0.0544, ],
                [  4.1588,   7.9959,   4.4725,   0.0850,   0.0178,   0.0603, ],
                [  4.2752,   4.4725,  23.5791,   0.5743,   0.4574,   0.0304, ],
                [  0.0098,   0.0850,   0.5743,   5.2488,   0.0750,   0.0457, ],
                [  0.0659,   0.0178,   0.4575,   0.0751,   4.7997,   0.0427, ],
                [  0.0544,   0.0603,   0.0304,   0.0457,   0.0427,   3.6098, ],
            ]
        ),

    "N0.4_meso":np.array( #"B6_meso_rotated_x_y with B_averaged_B3_first_B6_fourth"
            [
                [  4.20414620,   2.10070940,   2.24024940,  -0.00310092,  -0.01203265,  -0.04401174, ],
                [  2.10102360,   7.06352420,   3.44265770,  -0.00591625,  -0.00579415,  -0.04515337, ],
                [  2.24020430,   3.44224850,  17.69452900,  -0.01699946,  -0.01477177,  -0.03176885, ],
                [ -0.00310855,  -0.00592676,  -0.01698856,   6.32614460,  -0.02544775,  -0.00185493, ],
                [ -0.01203245,  -0.00580224,  -0.01481233,  -0.02527851,   3.74926480,  -0.00446579, ],
                [ -0.04411948,  -0.04519950,  -0.03179352,  -0.00176259,  -0.00438645,   2.85308220, ],
            ]
        ),

    "N0.4_micro":np.array( #"B_averaged_B3_first_B6_fourth"
            [
                [  8.37110000,   4.50040000,   4.62430000,   0.02630000,   0.10010000,   0.05340000, ],
                [  4.50040000,  10.17620000,   5.55010000,   0.38500000,   0.06080000,   0.13460000, ],
                [  4.62420000,   5.55020000,  22.24050000,   1.03880000,   0.61770000,   0.09310000, ],
                [  0.02630000,   0.38500000,   1.03870000,   7.78800000,   0.18000000,   0.12560000, ],
                [  0.10010000,   0.06080000,   0.61770000,   0.18000000,   5.42600000,   0.10320000, ],
                [  0.05350000,   0.13460000,   0.09310000,   0.12560000,   0.10320000,   4.28790000, ],
            ]
        ),
    "N0.4_micr_90-90":np.array( #"sampleT1_90_90"
            [
                # rotated by theta 90 and phi 90
                [  8.2884,   4.6037,   4.3704,  -0.0080,  -0.0101,   0.0941, ],
                [  4.6038,  26.7439,   4.4997,  -0.3148,  -0.0240,   0.7632, ],
                [  4.3703,   4.4994,   8.0740,  -0.0390,  -0.0094,   0.0097, ],
                [ -0.0080,  -0.3147,  -0.0390,   2.4441,   0.0202,  -0.0156, ],
                [ -0.0102,  -0.0241,  -0.0094,   0.0202,   1.8653,  -0.0147, ],
                [  0.0941,   0.7632,   0.0097,  -0.0155,  -0.0147,   2.6970, ],
            ]
        ),

    "N0.4_micro_0-90":np.array( #"D3_meso_averaged"
            [
                [  8.3726,   4.5677,   4.5365,   0.0370,   0.1272,   0.0854, ],
                [  4.5682,  18.6165,   5.3381,   0.8751,   0.1676,   0.3508, ],
                [  4.5368,   5.3379,  13.0631,   0.4295,   0.3976,   0.0754, ],
                [  0.0370,   0.8751,   0.4295,   7.3480,   0.2267,   0.3102, ],
                [  0.1271,   0.1676,   0.3975,   0.2267,   4.6303,   0.1162, ],
                [  0.0854,   0.3508,   0.0755,   0.3102,   0.1162,   5.0073, ],
            ]
        ),

    "N0.8_micro":np.array( #"Averaged_C2_microstructure_T300_nanoIndent"
            [
                [  6.94550000,   3.52570000,   3.68580000,   0.02480000,   0.06250000,   0.04690000, ],
                [  3.52550000,   7.29520000,   3.99720000,   0.20950000,   0.01570000,   0.05500000, ],
                [  3.68550000,   3.99710000,  24.46210000,   1.30790000,   0.42350000,   0.04670000, ],
                [  0.02480000,   0.20950000,   1.30790000,   5.24100000,   0.08070000,   0.04530000, ],
                [  0.06260000,   0.01570000,   0.42350000,   0.08070000,   4.87350000,   0.10100000, ],
                [  0.04690000,   0.05500000,   0.04670000,   0.04540000,   0.10100000,   3.48940000, ],
            ]
        ),

    "N0.8_meso":np.array( #"Averaged_C2_microstructure_T300_nanoIndent"
            [
                [  5.60445510,   2.68751990,   2.84213760,  -0.00125919,  -0.00003707,  -0.03158125, ],
                [  2.68740090,   6.08240900,   3.15432550,  -0.00233425,   0.00022954,  -0.03733452, ],
                [  2.84211610,   3.15429670,  20.93462600,  -0.00624474,  -0.00056547,  -0.01837848, ],
                [ -0.00125847,  -0.00234006,  -0.00625619,   4.63147540,  -0.01662821,   0.00034491, ],
                [ -0.00004968,   0.00021840,  -0.00059251,  -0.01658423,   4.23049420,  -0.00133467, ],
                [ -0.03160910,  -0.03735381,  -0.01841333,   0.00034142,  -0.00131093,   2.97363000, ],
            ]
        ),

    "N0.4_meso_0-90":np.array( #"D3_meso_averaged"
            [
                [  5.13448860,   2.68987760,   2.65274530,  -0.00236056,   0.00664505,  -0.01110875, ],
                [  2.68979970,  15.33169400,   3.77847060,  -0.00536671,   0.00402347,  -0.00949717, ],
                [  2.65275260,   3.77808980,  10.57812700,  -0.00677505,  -0.00107741,  -0.00670500, ],
                [ -0.00242884,  -0.00543397,  -0.00684977,   6.41478080,  -0.00390887,   0.00218730, ],
                [  0.00663086,   0.00399784,  -0.00105763,  -0.00360242,   3.40828740,  -0.00462064, ],
                [ -0.01118395,  -0.00958541,  -0.00671294,   0.00217858,  -0.00447447,   3.77055860, ],
            ]
        ),
    "N0.4_micro_0-90Plate":np.array( #PlaqueN0.4_0-90_first_2022-04-28_174618 averaged vertical and OOP
            [
                [  7.8874,  3.9934,  4.0124,   0.1180,  -0.0067,  -0.0517, ],
                [  3.9939, 14.0757,  3.8585,   0.0677,  -0.0624,   0.1865, ],
                [  4.0124,  3.8586, 13.6044,   0.4068,  -0.0235,  -0.0675, ],
                [  0.1180,  0.0677,  0.4067,   3.8297,  -0.0921,  -0.0075, ],
                [ -0.0067, -0.0623, -0.0235,  -0.0921,   3.3439,   0.0748, ],
                [ -0.0518,  0.1864, -0.0675,  -0.0076,   0.0749,   3.1633, ],
            ]
        ),


    "N0.4_meso_90-90":np.array( #"B6_meso with  mat: sampleT1_90_90
            [
                # rotated by theta 90 phi 90
                [  4.0075,   2.1528,   1.9512,   0.0024,   0.0400,  -0.0100, ],
                [  2.1525,  21.2573,   2.6182,   0.0150,   0.0275,  -0.0149, ],
                [  1.9508,   2.6186,   5.4453,   0.0039,   0.0388,  -0.0048, ],
                [  0.0024,   0.0151,   0.0039,   2.0198,  -0.0006,   0.0096, ],
                [  0.0401,   0.0276,   0.0389,  -0.0005,   1.2183,   0.0022, ],
                [ -0.0101,  -0.0149,  -0.0048,   0.0094,   0.0023,   1.8101, ],
            ]
        ),
}


C_list_micro=[
    "N0.8_micro",
    "N0.4_micro",
    "N0.25_micro",
    "N0.4_micr_90-90",
    "N0.4_micro_0-90",
    "N0.4_micro_0-90Plate"
    ]

C_list_meso=[
    "N0.25_all",
    "N0.25_dual",
    "N0.8_meso",
    "N0.4_meso",
    "N0.4_meso_0-90",
    "N0.4_meso_90-90"
    ]

labels={
    "N0.8_micro"        :"0.8",
    "N0.4_micro"        :"0.4",
    "N0.25_micro"       :"0.25",
    "N0.4_micr_90-90"   :"0.4",
    "N0.4_micro_0-90"   :"0.4",
    "N0.4_micro_0-90Plate":"0.4P",

    "N0.25_all"         :"0.25 (simult.)",
    "N0.25_dual"        :"0.25 (dual)",
    "N0.8_meso"         :"0.8",
    "N0.4_meso"         :"0.4",
    "N0.4_meso_0-90"    :"0.4",
    "N0.4_meso_90-90"   :"0.4",
}

pattern={
    "N0.25_micro"       :"0°-0°",
    "N0.4_micro"        :"0°-0°",
    "N0.8_micro"        :"0°-0°",
    "N0.4_micro_0-90"   :"0°-90°",
    "N0.4_micro_0-90Plate":"0°-90°P",
    "N0.4_micr_90-90"   :"90°-90°",
    
    "N0.25_all"         :"0°-0°",
    "N0.25_dual"        :"0°-0°",
    "N0.4_meso"         :"0°-0°",
    "N0.8_meso"         :"0°-0°",
    "N0.4_meso_0-90"    :"0°-90°",
    "N0.4_meso_90-90"   :"90°-90°",
}

# fft  18.7552    exp:  19.4092   error:   0.0337
# fft  15.6034    exp:  16.3565   error:   0.0460
# fft  18.8238    exp:  18.5227   error:  -0.0163
# fft   8.7978    exp:   9.7354   error:   0.0963
# fft   4.3725    exp:   3.2096   error:  -0.3623

tensileResults={
    "N0.25_all" :19.4092,
    "N0.25_dual":19.409,
    "N0.8_meso" :18.5227,
    "N0.4_meso" :16.3565,
    "N0.4_meso_0-90":9.7354,
    "N0.4_meso_90-90":3.2096,
}

distance_ortho={}
distance_trans_iso={}
ratio_Ez_Ey={}
ratio_Ez_Ex={}
ratio_Ey_Ex={}

Ex={}
Ey={}
Ez={}
El={}


def rotateAndPlot(keyList,figName):

    plt.figure(figsize=[8,8],num=figName)

    for typeKey in keyList:

        C=C_perType[typeKey]

        #get orthotropic Values

        S=np.linalg.inv(C)

        S_ortho=np.zeros((6,6),np.float32)

        for i in range(3):
            for j in range(3):
                S_ortho[i,j]=S[i,j]

        for i in range(3,6):
            S_ortho[i,i]=S[i,i]

        C_ortho=np.linalg.inv(S_ortho)

        distance_ortho[typeKey]=tensF.matrix_distance(C,C_ortho)

        print("##############################################################\n\nType: {}".format(typeKey))
        print("\n")

        print("C input")
        tensF.printVoigt4(C)

        # I use a different reference frame in the paper
        Ez[typeKey]=1/S[0,0]
        Ey[typeKey]=1/S[1,1]
        Ex[typeKey]=1/S[2,2]

        # Orthotropic projections
        print("E_x\t{: >8.3f}".format(Ex[typeKey]))
        print("E_y\t{: >8.3f}".format(Ey[typeKey]))
        print("E_z\t{: >8.3f}".format(Ez[typeKey]))

        ratio_Ez_Ey[typeKey]=Ez[typeKey]/Ey[typeKey]
        ratio_Ez_Ex[typeKey]=Ez[typeKey]/Ex[typeKey]
        ratio_Ey_Ex[typeKey]=Ey[typeKey]/Ex[typeKey]

        print("C_ortho")
        tensF.printVoigt4(C_ortho)

        print("distance orthotropy: {: 8.4f}\n###########\n".format(distance_ortho[typeKey]))
      
        axis=2
        
        El[typeKey],E_t,nu_l,nu_t,G_l=tensF.transverse_isotropic_projector(C,axis)[-5:]
        
        C_trans_iso=tensF.generate_trans_isoC_from_E_nu_G(El[typeKey],E_t,nu_l,nu_t,G_l,axis)

        print("C_trans_iso")
        tensF.printVoigt4(C_trans_iso)

        distance_trans_iso[typeKey]=tensF.matrix_distance(C,C_trans_iso)

        print("distance transverse isotropy: {: 8.4f}".format(distance_trans_iso[typeKey]))
      
        print("\nCalculated homogeneous properties (with axis being z) :\n\nE_l\t\t={: >10.4f}\nE_t\t\t={: >10.4f}\nnu_l\t\t={: >10.4f}\nnu_t\t\t={: >10.4f}\nG_l\t\t={: >10.4f}\n".\
                                        format(El[typeKey],E_t,nu_l,nu_t,G_l))

        # evaluate E value for all directions in the transverse plane (x-y plane)

        theta=0.
        phi=np.linspace(0,360,10)

        results=[rotateAndProject([theta,p],C) for p in phi]
        E_xx=[]
        E_yy=[]
        E_zz=[]

        for exx,eyy,ezz in results:
            E_xx.append(exx)
            E_yy.append(eyy)
            E_zz.append(ezz)

        ax = plt.subplot(111, projection='polar')

        ax.plot(np.radians(phi+90),E_xx,label=typeKey)

    plt.legend()



rotateAndPlot(C_list_micro,"Micro")
rotateAndPlot(C_list_meso,"Meso")

print("Error table")

print("\t\ttrans. iso\torthotropy\tEx/Ey\t\tEx/Ez\t\tEy/Ez")
for typeKey in distance_ortho.keys():
    print("{: >14}\t{: >8.4f}\t{: >8.4f}\t{: >8.4f}\t{: >8.4f}\t{: >8.4f}".format(
        typeKey,
        distance_trans_iso[typeKey],
        distance_ortho[typeKey],
        ratio_Ez_Ey[typeKey],
        ratio_Ez_Ex[typeKey],
        ratio_Ey_Ex[typeKey]
        )
    )

print("\n\tMicrostructures:")

print("\t\t\t  ;Pattern ;\t\t\tE_x\t\t   ;E_y\t\t    ;E_z\t\t")
for typeKey in C_list_micro:
    print("{: >16}\t;{: >16}\t;{: >8.2f}\t;{: >8.2f}\t;{: >8.2f}\t".format(
        labels[typeKey],
        pattern[typeKey],
        Ex[typeKey],
        Ey[typeKey],
        Ez[typeKey],
        )
    )


print("\n\tMacrostructures:")

print("\t\t\t  ;Pattern ;\t\t\tE_x\t\t   ;E_y\t\t    ;E_z\t    ;Exp. E_x;\tRelative error (%)")
for typeKey in C_list_meso:
    print("{: >16}\t;{: >16}\t;{: >8.2f}\t;{: >8.2f}\t;{: >8.2f}\t;{: >8.2f}\t;{: >8.2%}".format(
        labels[typeKey],
        pattern[typeKey],
        Ex[typeKey],
        Ey[typeKey],
        Ez[typeKey],
        tensileResults[typeKey],
        np.abs(1-Ex[typeKey]/tensileResults[typeKey])
        )
    )

plt.show()
