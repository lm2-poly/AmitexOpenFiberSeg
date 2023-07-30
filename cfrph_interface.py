# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import multiprocessing
import os
from joblib import Parallel, delayed

import re

import numpy as np
from skimage.draw import ellipse
import quaternion as qt
import vtk
from vtk.util import numpy_support
import xml.etree.ElementTree as et
import intervals as it
import itertools
from cfrph_utils import *
import subprocess
from tensor_personal_functions import generate_trans_isoC_from_E_nu_G, isotropic_projector_Facu,\
    printVoigt4, transverse_isotropic_projector, write_to_file_Voigt4, modifyConvention,\
    voigt4_to_tensor4_Facu,generate_isoC_from_E_nu,convert_kappa_mu_to_lambda_mu,extract_isotropic_parameters

from tifffile import TiffFile
import pickle

from materialProperties import pathMT_library,max_vtk_size,showPlotsViscoelasticity,nIterMaxAmitex

import sys

from matplotlib import pyplot as plt

sys.path.insert(0, pathMT_library)

print("path to Mori-Tanaka librairies: {}".format(pathMT_library))

from MoriTanakaFunctions import parallelEval
import hmgnzt_personal_functions as homF

from cfrph_utils import import_hmgnzt_quad

from postProcessing import compactifySubVolume


#parametres de simulation
num_cores = multiprocessing.cpu_count()-2

_craft_foldername = '/craft_files'
_amitex_foldername = '/amitex_files'
_gmsh_foldername = '/gmsh_files'
_abaqus_foldername = '/abaqus_files'
_MoriTanaka_foldername="/MoriTanaka_files"


class CFRPHInterface(ABC):
    """Implementation of the tools required to interface the VER random
    generation code with the FFT and FEM homogenization codes

    Returns:
        HomogenizationInterface -- [description]
    """

    def __init__(
            self,
            rootpath,
            microstructure_name,
            materialDict,
            resolution,
            OpenFiberSeg=False,
            origin=None):
        """Interface instanciation

        Arguments:
            rootpath {str} -- Path where amtex_files, craft_files, etc. folders are stored
                              Example: /media/LM2_network/Public_Folders
            microstructure_name {str} -- Name of the .txt microstructure under study
                                         Example: N=35_R1=1.000_R2=0.100_f=0.100_n=1.txt
            material {dict} -- Mechanical properties of the materials
                               e.g.
                               * for isotropic fibers
                                material = {
                                    'matrix': {
                                        'young'  : 3,
                                        'poisson': 0.3
                                    },
                                    'fiber': {
                                        'behavior': 'iso',
                                        'young'    : 230,
                                        'poisson'  : 0.3
                                    }
                                }

                                * for transverse isotropic fibers
                                material = {
                                    'matrix': {
                                        'young'  : 3,
                                        'poisson': 0.3
                                    },
                                    'fiber': {
                                        'behavior': 'trans_iso',
                                        'young_l'  : 230,
                                        'young_t'  : 10.4,
                                        'poisson_l': 0.256,
                                        'poisson_t': 0.1,
                                        'shear_l'  : 27.3,
                                        'axis'     : 2
                                    }
                                }
            resolution {int} -- Voxel resolution e.g 128 for a 128*128*128 RVE
        """
        self.rootpath = rootpath if rootpath[-1]!="/" else rootpath[:-1] 
        self.rootpath.replace(" ","")
        if OpenFiberSeg:
            self.OpenFiberSeg = True
            self.microstructure_name = microstructure_name

            cmd = ["find", self.get_rootpath(), "-name",
                   "fiberStruct_*.pickle", "-type", "f"]
            systemCall = subprocess.run(cmd, stdout=subprocess.PIPE)

            nameFiberStruct = systemCall.stdout.decode(
                "utf-8").split("/")[-1].replace("\n", "")

            cmd = ["find", self.get_rootpath(), "-name",
                   "V_fiberMapCombined_postProcessed*.tiff", "-type", "f"]
            systemCall = subprocess.run(cmd, stdout=subprocess.PIPE)

            nameV_fibermap = systemCall.stdout.decode(
                "utf-8").split("/")[-1].replace("\n", "")

            cmd = ["find", self.get_rootpath(), "-name",
                   "V_pores*.tiff", "-type", "f"]
            systemCall = subprocess.run(cmd, stdout=subprocess.PIPE)

            nameV_pores = systemCall.stdout.decode(
                "utf-8").split("/")[-1].replace("\n", "")

            if nameFiberStruct == nameV_fibermap == nameV_pores == "":
                self.filesInFolder = microstructure_name
            else:
                self.nameFiberStruct = nameFiberStruct
                self.nameV_fibermap = nameV_fibermap
                self.nameV_pores = nameV_pores

                useInterface="interfaceVal" in materialDict["matrix"].keys()

                if useInterface:
                    cmd = ["find", self.get_rootpath(), "-name",
                    "V_interface*.tiff", "-type", "f"]
                    systemCall = subprocess.run(cmd, stdout=subprocess.PIPE)

                    nameV_interface = systemCall.stdout.decode(
                        "utf-8").split("/")[-1].replace("\n", "")

                    self.nameV_interface = nameV_interface


                self.filesInFolder = None
        else:
            self.OpenFiberSeg = False
            self.microstructure_name = microstructure_name

        check_material_format(materialDict)

        self.material = materialDict

        self.mat_lib_path = materialDict['mat_lib_path']



        if self.material['fiber']['behavior'] == 'trans_iso':
            # entire C tensor is only required for non-isotropic symmetry
            self.C = modifyConvention(
                generate_trans_isoC_from_E_nu_G(
                    self.get_E_l(),
                    self.get_E_t(),
                    self.get_nu_l(),
                    self.get_nu_t(),
                    self.get_G_l(),
                    self.get_axis()
                )
            )

        if self.material['fiber']['behavior'] == 'orthotropic':
            # entire C tensor is only required for non-isotropic symmetry
            self.C_fiber = modifyConvention(self.material['fiber']['C'] )

        if self.material['matrix']['behavior'] == 'orthotropic':
            # entire C tensor is only required for non-isotropic symmetry
            self.C_matrix = modifyConvention(self.material['matrix']['C'] )

        self.resolution=resolution

        if type(resolution) == int:
            self.resolution_type = "single value"

            self.nx = resolution
            self.ny = resolution
            self.nz = resolution

        elif type(resolution) == tuple and\
                len(resolution)==3:

            self.resolution_type = "three values"
            self.nx = resolution[0]
            self.ny = resolution[1]
            self.nz = resolution[2]

        elif resolution == "all":
            # this will encode to reshape the mat and zones array to the shape of tiff files found on file
            self.resolution_type = "all"

            self.nx = 1
            self.ny = 1
            self.nz = 1
        else:
            raise ValueError(
                "resolution passed in the wrong format: {}".format(resolution))

        if self.OpenFiberSeg:
            if origin is None:
                raise ValueError("Origin must be given, as a 3-value array")
            if len(origin) != 3:
                raise ValueError(
                    "Origin must be a 3-valued array, passed as {}".format(origin))
            self.origin_x, self.origin_y, self.origin_z = origin
        else:
            microstructure_features = read_input_microstructure(
                self.get_txt_filename())
            self.n_ellipsoids = microstructure_features['n_ellipsoids']
            self.first_aspect_ratio = microstructure_features['first_aspect_ratio']
            self.second_aspect_ratio = microstructure_features['second_aspect_ratio']
            self.desired_volume_fraction = microstructure_features['desired_volume_fraction']
            self.simulation_time = microstructure_features['simulation_time']
            self.reached_fiber_volume_fraction = microstructure_features[
                'reached_fiber_volume_fraction']
            self.reached_porosity_volume_fraction = microstructure_features[
                'reached_porosity_volume_fraction']
            self.ellipsoids = microstructure_features['ellipsoids']

        # start number for individual zones (for Amitex only)
        self.n_fiber_zones = 1  # wont work for any other number
        self.n_porosity_zones = 1  # wont work for any other number

        self.LUT_fibers_zones_to_base_vectors = {}

        if type(self).__name__ == 'CraftInterface':
            self.rve_array = np.zeros(
                (self.get_nx(), self.get_ny(), self.get_nz()), np.uint16)
        elif type(self).__name__ == 'AmitexInterface':
            self.rve_mat_array = np.ones(
                (self.get_nx(), self.get_ny(), self.get_nz()), np.int16)
            self.rve_zone_array = np.ones(
                (self.get_nx(), self.get_ny(), self.get_nz()), np.int16)
        elif type(self).__name__ == 'AbaqusInterface':
            self.n_nodes = 0
            self.node_sets = {
                'inner_volume': dict.fromkeys([get_volume_key()]),
                'inner_surface': dict.fromkeys(get_surface_keys()),
                'inner_edge': dict.fromkeys(get_edge_keys()),
                'vertex': dict.fromkeys(get_vertex_keys())
            }

        super(CFRPHInterface, self).__init__()

    @abstractmethod
    def preprocessing(self):
        pass

    @abstractmethod
    def processing(self):
        pass

    @abstractmethod
    def postprocessing(self):
        pass

    def ellipsoid_voxelization(self, el_id, zone_id):
        """Discretizes the ellipsoidal inclusion of identifier 'el_id' and
        incorporates it into the RVE arrays

        Arguments:
            el_id {int} -- Ellispoid identifier
            zone_id {int} -- Zone identifier (see Amitex documentation)
        @staticmethod
        Returns:
            [bool] -- Indicates if the RVE arrays has been filled in
        """
        has_point = False
        nx, ny, nz = self.get_nx(), self.get_ny(), self.get_nz()
        ellipsoid_features = self.get_ellipsoid(el_id)
        mat, rx, ry, rz, xc, yc, zc, quat, _ = ellipsoid_features.values()
        R = qt.as_rotation_matrix(quat)
        uz = np.matrix([0, 0, 1])
        ur = (np.linalg.inv(R)*uz.transpose()).transpose()
        ur_norm = np.sqrt((ur[0, 0]*rx)**2+(ur[0, 1]*ry)**2+(ur[0, 2]*rz)**2)
        m = (1/ur_norm)*np.matrix([ur[0, 0]*rx, ur[0, 1]*ry, ur[0, 2]*rz])
        z_interval = check_z_lim(it.closed(-ur_norm + zc, ur_norm + zc))

        if not(z_interval.is_empty()):
            d_inf_lim, d_max_lim = z_interval.lower-zc, z_interval.upper-zc
            current_z = np.arange(d_inf_lim, d_max_lim, 1/nz)

            for d in current_z:
                delta = d/ur_norm

                e0 = delta*m
                rho = np.sqrt(max(1-delta**2, 0.))

                e1 = rho/np.sqrt(m[0, 0]**2+m[0, 1]**2) * \
                    np.matrix([m[0, 1], -m[0, 0], 0])
                e2 = np.cross(m, e1)

                f0 = np.matrix([e0[0, 0]*rx, e0[0, 1]*ry, e0[0, 2]*rz])
                f1 = np.matrix([e1[0, 0]*rx, e1[0, 1]*ry, e1[0, 2]*rz])
                f2 = np.matrix([e2[0, 0]*rx, e2[0, 1]*ry, e2[0, 2]*rz])

                f0_r = R*f0.transpose()
                f1_r = R*f1.transpose()
                f2_r = R*f2.transpose()

                def x(t): return f1_r[0, 0]*np.cos(t)+f2_r[0, 0]*np.sin(t)
                def y(t): return f1_r[1, 0]*np.cos(t)+f2_r[1, 0]*np.sin(t)
                def norm(t): return np.sqrt(x(t)**2+y(t)**2)

                alpha = f1_r[0, 0]**2+f1_r[1, 0]**2
                beta = f1_r[0, 0]*f2_r[0, 0]+f1_r[1, 0]*f2_r[1, 0]
                gamma = f2_r[0, 0]**2+f2_r[1, 0]**2
                arg_min = -0.5*np.arctan2(2*beta, gamma-alpha)
                arg_max = -0.5*(np.arctan2(2*beta, gamma-alpha)+np.pi)

                gamma = np.arctan2(y(arg_min), x(arg_min))
                el_rx = norm(arg_min)
                el_ry = norm(arg_max)

                xx, yy = ellipse(
                    nx*(f0_r[0, 0]+xc),
                    ny*(f0_r[1, 0]+yc),
                    nx*el_rx,
                    ny*el_ry,
                    rotation=gamma)
                xx, yy = check_xy_lim(nx, ny, xx, yy)
                kz = int(nz*(d+zc))

                if len(xx) > 0:
                    self.set_rve_arrays(xx, yy, kz, zone_id, mat)
                    has_point = True

        return has_point, mat, quat

    def voxelize_rve(self):
        """Stores the RVE in arrays"""
        i = 1
        e1 = np.array([1., 0., 0.])
        e2 = np.array([0., 1., 0.])

        for ellipsoid_id in self.get_ellipsoids().keys():
            has_point, mat, quat = self.ellipsoid_voxelization(
                ellipsoid_id, i)  # i used in Craft numbering of regions
            if has_point:
                i += 1
                if mat == 'fiber':

                    e1prime = qt.rotate_vectors(quat, e1)
                    e2prime = qt.rotate_vectors(quat, e2)

                    self.LUT_fibers_zones_to_base_vectors[self.n_fiber_zones] = (
                        e1prime, e2prime)
                    self.n_fiber_zones += 1
                elif mat == 'void':
                    self.n_porosity_zones += 1

    @abstractmethod
    def set_rve_arrays(self, *_args):
        pass

    def get_rootpath(self):
        return self.rootpath

    def get_microstructure_name(self):
        return self.microstructure_name

    def get_generic_name(self):
        return '{}'.format(self.get_microstructure_name())

    def get_nx(self):
        return self.nx

    def get_ny(self):
        return self.ny

    def get_nz(self):
        return self.nz

    def get_txt_filename(self):
        #only used when if OpenFiberSeg==False (legacy microstructure generation)
        return '{}.txt'.format(
            os.path.join(
                os.path.dirname(self.get_rootpath()), 
                self.get_microstructure_name()
            )
        )

    def get_n_ellipsoids(self):
        return self.n_ellipsoids

    def get_ellipsoids(self):
        return self.ellipsoids

    def get_ellipsoid(self, id):
        return self.ellipsoids[id]

    def get_first_aspect_ratio(self):
        return self.first_aspect_ratio

    def get_second_aspect_ratio(self):
        return self.second_aspect_ratio

    def get_desired_volume_fraction(self):
        return self.desired_volume_fraction

    def get_reached_porosity_volume_fraction(self):
        return self.reached_porosity_volume_fraction

    def get_E_m(self):
        return self.material['matrix']['young']

    def get_nu_m(self):
        return self.material['matrix']['poisson']

    def get_E_f(self):
        return self.material['fiber']['young']

    def get_nu_f(self):
        return self.material['fiber']['poisson']

    def get_symmetry_class(self):
        return self.material['fiber']['behavior']

    def get_E_l(self):
        return self.material['fiber']['young_l']

    def get_E_t(self):
        return self.material['fiber']['young_t']

    def get_nu_l(self):
        return self.material['fiber']['poisson_l']

    def get_nu_t(self):
        return self.material['fiber']['poisson_t']

    def get_G_l(self):
        return self.material['fiber']['shear_l']

    def get_axis(self):
        return self.material['fiber']['axis']

    def loadVolumes(self,useInterface=False):
        with TiffFile(self.get_rootpath()+"/"+self.nameV_fibermap) as tif:
            # V_fibers = np.transpose(tif.asarray(),axes=(1,2,0))
            V_fibers = tif.asarray()


        with TiffFile(self.get_rootpath()+"/"+self.nameV_pores) as tif:
            # V_pores = np.transpose(tif.asarray(),axes=(1,2,0))
            V_pores = tif.asarray()

        if useInterface:
            with TiffFile(self.get_rootpath()+"/"+self.nameV_interface) as tif:
                V_interface = tif.asarray()

        else:
            V_interface=None
            V_interface_cropped = None


        with open(self.get_rootpath()+"/"+self.nameFiberStruct, "rb") as f:
            fiberStruct = pickle.load(f)

        if self.resolution_type == "all":
            self.nz, self.ny, self.nx = V_fibers.shape

            if self.nz>max_vtk_size or\
                self.nx>max_vtk_size or \
                self.ny>max_vtk_size:

                self.nz=min(self.nz,max_vtk_size)
                self.ny=min(self.ny,max_vtk_size)
                self.nx=min(self.nx,max_vtk_size)

                V_fibers_cropped = V_fibers[
                    self.origin_z:self.origin_z+self.nz,
                    self.origin_y:self.origin_y+self.ny,
                    self.origin_x:self.origin_x+self.nx,
                ]
                V_pores_cropped = V_pores[
                    self.origin_z:self.origin_z+self.nz,
                    self.origin_y:self.origin_y+self.ny,
                    self.origin_x:self.origin_x+self.nx,
                ]

                if useInterface:
                    V_interface_cropped = V_interface[
                        self.origin_z:self.origin_z+self.nz,
                        self.origin_y:self.origin_y+self.ny,
                        self.origin_x:self.origin_x+self.nx,
                    ]

            else:
                V_fibers_cropped = V_fibers
                V_pores_cropped = V_pores
                V_interface_cropped = V_interface

        else:
            V_fibers_cropped = V_fibers[
                self.origin_z:self.origin_z+self.nz,
                self.origin_y:self.origin_y+self.ny,
                self.origin_x:self.origin_x+self.nx,
            ]

            V_pores_cropped = V_pores[
                self.origin_z:self.origin_z+self.nz,
                self.origin_y:self.origin_y+self.ny,
                self.origin_x:self.origin_x+self.nx,
            ]

            if useInterface:
                V_interface_cropped = V_interface[
                    self.origin_z:self.origin_z+self.nz,
                    self.origin_y:self.origin_y+self.ny,
                    self.origin_x:self.origin_x+self.nx,
                ]

        # Handle the cases where the tiff files shape is smaller than the input resolution

        if not((self.nz,self.ny,self.nx)==V_fibers_cropped.shape):
            raise RuntimeError("Incoherent dimensions: V_fibers_cropped.shape={}, (nz,ny,nx)=({},{},{})".\
                format(V_fibers_cropped.shape,self.nz,self.ny,self.nx))


        self.fiberVoxelCount=len(np.where(V_fibers_cropped>-1)[0])

        self.poresVoxelCount=len(np.where(V_pores_cropped==255)[0])

        if useInterface:
            self.interfaceVoxelCount=len(np.where(V_interface_cropped==255)[0])

        self.totalVoxelCount=V_fibers_cropped.shape[0]*V_fibers_cropped.shape[1]*V_fibers_cropped.shape[2]

        self.fiberVolumeFraction=self.fiberVoxelCount/self.totalVoxelCount
        print("Total fiber volume fraction:\t{: >10.3%}".format(self.fiberVolumeFraction))

        if useInterface:
            self.interfaceVolumeFraction=self.interfaceVoxelCount/self.totalVoxelCount
            print("Total interface volume fraction:\t{: >10.3%}".format(self.interfaceVolumeFraction))

            self.fiberMatrixVolumeFraction=self.fiberVoxelCount/(self.totalVoxelCount-self.poresVoxelCount-self.interfaceVoxelCount)
            print("Total fiber/matrix volume fraction:\t{: >10.3%}".format(self.fiberMatrixVolumeFraction))

        else:

            self.fiberMatrixVolumeFraction=self.fiberVoxelCount/(self.totalVoxelCount-self.poresVoxelCount)
            print("Total fiber/matrix volume fraction:\t{: >10.3%}".format(self.fiberMatrixVolumeFraction))

        self.poresVolumeFraction=self.poresVoxelCount/self.totalVoxelCount
        print("Total pores volume fraction:\t{: >10.3%}".format(self.poresVolumeFraction))


        return V_fibers_cropped,V_pores_cropped,V_interface_cropped,fiberStruct

    @staticmethod
    def checkIfProcessedAlready(
        instanceType,
        current_path,
        material_tag,
        resolution,
        origin,
        symmetry_fibres,
        dict_material_properties,
        OpenFiberSeg=True,
        convergence_acceleration=None,
        openMP_threads=None
        ):

        log_file_name    = '{}log_mat_{}_res={}_origin_{}_sym={}_{}.txt'.format(
            current_path,material_tag,resolution,origin,symmetry_fibres,instanceType).replace("\'","")
        process_now_bool = False

        # Check if this microstructure has been processed
        if os.path.isfile(log_file_name):
            print(log_file_name,'found...\n')

            with open(log_file_name, 'r') as f:
                lines=f.readlines()

            if lines[0][:19] == 'processing complete':
                print('Already fully processed: {}\n'.format(current_path.split('/')[-1]))
            #if processing has not been completed, attempt again
            else:
                with open(log_file_name, 'w') as f:
                    lines[0]='Attemping to process...\n'
                    print("Attemping to process...\n")
                    process_now_bool=True               
                    f.writelines(lines) 
        else:
            #first attempt
            with open(log_file_name, 'w') as f:
                f.write('Attemping to process...\n')
                print('Attemping to process...\n')
                process_now_bool = True

        return process_now_bool


class CraftInterface(CFRPHInterface):

    def write_vtk_microstructure(self, printBool):
        self.voxelize_rve()
        nx, ny, nz = self.get_nx(), self.get_ny(), self.get_nz()

        rve = vtk.vtkStructuredPoints()
        rve.SetDimensions(nx, ny, nz)
        rve.SetOrigin(0, 0, 0)
        rve.SetSpacing(1/nx, 1/ny, 1/nz)
        scalars = numpy_support.numpy_to_vtk(
            num_array=self.get_rve_array().ravel(), array_type=vtk.VTK_FLOAT)
        scalars.SetName('scalars')
        rve.GetPointData().SetScalars(scalars)

        filename = self.get_vtk_filename()
        writer = vtk.vtkStructuredPointsWriter()
        writer.SetFileName(filename)
        writer.SetFileTypeToASCII()
        writer.SetInputData(rve)
        writer.Write()

        if printBool:
            print('Writing .vtk ...')

    def write_in(self):
        """Writes the CRaFT input file (see CRaFT documentation pp. 4-5)"""
        generic_name = self.get_generic_name()
        with open(self.get_in_filename(), 'w') as f:
            f.write('microstructure=craft_{}.vtk\n'.format(generic_name))
            f.write('phases={}.phases\n'.format(generic_name))
            f.write('materials={}.mat\n'.format(generic_name))
            f.write('loading={}.load\n'.format(generic_name))
            f.write('output={}.output\n'.format(generic_name))
            f.write('C0=auto\n')
            f.write('precision={}, {}'.format('1.e-2', '1.e-4'))

    def write_phases(self):
        """Writes the file describing the phases (see CRaFT documentation pp. 5-6)"""
        with open(self.get_phases_filename(), 'w') as f:
            f.write("#------------------------------------------------------------\n")
            f.write("# this file gives for each phase:\n")
            f.write("# * the material it is made of\n")
            f.write("# * its orientation (described by 3 Euler angles)\n")
            f.write("#\n")
            f.write("# This material comprises tho constituents made of two\n")
            f.write(
                "# different isotropic material (thus the Euler angles play no\n")
            f.write("# role and can be set to any value, e.g. 0,0,0)\n")
            f.write("#\n")
            f.write("#------------------------------------------------------------\n")
            f.write("# phase    material       phi1    Phi   phi2\n")
            f.write("#------------------------------------------------------------\n")

            # matrix
            f.write('0\t0\t0.\t\t0.\t\t0.\n')

            # inclusions (fibers and porosities)
            datas = ''
            i = 1
            for ellipsoid_id in self.get_ellipsoids().keys():
                ellipsoid_features = self.get_ellipsoid(ellipsoid_id)
                euler_angles = qt.as_euler_angles(ellipsoid_features['quat'])
                datas += '{}\t'.format(i)
                # matériau
                if ellipsoid_features['mat'] == 'fiber':
                    mat_id = 1
                elif ellipsoid_features['mat'] == 'void':
                    mat_id = 2
                datas += '{}\t'.format(mat_id)
                # angles d'Euler
                datas += '{0:.6f}\t{1:.6f}\t{2:.6f}\n'.format(*euler_angles)
                i += 1

            f.write(datas)

    def write_mat(self):
        """Writes the file describing the materials"""

        with open(self.get_mat_filname(), 'w') as f:
            f.write("#----------------------------------------------------\n")
            f.write("#\n")
            f.write("# material 0 is isotropic linear elastic:\n")
            f.write('0   1\n')
            f.write("#\n")
            f.write("# Young's Modulus\n")
            f.write('{0:.6f}\n'.format(self.get_E_m()))
            f.write("# Poisson coefficient\n")
            f.write('{0:.6f}\n'.format(self.get_nu_m()))
            if self.get_symmetry_class() == 'iso':  # isotropic symmetry
                f.write("#----------------------------------------------------\n")
                f.write("#\n")
                f.write("# material 1 is isotropic linear elastic:\n")
                f.write('1   1\n')  # material number \t behavior identifier
                f.write("# Young's Modulus\n")
                f.write('{0:.6f}\n'.format(self.get_E_f()))
                f.write("# Poisson coefficient\n")
                f.write('{0:.6f}\n'.format(self.get_nu_f()))
            elif self.get_symmetry_class() == 'trans_iso':  # transversely isotropic symmetry
                f.write("#----------------------------------------------------\n")
                f.write("#\n")
                f.write("# material 1 is transverse isotropic linear elastic:\n")
                f.write('1   3\n')  # material number \t behavior identifier
                f.write("# its stiffness matrix will be entered:\n")
                f.write("0\n")
                f.write("# stiffness matrix \n")
                C = self.C
                for i in range(6):
                    temp = ''
                    filler = ''
                    filler += '             '*i
                    my_list = ['{:4.6f}    '.format(val) for val in C[0+i, i:]]
                    my_list.append('\n')
                    f.write(filler+temp.join(my_list))

            # Porosity phase
            f.write("#----------------------------------------------------\n")
            f.write("#\n")
            f.write("# material 2  is a void material:\n")
            f.write('2   0\n')
            f.write("#no further parameters are required\n")
            f.write("#----------------------------------------------------\n")

    def write_load(self, prescribed_quantity, xfactor, prescribed_values=[0]*6):
        """Writes the file of the loading specification (see CRaFT documentation pp. 7-10)

        Arguments:
            prescribed_quantity {str} -- 'C', 'D' or 'S' respectively for prescribed
                                          stress, strain or direction of stress
            xfactor {float} -- amplitude of applied values, 1 for unitary loads

        Keyword Arguments:
            direction_vector {list} -- The 6 components of a 2d symmetrical tensor
                                        in the following order [11, 22, 33, 12, 13, 23]
                                        (default: {[0]*6})

        NB: Since the calculation is based on the hypothesis of small perturbations,
        it does not make sense to impose a high magnitude strain or stress.
        """
        with open(self.get_load_filname(), 'w') as f:
            f.write("#\n")
            f.write("# This file describes the loading conditions\n")
            f.write("#\n")
            f.write("#\n")
            f.write("#-----------------------------------------------\n")
            f.write(
                "#loading condition (prescribed stress (C), prescribed strain (D) or prescribed direction of stress (S))\n")
            f.write('{}\n'.format(prescribed_quantity))
            f.write('#- - - - - - - - - - - - - - - - - - - - - - - -\n')
            f.write('# time          direction of stress         \txfactor\n')
            f.write('#               11 22 33 12 13 23\n')

            # this is so the loading respects Voigt Modified convention
            list_factors = [1, 1, 1, 0.5, 0.5, 0.5]
            for i in range(6):
                direction_vector = [0]*6
                direction_vector[i] = 1
                f.write('{0}\t\t{1:.6f} {2:.6f} {3:.6f} {4:.6f} {5:.6f} {6:.6f}\t\t{7:.6f}\n'.format(
                    i+1, *direction_vector, list_factors[i]))
            f.write('#\n')
            # Temporary solution because CRaFT does not write the 6th step result files for simulation containing porosities
            f.write(
                '7\t\t1.000000 0.000000 0.000000 0.000000 0.000000 0.000000\t\t1.000000')
            f.write('#-----------------------------------------------')

    def write_output(self):
        """Writes the file of output specifications"""
        with open(self.get_output_filename(), 'w') as f:
            f.write('#\n')
            f.write('# File telling which outputs one wants to get\n')
            f.write('#--------------------------------------------------\n')
            f.write('# generic name of all result files:\n')
            f.write('generic name = {}\n'.format(self.get_generic_name()))
            f.write('#- - - - - - - - - - - - - - - - - - - - - - - - - \n')
            f.write('# stress moment has to be stored:\n')
            f.write('stress moment = yes\n')
            f.write('#\n')
            f.write('#- - - - - - - - - - - - - - - - - - - - - - - - - \n')
            f.write('# the image of the equivalent stress field has to \n')
            f.write('# be stored:\n')
            f.write('equivalent stress image = yes\n')
            f.write('#- - - - - - - - - - - - - - - - - - - - - - - - - \n')
            f.write('# An image of each component of the stress field \n')
            f.write('# has to be stored\n')
            f.write('stress image = yes 1.:6.:@1\n')
            f.write('# An image of each component of the strain field \n')
            f.write('# has to be stored\n')
            f.write('strain image = yes 1.:6.:@1\n')
            f.write('#- - - - - - - - - - - - - - - - - - - - - - - - - \n')

            f.write('# the image files in output has to be in vtk format:\n')
            f.write('# (default)\n')

    def get_average_stress(self, filename):
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(filename)
        reader.ReadAllScalarsOn()
        reader.Update()
        datas = reader.GetOutput()
        stress = numpy_support.vtk_to_numpy(
            datas.GetPointData().GetArray('scalars'))
        return np.mean(stress)

    def write_homogenized_stiffness_matrix(self):
        components = {1: 11, 2: 22, 3: 33, 4: 12, 5: 13, 6: 23}
        C_hom = np.zeros((6, 6))
        voigt_weights = modified_voigt_weights()
        print('Writing C_hom file ...')
        for i in components.keys():
            for j, direction in components.items():
                filename = '{}_t=0{}.00000000e+00_stress{}.vtk'.format(
                    self.get_generic_path(), i, direction)
                C_hom[i-1, j-1] = voigt_weights[i-1, j-1] * \
                    self.get_average_stress(filename)
        print("\n Homogenized tensor: \n")
        printVoigt4(C_hom)
        write_to_file_Voigt4(self.get_C_hom_filename(), C_hom)

    def write_homogenized_strain_matrix(self):
        components = {1: 11, 2: 22, 3: 33, 4: 12, 5: 13, 6: 23}
        strain_hom = np.zeros((6, 6))
        print('Writing strain_hom file ...')
        for i in components.keys():
            column = 6*[0]
            for j, direction in components.items():
                filename = '{}_t=0{}.00000000e+00_strain{}.vtk'.format(
                    self.get_generic_path(), i, direction)
                column[j-1] = self.get_average_stress(filename)
            strain_hom[:, i-1] = column

        write_to_file_Voigt4(self.get_strain_hom_filename(), strain_hom)

    def preprocessing(self, print_bool=True):
        self.write_vtk_microstructure(printBool)
        if print_bool:
            print('Writing .in .phases .mat .load .output ...')
        self.write_in()
        self.write_phases()
        self.write_mat()
        self.write_load('D', 1)
        self.write_output()

    def processing(self, craft_args, path_to_craft_binaries):
        """Runs CRaFT for the case under study

        Arguments:
            craft_args {list} -- List of the CRaFT arguments
                                 Example : ['-v', '-n 4', '-f']
            path_to_craft_binaries {str} -- path on local machine
        """
        print('Launching craft subprocess...')
        full_command = ['{}craft'.format(
            path_to_craft_binaries), *craft_args, self.get_in_filename()]
        subprocess.run(full_command, cwd=self.get_workspace_path())

    def postprocessing(self):
        self.write_homogenized_stiffness_matrix()
        self.write_homogenized_strain_matrix()

    def get_workspace_path(self):
        return '{}{}_res={}_sym={}/{}'.format(
            self.get_rootpath(),
            _craft_foldername,
            self.get_nx(),
            self.get_symmetry_class(),
            self.get_generic_name()
        )

    def get_rve_array(self):
        return self.rve_array

    def get_generic_path(self):
        return '{}/{}'.format(self.get_workspace_path(), self.get_generic_name())

    def get_in_filename(self):
        return '{}.in'.format(self.get_generic_path())

    def get_vtk_filename(self):
        return '{}/craft_{}.vtk'.format(self.get_workspace_path(), self.get_generic_name())

    def get_phases_filename(self):
        return '{}.phases'.format(self.get_generic_path())

    def get_mat_filname(self):
        return '{}.mat'.format(self.get_generic_path())

    def get_load_filname(self):
        return '{}.load'.format(self.get_generic_path())

    def get_output_filename(self):
        return '{}.output'.format(self.get_generic_path())

    def get_C_hom_filename(self):
        return '{}_C_hom.txt'.format(self.get_generic_path())

    def get_strain_hom_filename(self):
        return '{}_strain_hom.txt'.format(self.get_generic_path())

    def set_rve_arrays(self, xx, yy, kz, id, mat):
        self.rve_array[kz, yy, xx] = id


def read_homogenized_matrix(path):
    # both subfolder and files inside start with name of microstructure
    K = np.loadtxt('{}/{}_C_hom.txt'.format(path, os.path.basename(path)))
    return K


def getTiffProperties(tif, getDescription=False, getDimensions=False):

    try:
        # resolution is returned as a ratio of integers. value is in inches by default
        xRes = tif.pages[0].tags['XResolution'].value
        yRes = tif.pages[0].tags['YResolution'].value

        if xRes != yRes:
            raise ValueError('not implemented for unequal x and y scaling')

        unitEnum = tif.pages[0].tags['ResolutionUnit'].value
        if repr(unitEnum) == '<RESUNIT.CENTIMETER: 3>':
            unitTiff = "CENTIMETER"
        elif repr(unitEnum) == '<RESUNIT.INCH: 2>':
            unitTiff = "INCH"
        else:
            raise ValueError("not implemented for {}".format(repr(unitEnum)))
    except:
        print("\n\tTiff files do not contain scaling information, assigning default value of 1 micron/pixel")
        xRes = (int(1e4), 1)
        unitTiff = "CENTIMETER"

    if getDescription:
        descriptionStr = tif.pages[0].tags["ImageDescription"].value

        if getDimensions:
            return xRes, unitTiff, descriptionStr, tif.pages[0].shape
        else:
            return xRes, unitTiff, descriptionStr

    else:
        if getDimensions:
            return tif.pages[0].shape
        else:
            return xRes, unitTiff


class AmitexInterface(CFRPHInterface):

    # identifiers for each material in the RVE. used in input .vtk and .xlm files
    # must be seqential, so if there are no voids, the void ID must be greater than
    # the fiber and matrix ID.
    mat_ID_matrix_int = 1
    mat_ID_fiber_int = 2
    mat_ID_void_int = 3

    mat_ID_matrix_str = str(mat_ID_matrix_int)
    mat_ID_fiber_str = str(mat_ID_fiber_int)
    mat_ID_void_str = str(mat_ID_void_int)

    # setting this to False does not perform rotations on C tensors for inclusions, used in debugging
    use_zones_in_xml = True

    def __init__(
        self,
        rootpath,
        microstructure_name,
        materialDict,
        material_tag,
        resolution,
        OpenFiberSeg=False,
        origin=None,
        convergence_acceleration=True,
        openMP_threads=1,
        makeVTK_stress_strain=False
    ):
        super().__init__(rootpath, microstructure_name, materialDict,
                         resolution, OpenFiberSeg=OpenFiberSeg, origin=origin)

        self.useInterface="interfaceVal" in self.material["matrix"].keys()
        if self.useInterface:
            self.mat_ID_interface_int = 4
            self.mat_ID_interface_str = str(self.mat_ID_interface_int)
            
        self.makeVTK_stress_strain=makeVTK_stress_strain


        if "loadingSpecs" in self.material: #creep or relaxation

            loadingType=self.material["loadingSpecs"]["loadingType"].replace(" ","_")
            
            if "porosityThreshold" in self.material["loadingSpecs"]:
                porosityThreshold=self.material["loadingSpecs"]["porosityThreshold"]
            else:
                porosityThreshold=None

            if porosityThreshold is None:

                if self.useInterface:

                    self.workspace_path   = '{}{}_mat_{}_interfaceVal_({:3.2f},{:3.2f})_res={}_origin={}_loading={}'.format(
                        rootpath,
                        _amitex_foldername,
                        material_tag,
                        *self.material["matrix"]["interfaceVal"],
                        resolution,
                        origin,
                        loadingType
                    ).replace(" ","").replace("\'","").replace("//","/")

                else:

                    self.workspace_path   = '{}{}_mat_{}_res={}_origin={}_loading={}'.format(
                        rootpath,
                        _amitex_foldername,
                        material_tag,
                        resolution,
                        origin,
                        loadingType
                    ).replace(" ","").replace("\'","").replace("//","/")
            else:

                if self.useInterface:

                    self.workspace_path   = '{}{}_mat_{}_interfaceVal_({:3.2f},{:3.2f})_res={}_origin={}_loading={}_poroThresh={:3.2f}'.format(
                        rootpath,
                        _amitex_foldername,
                        material_tag,
                        *self.material["matrix"]["interfaceVal"],
                        resolution,
                        origin,
                        loadingType,
                        porosityThreshold
                    ).replace(" ","").replace("\'","").replace("//","/")

                else:
                    self.workspace_path   = '{}{}_mat_{}_res={}_origin={}_loading={}_poroThresh={:3.2f}'.format(
                        rootpath,
                        _amitex_foldername,
                        material_tag,
                        resolution,
                        origin,
                        loadingType,
                        porosityThreshold
                    ).replace(" ","").replace("\'","").replace("//","/")


        else: #elasticity

            if self.useInterface:

                self.workspace_path   = '{}{}_mat_{}_interfaceVal_({:3.2f},{:3.2f})_res={}_origin={}_sym={}'.format(
                    rootpath,
                    _amitex_foldername,
                    material_tag,
                    *self.material["matrix"]["interfaceVal"],
                    resolution,
                    origin,
                    materialDict["fiber"]["behavior"]).replace(" ","").replace("\'","").replace("//","/")

            else:
        
                self.workspace_path   = '{}{}_mat_{}_res={}_origin={}_sym={}'.format(
                    rootpath,
                    _amitex_foldername,
                    material_tag,
                    resolution,
                    origin,
                    materialDict["fiber"]["behavior"]).replace(" ","").replace("\'","").replace("//","/")

        self.material_tag=material_tag

        create_folder(self.workspace_path)

        # set default material index as "matrix"
        self.rve_mat_array *= int(self.mat_ID_matrix_int)

        assert type(
            convergence_acceleration) == bool, "openMP_threads must be a bool"
        self.convergence_acceleration = "true" if convergence_acceleration else "false"

        assert type(openMP_threads) == int, "openMP_threads must be an int"
        self.openMP_threads = str(openMP_threads)

    def get_vtk_filename(self):
        return '{}/amitex_mat_{}.vtk'.format(self.get_workspace_path(), self.get_generic_name()),\
            '{}/amitex_zones_{}.vtk'.format(self.get_workspace_path(),
                                            self.get_generic_name())

    def write_vtk_microstructure(self):

        if self.OpenFiberSeg:

            V_fibers_cropped,V_pores_cropped,V_interface,fiberStruct=self.loadVolumes(self.useInterface)

            if "actually_matrix" in self.material["fiber"].keys():
                #this is to homogenize two different matrix descriptions

                V_zones=np.ones(V_fibers_cropped.shape,np.int16)*self.mat_ID_matrix_int

                V_materials=np.ones(V_fibers_cropped.shape,np.int16)*self.mat_ID_matrix_int
                V_materials[V_fibers_cropped==1]=self.mat_ID_fiber_int

                self.rve_mat_array  = V_materials
                self.rve_zone_array = V_zones

                self.n_fiber_zones = 1

            else:

                V_zones, LUT_markerToQuaternion, nZones, print_e1e2 = compactifySubVolume(
                    V_fibers_cropped, fiberStruct, parallelHandle=True)

                if nZones>=32767:
                    raise ValueError(
                        "nZones= {} >=32767 will cause Amitex to crash due to overflow error, this should never happen"\
                        .format(nZones)
                        )

                if len(np.unique(V_fibers_cropped))==1:
                    # no fibers are present
                    # rearrange identifiers: in this scenario (mesoscale, or pure matrix) no fibers are specified. 
                    # since materials must be given in order (1,2,3) AND present in vtk file, assign labels for matrix and pores first:
                    
                    self.mat_ID_matrix_int = 1
                    self.mat_ID_fiber_int = 3
                    self.mat_ID_void_int = 2

                    self.mat_ID_matrix_str = str(self.mat_ID_matrix_int)
                    self.mat_ID_fiber_str  = str(self.mat_ID_fiber_int)
                    self.mat_ID_void_str   = str(self.mat_ID_void_int)

                V_materials = np.ones(V_zones.shape, np.int16) * \
                    self.mat_ID_fiber_int

                V_materials[V_zones == -1         ] = self.mat_ID_matrix_int
                V_materials[V_pores_cropped == 255] = self.mat_ID_void_int

                if self.useInterface:
                    V_interface[V_fibers_cropped >-1]=0 #HACK there is a collision between voxels labelled as fiber and interface. if there is no fiber with zone number 0, amitex crashes

                    V_materials[V_interface == 255] = self.mat_ID_interface_int
                    #HACK
                    V_zones[V_interface == 255] = 1

                V_zones[V_zones == -1] = 1 #the default zone for every material is 1, matrix will have zone 1
                # there is also going to be a zone 1 fiber, but this will be differentiated by 
                # having material value==mat_ID_fiber_int. Amitex demands minimal zone value to be 0 or 1

                # plt.figure(num="V_fibers")

                # plt.imshow(V_fibers[
                #     self.origin_z,
                #     self.origin_x:self.origin_x+self.nx,
                #     self.origin_y:self.origin_y+self.ny
                # ])

                # plt.figure(num='V_zones')

                # plt.imshow(V_zones[0,:,:])

                # plt.figure(num='V_materials')

                # plt.imshow(V_materials[0,:,:])

                # plt.show()

                self.rve_mat_array = V_materials
                self.rve_zone_array = V_zones

                e1 = np.array([1.,  0.,  0. ])
                e2 = np.array([0.,  1.,  0. ])

                for zoneID, quat in LUT_markerToQuaternion.items():

                    e1prime = qt.rotate_vectors(quat, e1)
                    e2prime = qt.rotate_vectors(quat, e2)
                    # e1prime = e1
                    # e2prime = e2

                    if print_e1e2: #cumbersome for numerous fibers
                        print("e1prime={: >8.4f}{: >8.4f}{: >8.4f} e2prime={: >8.4f}{: >8.4f}{: >8.4f}".format(*e1prime,*e2prime))

                    self.LUT_fibers_zones_to_base_vectors[zoneID] = (
                        e1prime, e2prime)

                self.n_fiber_zones = nZones

        else:
            self.voxelize_rve()
        nx, ny, nz = self.get_nx(), self.get_ny(), self.get_nz()

        rve_mat_array  = self.get_rve_mat_array()
        rve_zone_array = self.get_rve_zone_array()

        z,y,x=np.where(rve_mat_array==self.mat_ID_void_int)

        indexErr=np.where(rve_zone_array[z,y,x]!=1)[0]# all voids should have zone 1, this finds erroneous positions

        if len(indexErr>0):
            print("\n\tWarning, there are: {} pixels where fiber and void info are in conflict. Setting those pixels to voids".format(len(indexErr)))

        rve_zone_array[z[indexErr],y[indexErr],x[indexErr]]=1

        print("\t\trve_mat_array.shape  (z,y,x)\t={}".format(self.get_rve_mat_array().shape))
        print("\t\trve_zone_array.shape (z,y,x)\t={}".format(self.get_rve_zone_array().shape))

        uniqueMaterialMarkers=np.unique(rve_mat_array)

        if self.useInterface:

            print("\n\t\tMaterial identifiers present in rve_mat_array: [{}], with matrix={}, fiber={}, pores={}, interface={}".\
                format(uniqueMaterialMarkers,
                    self.mat_ID_matrix_int,
                    self.mat_ID_fiber_int,
                    self.mat_ID_void_int,
                    self.mat_ID_interface_int
                    )
                )

        else:

            print("\n\t\tMaterial identifiers present in rve_mat_array: [{}], with matrix={}, fiber={}, pores={}".\
                format(uniqueMaterialMarkers,
                    self.mat_ID_matrix_int,
                    self.mat_ID_fiber_int,
                    self.mat_ID_void_int
                    )
                )

        print("\t\tHighest zone number in rve_zone_array: {}".format(len(np.unique(rve_zone_array))))

        if self.mat_ID_fiber_int in uniqueMaterialMarkers and self.material["fiber"]["behavior"]=="none":
            raise ValueError("This microstructure contains fibers, but the material description specifies no fiber properties.")

        writer_mat = vtk.vtkStructuredPointsWriter()
        writer_zone = vtk.vtkStructuredPointsWriter()

        rve_mat = vtk.vtkStructuredPoints()
        rve_zone = vtk.vtkStructuredPoints()

        if self.OpenFiberSeg:
            rve_mat.SetDimensions (    
                V_materials.shape[2]+1,
                V_materials.shape[1]+1,
                V_materials.shape[0]+1,
            )
            rve_zone.SetDimensions(            
                V_zones.shape[2]+1,
                V_zones.shape[1]+1,
                V_zones.shape[0]+1,
                )
        else:
            rve_mat .SetDimensions(nz + 1, ny + 1, nx + 1 )
            rve_zone.SetDimensions(nz + 1, ny + 1, nx + 1 )

            rve_mat .SetDimensions(nx + 1, ny + 1, nz + 1 )
            rve_zone.SetDimensions(nx + 1, ny + 1, nz + 1 )

            rve_mat .SetDimensions(ny + 1, nx + 1, nz + 1 )
            rve_zone.SetDimensions(ny + 1, nx + 1, nz + 1 )


        rve_mat.SetOrigin(0, 0, 0)
        rve_zone.SetOrigin(0, 0, 0)

        rve_mat.SetSpacing(1, 1, 1)
        rve_zone.SetSpacing(1, 1, 1)

        scalars_mat = numpy_support.numpy_to_vtk(
            num_array=rve_mat_array.ravel(),
            array_type=vtk.VTK_UNSIGNED_SHORT
        )

        scalars_zone = numpy_support.numpy_to_vtk(
            num_array=rve_zone_array.ravel(),
            array_type=vtk.VTK_UNSIGNED_SHORT
        )

        scalars_mat.SetName('MaterialId')
        scalars_zone.SetName('ZoneId')

        rve_mat.GetCellData().SetScalars(scalars_mat)
        rve_zone.GetCellData().SetScalars(scalars_zone)

        writer_mat.SetFileTypeToBinary()
        writer_zone.SetFileTypeToBinary()

        filename_mat, filename_zones = self.get_vtk_filename()

        writer_mat.SetFileName(filename_mat)
        writer_zone.SetFileName(filename_zones)

        writer_mat.SetInputData(rve_mat)
        writer_zone.SetInputData(rve_zone)

        print("\n\tcalling writer inside write_vtk_microstructure()")

        writer_mat.Write()
        writer_zone.Write()

    def amitex_write_mat(self):

        parent = et.Element('Materials')

        # Matrix properties

        if self.material["matrix"]["behavior"]=="iso":
            E_m, nu_m = self.get_E_m(), self.get_nu_m()
            lambda_m = E_m*nu_m/((1 + nu_m)*(1 - 2*nu_m))
            mu_m = E_m/(2*(1+nu_m))

            # isotropic material law
            matrix_material_child = et.SubElement(parent, 'Material', {
                                            'numM': self.mat_ID_matrix_str, 'Lib': self.mat_lib_path, 'Law': 'elasiso'})
            et.SubElement(matrix_material_child, 'Coeff', {
                        'Index': '1', 'Type': 'Constant', 'Value': '{:.6f}'.format(lambda_m)})
            et.SubElement(matrix_material_child, 'Coeff', {
                        'Index': '2', 'Type': 'Constant', 'Value': '{:.6f}'.format(mu_m)})

        elif self.material["matrix"]["behavior"]=="orthotropic":
            
            # anisotropic (orthotropic) material law
            matrix_material_child = et.SubElement(parent, 'Material', {
                                            'numM': self.mat_ID_matrix_str, 'Lib': self.mat_lib_path, 'Law': 'elasaniso'})
            
            # Notation CAST3M (Voigt, ordre 11 22 33 12 13 23)
            """Amitex and Abaqus tensors are in notation 11 22 33 12 13 23"""

            coeff = {k+1: 0. for k in range(12)}

            coeff[1] = self.C_matrix[0, 0]
            coeff[2] = self.C_matrix[0, 1]
            coeff[3] = self.C_matrix[0, 2]
            coeff[4] = self.C_matrix[1, 1]
            coeff[5] = self.C_matrix[1, 2]
            coeff[6] = self.C_matrix[2, 2]
            coeff[7] = self.C_matrix[3, 3]/2.
            coeff[8] = self.C_matrix[4, 4]/2.
            coeff[9] = self.C_matrix[5, 5]/2.

            # vector e1
            coeff[10] = 1.
            coeff[11] = 0.
            coeff[12] = 0.

            # vector e2
            coeff[13] = 0.
            coeff[14] = 1.
            coeff[15] = 0.

            for key, val in coeff.items():
                et.SubElement(matrix_material_child, 'Coeff', {'Index': str(
                    key), 'Type': 'Constant', 'Value': '{:.6f}'.format(val)})

            alpha, beta, E, nu=extract_isotropic_parameters(self.C_matrix)

            kappa_m=alpha/3
            mu_m   =beta/2
            
            lambda_m,mu_m=convert_kappa_mu_to_lambda_mu( kappa_m,mu_m)

        elif self.material["matrix"]["behavior"] in ["viscoelas_maxwell"]:
            
            # viscoelastic material law
            matrix_material_child = et.SubElement(
                parent, 
                'Material', 
                {
                    'numM': self.mat_ID_matrix_str, 
                    'Lib': self.mat_lib_path, 
                    'Law': self.material["matrix"]["behavior"]
                }
            )
            
            # Notation CAST3M (Voigt, ordre 11 22 33 12 13 23)
            """Amitex and Abaqus tensors are in notation 11 22 33 12 13 23"""

            numChains=len(self.material["matrix"]["chains"])+1

            coeff = {k+1: 0. for k in range(2+numChains*3)}
          
            lambda_0,mu_0=convert_kappa_mu_to_lambda_mu(
                self.material["matrix"]["kappa_0"],
                self.material["matrix"]["mu_0"]
            )

            # coeff[1] = Not Used
            # coeff[2] = Not Used

            coeff[3] = numChains
            coeff[4] = lambda_0
            coeff[5] = mu_0

            for i in range(1,numChains):
                lambda_i,mu_i=convert_kappa_mu_to_lambda_mu(
                    self.material["matrix"]["chains"][i-1][0],
                    self.material["matrix"]["chains"][i-1][1]
                )

                coeff[i*3+3] = lambda_i 
                coeff[i*3+4] = mu_i
                coeff[i*3+5] = self.material["matrix"]["chains"][i-1][2] # tau_i


            for key, val in coeff.items():
                et.SubElement(
                    matrix_material_child, 
                    'Coeff', 
                    {
                        'Index': str(key), 
                        'Type': 'Constant', 
                        'Value': '{:.5e}'.format(val)
                    }
                )

            for num in range(numChains*9):
                #initialize all internal variables to 0. 9 internal variables per parameter chain
                et.SubElement(
                    matrix_material_child, 
                    'IntVar', 
                    {
                        'Index': str(num+1), 
                        'Type': 'Constant', 
                        'Value': '0.'
                    }
                )

            kappa_m=self.material["matrix"]["kappa_0"]
            mu_m   =self.material["matrix"]["mu_0"]
            
            lambda_m,mu_m=convert_kappa_mu_to_lambda_mu( kappa_m,mu_m)

        elif self.material["matrix"]["behavior"]=="UMAT_viscoelastic":
            matrix_material_child = et.SubElement(
                parent, 
                'Material', 
                {
                    'numM': self.mat_ID_matrix_str, 
                    'Lib' : self.mat_lib_path, 
                    'Law' : self.material["matrix"]["subroutine"] # name of Fortran subroutine
                }
            )

            nTau=14
            nTens=6 # number of elements for 3D stress tensors

            for num in range(nTau*nTens):
                #initialize all internal variables to 0. 9 internal variables per parameter chain
                et.SubElement(
                    matrix_material_child, 
                    'IntVar', 
                    {
                        'Index': str(num+1), 
                        'Type': 'Constant', 
                        'Value': '0.'
                    }
                )

            kappa_m=self.material["matrix"]["kappa_0"]
            mu_m   =self.material["matrix"]["mu_0"]
            
            lambda_m,mu_m=convert_kappa_mu_to_lambda_mu( kappa_m,mu_m)

        else:
            raise ValueError(
                "Material symmetry class {} for matrix is not supported".format(self.material["matrix"]["behavior"])
                )

        # Fiber properties

        if self.mat_ID_fiber_int in self.rve_mat_array:
            if self.material["fiber"]["behavior"] == "iso":
                E_f, nu_f = self.get_E_f(), self.get_nu_f()

                lambda_f = E_f*nu_f/((1 + nu_f)*(1 - 2*nu_f))
                mu_f = E_f/(2*(1+nu_f))

                # isotropic material law
                fiber_material_child = et.SubElement(parent, 'Material', {
                                                'numM': self.mat_ID_fiber_str, 'Lib': self.mat_lib_path, 'Law': 'elasiso'})
                et.SubElement(fiber_material_child, 'Coeff', {
                            'Index': '1', 'Type': 'Constant', 'Value': '{:.6f}'.format(lambda_f)})
                et.SubElement(fiber_material_child, 'Coeff', {
                            'Index': '2', 'Type': 'Constant', 'Value': '{:.6f}'.format(mu_f)})

            elif self.material["fiber"]["behavior"] == "trans_iso":

                E_l, nu_l = self.get_E_l(), self.get_nu_l()

                lambda_f = E_l*nu_l/((1 + nu_l)*(1 - 2*nu_l))
                mu_f = E_l/(2*(1+nu_l))

                #   9 coefficients rangés par ligne
                #   1 2 3 0 0 0
                #   2 4 5 0 0 0
                #   3 5 6 0 0 0
                #   0 0 0 7 0 0
                #   0 0 0 0 8 0
                #   0 0 0 0 0 9

                # vecteurs e1 et e2 de la base locale

                # Notation CAST3M (Voigt, ordre 11 22 33 12 13 23)
                """Amitex and Abaqus tensors are in notation 11 22 33 12 13 23"""

                coeff = {k+1: 0. for k in range(12)}

                coeff[1] = self.C[0, 0]
                coeff[2] = self.C[0, 1]
                coeff[3] = self.C[0, 2]
                coeff[4] = self.C[1, 1]
                coeff[5] = self.C[1, 2]
                coeff[6] = self.C[2, 2]
                coeff[7] = self.C[3, 3]/2.
                coeff[8] = self.C[4, 4]/2.
                coeff[9] = self.C[5, 5]/2.

                # vector e1
                coeff[10] = 1.
                coeff[11] = 0.
                coeff[12] = 0.

                # vector e2
                coeff[13] = 0.
                coeff[14] = 1.
                coeff[15] = 0.

                if self.use_zones_in_xml:
                    # anisotropic material law
                    fiber_material_child = et.SubElement(
                        parent, 
                        'Material', 
                        {
                            'numM': self.mat_ID_fiber_str, 
                            'Lib' : self.mat_lib_path, 
                            'Law' : 'elasaniso'
                        }
                    )

                    coeff_per_zone = {val: {zoneNum: {
                    } for zoneNum in self.LUT_fibers_zones_to_base_vectors.keys()} for val in range(10, 16)}

                    for zoneNum, vectorTuple in self.LUT_fibers_zones_to_base_vectors.items():

                        coeff_per_zone[10][zoneNum] = {'numZ': str(
                            zoneNum), 'Value': '{:.6f}'.format(vectorTuple[0][0])}
                        coeff_per_zone[11][zoneNum] = {'numZ': str(
                            zoneNum), 'Value': '{:.6f}'.format(vectorTuple[0][1])}
                        coeff_per_zone[12][zoneNum] = {'numZ': str(
                            zoneNum), 'Value': '{:.6f}'.format(vectorTuple[0][2])}
                        coeff_per_zone[13][zoneNum] = {'numZ': str(
                            zoneNum), 'Value': '{:.6f}'.format(vectorTuple[1][0])}
                        coeff_per_zone[14][zoneNum] = {'numZ': str(
                            zoneNum), 'Value': '{:.6f}'.format(vectorTuple[1][1])}
                        coeff_per_zone[15][zoneNum] = {'numZ': str(
                            zoneNum), 'Value': '{:.6f}'.format(vectorTuple[1][2])}

                    for key, val in coeff.items():
                        if key < 10:
                            et.SubElement(fiber_material_child, 'Coeff', {'Index': str(
                                key), 'Type': 'Constant', 'Value': '{:.6f}'.format(val)})
                        else:
                            coeffElement = et.SubElement(
                                fiber_material_child,
                                'Coeff', 
                                {
                                    'Index': str(key), 
                                    'Type': 'Constant_Zone'
                                }
                            )

                            for entry in coeff_per_zone[key].values():
                                et.SubElement(coeffElement, 'Zone', entry)
                            # et.SubElement(coeffElement, 'Zone', {'numZ': str(2), 'Value': '{:.6f}'.format(44)})

            elif self.material["fiber"]["behavior"] == "orthotropic":
                
                # anisotropic (orthotropic) material law
                fiber_material_child = et.SubElement(
                    parent, 'Material', {
                        'numM'  : self.mat_ID_fiber_str, 
                        'Lib'   : self.mat_lib_path, 
                        'Law'   : 'elasaniso'
                        }
                    )
                
                # Notation CAST3M (Voigt, ordre 11 22 33 12 13 23)
                """Amitex and Abaqus tensors are in notation 11 22 33 12 13 23"""

                coeff = {k+1: 0. for k in range(12)}

                coeff[1] = self.C_fiber[0, 0]
                coeff[2] = self.C_fiber[0, 1]
                coeff[3] = self.C_fiber[0, 2]
                coeff[4] = self.C_fiber[1, 1]
                coeff[5] = self.C_fiber[1, 2]
                coeff[6] = self.C_fiber[2, 2]
                coeff[7] = self.C_fiber[3, 3]/2.
                coeff[8] = self.C_fiber[4, 4]/2.
                coeff[9] = self.C_fiber[5, 5]/2.

                # vector e1
                coeff[10] = 1.
                coeff[11] = 0.
                coeff[12] = 0.

                # vector e2
                coeff[13] = 0.
                coeff[14] = 1.
                coeff[15] = 0.

                for key, val in coeff.items():
                    et.SubElement(
                        fiber_material_child, 
                        'Coeff', {
                            'Index': str(key), 
                            'Type': 'Constant', 
                            'Value': '{:.6f}'.format(val)
                            }
                        )
                
                alpha, beta, E, nu=extract_isotropic_parameters(modifyConvention(self.C_fiber))

                kappa_f=alpha/3
                mu_f   =beta/2
                
                lambda_f,mu_f=convert_kappa_mu_to_lambda_mu( kappa_f,mu_f)

            elif self.material["fiber"]["behavior"] in ["viscoelas_maxwell","viscoelas_maxwell_mod"]:
                # viscoelastic material law
                fiber_material_child = et.SubElement(
                    parent, 
                    'Material', 
                    {
                        'numM': self.mat_ID_fiber_str, 
                        'Lib' : self.mat_lib_path, 
                        'Law' : self.material["fiber"]["behavior"]
                    }
                )
                
                # Notation CAST3M (Voigt, ordre 11 22 33 12 13 23)
                """Amitex and Abaqus tensors are in notation 11 22 33 12 13 23"""

                numChains=len(self.material["fiber"]["chains"])+1

                coeff = {k+1: 0. for k in range(2+numChains*3)}
            
                lambda_0,mu_0=convert_kappa_mu_to_lambda_mu(
                    self.material["fiber"]["kappa_0"],
                    self.material["fiber"]["mu_0"]
                )

                # coeff[1] = Not Used
                # coeff[2] = Not Used

                coeff[3] = numChains
                coeff[4] = lambda_0
                coeff[5] = mu_0

                for i in range(1,numChains):
                    lambda_i,mu_i=convert_kappa_mu_to_lambda_mu(
                        self.material["fiber"]["chains"][i-1][0],
                        self.material["fiber"]["chains"][i-1][1]
                    )

                    coeff[i*3+3] = lambda_i 
                    coeff[i*3+4] = mu_i
                    coeff[i*3+5] = self.material["fiber"]["chains"][i-1][2] # tau_i


                for key, val in coeff.items():
                    et.SubElement(
                        fiber_material_child, 
                        'Coeff', 
                        {
                            'Index': str(key), 
                            'Type': 'Constant', 
                            'Value': '{:.5e}'.format(val)
                        }
                    )

                for num in range(numChains*9):
                    #initialize all internal variables to 0. 9 internal variables per parameter chain
                    et.SubElement(
                        fiber_material_child, 
                        'IntVar', 
                        {
                            'Index': str(num+1), 
                            'Type': 'Constant', 
                            'Value': '0.'
                        }
                    )
            elif self.material["fiber"]["behavior"]=="UMAT_viscoelastic":

                if self.mat_ID_fiber_int in self.rve_mat_array:
                    # xml should only speif
                    fiber_material_child = et.SubElement(
                        parent, 
                        'Material', 
                        {
                            'numM': self.mat_ID_fiber_str, 
                            'Lib' : self.mat_lib_path, 
                            'Law' : self.material["fiber"]["subroutine"] # name of Fortran subroutine
                        }
                    )

                    nTau=14 # defined by previous optimization
                    nTens=6 # number of elements for 3D stress tensors

                    for num in range(nTau*nTens):
                        #initialize all internal variables to 0. 9 internal variables per parameter chain
                        et.SubElement(
                            fiber_material_child, 
                            'IntVar', 
                            {
                                'Index': str(num+1), 
                                'Type': 'Constant', 
                                'Value': '0.'
                            }
                        )

            elif self.material["fiber"]["behavior"]=="none":
                # just to enable the computation of reference material without doing sub conditions
                lambda_f=lambda_m
                mu_f=mu_m

        else:
            # no fibers present in RVE
            lambda_f=lambda_m
            mu_f=mu_m

        # Porosity data
        if self.mat_ID_void_int in self.get_rve_mat_array():

            void_mat_child = et.SubElement(parent, 'Material', {'numM': str(
                self.mat_ID_void_str), 'Lib': self.mat_lib_path, 'Law': 'elasiso'})
            et.SubElement(void_mat_child, 'Coeff', {
                        'Index': '1', 'Type': 'Constant', 'Value': '0.0'})
            et.SubElement(void_mat_child, 'Coeff', {
                        'Index': '2', 'Type': 'Constant', 'Value': '0.0'})

            if self.mat_ID_fiber_int in self.get_rve_mat_array():
                if self.material["matrix"]["behavior"]=="iso":
                    lambda_ref = max(lambda_m, lambda_f)/2
                    mu_ref = max(mu_m, mu_f)/2

                elif self.material["matrix"]["behavior"] in ["viscoelas_maxwell","UMAT_viscoelastic"]:
                    kappa_m=self.material["matrix"]["kappa_0"]
                    mu_m  =self.material["matrix"]["mu_0"]

                    lambda_m,mu_m=convert_kappa_mu_to_lambda_mu( kappa_m,mu_m)

                    lambda_ref = max(lambda_m, lambda_f)/2
                    mu_ref = max(mu_m, mu_f)/2

                else:
                    raise ValueError("Reference material not implemented for matrix behavior type: {}".format(self.material["matrix"]["behavior"]))
            else:
                # this case has voids but no fibers
                lambda_ref  = lambda_m
                mu_ref      = mu_m

        else:
            if self.material["matrix"]["behavior"]=="iso":
                lambda_ref = (min(lambda_m, lambda_f) + max(lambda_m, lambda_f))/2
                mu_ref = (min(mu_m, mu_f) + max(mu_m, mu_f))/2

            elif self.material["matrix"]["behavior"] in ["orthotropic","viscoelas_maxwell","UMAT_viscoelastic"]:
                lambda_ref = (min(lambda_m, lambda_f) + max(lambda_m, lambda_f))/2
                mu_ref = (min(mu_m, mu_f) + max(mu_m, mu_f))/2


        if self.useInterface:
            if self.material["matrix"]["behavior"] not in ["viscoelas_maxwell"]:
                raise ValueError('Only implemented for viscoelastic_maxwell behavior law')
            
            # viscoelastic material law
            interface_material_child = et.SubElement(
                parent, 
                'Material', 
                {
                    'numM': self.mat_ID_interface_str, 
                    'Lib': self.mat_lib_path, 
                    'Law': self.material["matrix"]["behavior"]
                }
            )
            
            # Notation CAST3M (Voigt, ordre 11 22 33 12 13 23)
            """Amitex and Abaqus tensors are in notation 11 22 33 12 13 23"""

            numChains=len(self.material["matrix"]["chains"])+1

            coeff = {k+1: 0. for k in range(2+numChains*3)}
          
            interfaceVal=self.material["matrix"]["interfaceVal"]

            lambda_interface,mu_interface=convert_kappa_mu_to_lambda_mu(
                self.material["matrix"]["kappa_0"]*interfaceVal[0],
                self.material["matrix"]["mu_0"]*interfaceVal[1]
            )

            # coeff[1] = Not Used
            # coeff[2] = Not Used

            coeff[3] = numChains
            coeff[4] = lambda_interface
            coeff[5] = mu_interface

            for i in range(1,numChains):
                lambda_i,mu_i=convert_kappa_mu_to_lambda_mu(
                    self.material["matrix"]["chains"][i-1][0],
                    self.material["matrix"]["chains"][i-1][1]
                )

                coeff[i*3+3] = lambda_i 
                coeff[i*3+4] = mu_i
                coeff[i*3+5] = self.material["matrix"]["chains"][i-1][2] # tau_i


            for key, val in coeff.items():
                et.SubElement(
                    interface_material_child, 
                    'Coeff', 
                    {
                        'Index': str(key), 
                        'Type': 'Constant', 
                        'Value': '{:.5e}'.format(val)
                    }
                )

            for num in range(numChains*9):
                #initialize all internal variables to 0. 9 internal variables per parameter chain
                et.SubElement(
                    interface_material_child, 
                    'IntVar', 
                    {
                        'Index': str(num+1), 
                        'Type': 'Constant', 
                        'Value': '0.'
                    }
                )


        et.SubElement(parent, 'Reference_Material', {
                      'Lambda0': '{:.6f}'.format(lambda_ref), 'Mu0': '{:.6f}'.format(mu_ref)})

        f = open('{}/amitex_mat_{}.xml'.format(self.get_workspace_path(),
                                               self.get_generic_name()), 'w')
        f.write(prettify(parent))
        f.close()

    def write_char(self):
        parent = et.Element('Loading_Output')

        output_child = et.SubElement(parent, 'Output')
        
        et.SubElement(
            output_child, 'vtk_StressStrain',
            {'Strain': '1', 'Stress': '1'}
        )

        if self.material["matrix"]["behavior"] in ["viscoelas_maxwell","UMAT_viscoelastic","viscoelas_maxwell_mod"]:

            assert self.material["matrix"]["units"]==self.material["loadingSpecs"]["units"],(
                "Time or Stress units do not coincide between material law and loading.\nUnits for matrix: {}\nUnits for loading: {}".\
                    format(self.material["matrix"]["units"],self.material["loadingSpecs"]["units"])
            )

            if self.material["fiber"]["behavior"] != 'none':
                assert self.material["matrix"]["units"]["stress"]==self.material["fiber"]["units"],(
                    "Stress units do not coincide between matrix and fiber material laws.\nUnits for matrix: {}\nUnits for fiber: {}".\
                        format(self.material["matrix"]["units"]["stress"],self.material["fiber"]["units"])
                )

            if self.material["loadingSpecs"]["loadingType"]=="Creep recovery":

                loadingSpecs=self.material["loadingSpecs"]

                StressMax   ="{:.4f}".format(loadingSpecs["StressMax"])
                t1          ="{:.4f}".format(loadingSpecs["t1"])
                t2          ="{:.4f}".format(loadingSpecs["t2"])
                t3          ="{:.4f}".format(loadingSpecs["t3"])
                tEnd        ="{:.4f}".format(loadingSpecs["tEnd"])

                if loadingSpecs["nIter_t1"]==0 or\
                    loadingSpecs["nIter_t2"]==0:
                        raise ValueError("nIter_t1 (loading increments) and nIter_t2 (creep increments) must be >0")

                nIter_t1    ="{:.0f}".format(loadingSpecs["nIter_t1"])
                nIter_t2    ="{:.0f}".format(loadingSpecs["nIter_t2"])
                nIter_t3    ="{:.0f}".format(loadingSpecs["nIter_t3"])
                nIter_tEnd  ="{:.0f}".format(loadingSpecs["nIter_tEnd"])

                Temperature ="{:.4f}".format(loadingSpecs["Temperature"])

                direction=loadingSpecs["direction"]

                self.loadingType="creep"

                InitLoadExt_child = et.SubElement(parent, 'InitLoadExt')
                et.SubElement(InitLoadExt_child, 'T',{'Value': Temperature})

                # apply load

                loading_child = et.SubElement(
                    parent, 'Loading', {'Tag': '1'}
                )
                et.SubElement(loading_child, 'Time_Discretization', {
                    'Discretization': 'Linear', 'Nincr': nIter_t1, 'Tfinal': t1})

                for loadingDirection in ['xx','yy','zz','xy','xz','yz']:

                    et.SubElement(
                        loading_child, loadingDirection, {
                            'Driving': 'Stress', 
                            'Evolution': 'Linear', 
                            'Value': StressMax if direction == loadingDirection else '0'
                        }
                    )

                et.SubElement(loading_child, 'T', {'Evolution': 'Constant'})

                #creep phase, constant stress

                loading_child = et.SubElement(
                        parent, 'Loading', {'Tag': '2'}
                    )
                et.SubElement(loading_child, 'Time_Discretization', {
                    'Discretization': 'Linear', 'Nincr': nIter_t2, 'Tfinal': t2})

                for loadingDirection in ['xx','yy','zz','xy','xz','yz']:

                    et.SubElement(
                        loading_child, loadingDirection, {
                                'Driving'   : 'Stress', 
                                'Evolution' : 'Constant', 
                                'Value'     : StressMax  if direction == loadingDirection else '0'
                            }
                        )

                et.SubElement(loading_child, 'T', {'Evolution': 'Constant'})

                if int(nIter_t3) >0: #no recovery otherwise

                    #release load

                    loading_child = et.SubElement(
                        parent, 'Loading', {'Tag': '3'}
                    )
                    et.SubElement(loading_child, 'Time_Discretization', {
                        'Discretization': 'Linear', 'Nincr': nIter_t3, 'Tfinal': t3})

                    for loadingDirection in ['xx','yy','zz','xy','xz','yz']:

                        et.SubElement(
                            loading_child, loadingDirection, {
                                'Driving'   : 'Stress', 
                                'Evolution' : 'Linear' if direction == loadingDirection else 'Constant', 
                                'Value'     : '0.'
                            }
                        )
                    
                    et.SubElement(loading_child, 'T', {'Evolution': 'Constant'})

                    if int(nIter_tEnd) >0: #no recovery otherwise

                        # recovery

                        loading_child = et.SubElement(
                            parent, 'Loading', {'Tag': '4'}
                        )
                        et.SubElement(loading_child, 'Time_Discretization', {
                            'Discretization': 'Linear', 'Nincr': nIter_tEnd, 'Tfinal': tEnd}) 

                        for loadingDirection in ['xx','yy','zz','xy','xz','yz']:

                            et.SubElement(
                                loading_child, loadingDirection, {
                                    'Driving'   : 'Stress', 
                                    'Evolution' : 'Constant', 
                                    'Value'     : '0.'
                                }
                            )

                        et.SubElement(loading_child, 'T', {'Evolution': 'Constant'})


            elif self.material["loadingSpecs"]["loadingType"]=="Relaxation":
                
                loadingSpecs=self.material["loadingSpecs"]

                StrainVal   =loadingSpecs["StrainVal"]
                t1          =loadingSpecs["t1"]
                t2          =loadingSpecs["t2"]

                nIter_t1    ="{:.0f}".format(loadingSpecs["nIter_t1"])
                nIter_t2    ="{:.0f}".format(loadingSpecs["nIter_t2"])

                if loadingSpecs["nIter_t1"]==0 or\
                    loadingSpecs["nIter_t2"]==0:
                        raise ValueError("nIter ==0 will cause a crash")

                Temperature ="{:.4f}".format(loadingSpecs["Temperature"])

                self.loadingType="relaxation"

                for i in range(6):
                    # create one loading xml file for each of the six load cases. 
                    # this is much faster and more stable than
                    # trying to run all 6 loading cases sequentially

                    parent = et.Element('Loading_Output')

                    output_child = et.SubElement(parent, 'Output')
                    et.SubElement(output_child, 'vtk_StressStrain',
                                {'Strain': '1', 'Stress': '1'})

                    InitLoadExt_child = et.SubElement(parent, 'InitLoadExt')
                    et.SubElement(InitLoadExt_child, 'T',{'Value': Temperature})

                    # strain application

                    strain_components = [str(float((component == i)*StrainVal)) for component in range(6)]

                    loadDirections={
                        0:'xx',
                        1:'yy',
                        2:'zz',
                        3:'xy',
                        4:'xz',
                        5:'yz',
                    }

                    loading_child = et.SubElement(parent, 'Loading', {'Tag': '1'})

                    et.SubElement(loading_child, 'Time_Discretization', {
                        'Discretization': 'Linear', 
                        'Nincr': nIter_t1, 
                        'Tfinal': '{}'.format(t1)
                        }
                    ) 


                    for iDir in range(6):
                        et.SubElement(loading_child, loadDirections[iDir], {
                            'Driving': 'Strain', 
                            'Evolution': 'Linear', 
                            'Value': strain_components[iDir]
                            }
                        )

                    et.SubElement(loading_child, 'T', {'Evolution': 'Constant'})

                    # relaxation, static strain

                    loading_child = et.SubElement(parent, 'Loading', {'Tag': '2'})

                    et.SubElement(loading_child, 'Time_Discretization', {
                        'Discretization': 'Linear', 
                        'Nincr': nIter_t2, 
                        'Tfinal': '{}'.format(t1+t2)
                        }
                    ) 

                    for iDir in range(6):
                        et.SubElement(loading_child, loadDirections[iDir], {
                            'Driving': 'Strain', 
                            'Evolution': 'Constant', 
                            'Value': strain_components[iDir]
                            }
                        )

                    et.SubElement(loading_child, 'T', {'Evolution': 'Constant'})

                    f = open('{}/amitex_char_{}_{}.xml'.format(
                        self.get_workspace_path(),
                        self.get_generic_name(),
                        loadDirections[i]
                        ), 'w')
                    f.write(prettify(parent))
                    f.close()

                return

            if self.material["loadingSpecs"]["loadingType"]=="StressTimeSeries":

                self.loadingType="StressTimeSeries"

                loadingSpecs=self.material["loadingSpecs"]

                TimeStressPairs     =loadingSpecs["TimeStressPairs"]
                timeIntervals       =loadingSpecs["timeIntervals"]

                Temperature ="{:.4f}".format(loadingSpecs["Temperature"])

                InitLoadExt_child = et.SubElement(parent, 'InitLoadExt')
                et.SubElement(InitLoadExt_child, 'T',{'Value': Temperature})

                direction=loadingSpecs["direction"]

                previousTime=0

                for i,(t,sigma) in enumerate(TimeStressPairs):

                    tagNum=i+1

                    increments=max((t-previousTime)//timeIntervals,2) #have at minimum 2 increments

                    previousTime=t

                    # apply load

                    loading_child = et.SubElement(
                        parent, 'Loading', {'Tag': str(tagNum)}
                    )
                    et.SubElement(
                        loading_child, 
                        'Time_Discretization', {
                            'Discretization': 'Linear', 
                            'Nincr': "{:.0f}".format(increments), 
                            'Tfinal': "{:6.4f}".format(t)
                        }
                    )


                    for loadingDirection in ['xx','yy','zz','xy','xz','yz']:

                        et.SubElement(
                            loading_child, loadingDirection, {
                                'Driving'   : 'Stress', 
                                'Evolution' : 'Linear' if direction == loadingDirection else 'Constant', 
                                'Value'     : "{:6.4f}".format(sigma) if direction == loadingDirection else '0'
                            }
                        )

                    et.SubElement(loading_child, 'T', {'Evolution': 'Constant'})

        else:
         
            self.loadingType="elasticity"
            
            for i in range(6):
                strain_components = [str(float(component == i))
                                    for component in range(6)]

                loading_child = et.SubElement(
                    parent, 'Loading', {'Tag': '{}'.format(i+1)})
                et.SubElement(loading_child, 'Time_Discretization', {
                            'Discretization': 'Linear', 'Nincr': '1', 'Tfinal': '{}'.format(i+1)})
                            
                if self.makeVTK_stress_strain:
                    vtk_tag = et.SubElement(loading_child, 'Output_vtkList')
                    vtk_tag.text = '1'

                et.SubElement(loading_child, 'xx', {
                            'Driving': 'Strain', 'Evolution': 'Linear', 'Value': strain_components[0]})
                et.SubElement(loading_child, 'yy', {
                            'Driving': 'Strain', 'Evolution': 'Linear', 'Value': strain_components[1]})
                et.SubElement(loading_child, 'zz', {
                            'Driving': 'Strain', 'Evolution': 'Linear', 'Value': strain_components[2]})
                et.SubElement(loading_child, 'xy', {
                            'Driving': 'Strain', 'Evolution': 'Linear', 'Value': strain_components[3]})
                et.SubElement(loading_child, 'xz', {
                            'Driving': 'Strain', 'Evolution': 'Linear', 'Value': strain_components[4]})
                et.SubElement(loading_child, 'yz', {
                            'Driving': 'Strain', 'Evolution': 'Linear', 'Value': strain_components[5]})

        f = open('{}/amitex_char_{}.xml'.format(self.get_workspace_path(),
                                                self.get_generic_name()), 'w')
        f.write(prettify(parent))
        f.close()

    def write_algo(self, algorithm_type='Basic_Scheme', convergence_criterion='Default', filter_type='Default', hpp='true',nIterMax=nIterMaxAmitex):
        parent = et.Element('Algorithm_Parameters')

        algo_child = et.SubElement(parent, 'Algorithm', {
                                   'Type': algorithm_type})
        et.SubElement(algo_child, 'Convergence_Criterion',
                      {'Value': convergence_criterion})
        et.SubElement(algo_child, 'Convergence_Acceleration', {
                      'Value': self.convergence_acceleration})
        et.SubElement(algo_child, 'Nitermax', {
                      'Value': nIterMax})


        mecha_child = et.SubElement(parent, 'Mechanics')
        et.SubElement(mecha_child, 'Filter', {'Type': filter_type})
        et.SubElement(mecha_child, 'Small_Perturbations', {'Value': hpp})

        f = open('{}/amitex_algo_{}.xml'.format(self.get_workspace_path(),
                                                self.get_generic_name()), 'w')
        f.write(prettify(parent))
        f.close()

    def preprocessing(self):
        print("\n\tcalling write_vtk_microstructure()...\n")
        self.write_vtk_microstructure()

        print("\n\tcalling write_algo()...\n")
        self.write_algo()

        print("\n\tcalling write_char()...\n")
        self.write_char()

        print("\n\tcalling amitex_write_mat()...\n")
        self.amitex_write_mat()

    def processing(self):
        gen_name = self.get_generic_name()
        work_path = self.get_workspace_path()

        print(gen_name, work_path)

        currentDirectory=os.getcwd()
        os.chdir(work_path)

        if self.loadingType=="relaxation":
            loadDirections={
                0:'xx',
                1:'yy',
                2:'zz',
                3:'xy',
                4:'xz',
                5:'yz',
            }

            for iEval in range(6):

                full_cmd=[
                    'mpirun',
                    '-np',
                    self.openMP_threads,
                    'amitex_fftp',
                    '-nm',
                    'amitex_mat_{}.vtk'.format(gen_name),
                    '-nz',
                    'amitex_zones_{}.vtk'.format(gen_name),
                    '-m',
                    'amitex_mat_{}.xml'.format(gen_name),
                    '-c',
                    'amitex_char_{}_{}.xml'.format(gen_name,loadDirections[iEval]),
                    '-a',
                    'amitex_algo_{}.xml'.format(gen_name),
                    '-s',
                    '{}_{}'.format( gen_name,loadDirections[iEval])
                ]

                print("executing command: \n",full_cmd)

                systemCall=subprocess.run(full_cmd,stderr=subprocess.PIPE)


                if systemCall.stderr.decode("utf-8"):
                    # Amitex process returned error signal
                    print("\n\nAmitex process returned error signal:")
                    print(systemCall.stderr.decode("utf-8"))

                    return False,systemCall.stderr.decode("utf-8")
            
            os.chdir(currentDirectory)

            return True,""

        else:
            full_cmd=[
                'mpirun',
                '-np',
                self.openMP_threads,
                'amitex_fftp',
                '-nm',
                'amitex_mat_{}.vtk'.format(gen_name),
                '-nz',
                'amitex_zones_{}.vtk'.format(gen_name),
                '-m',
                'amitex_mat_{}.xml'.format(gen_name),
                '-c',
                'amitex_char_{}.xml'.format(gen_name),
                '-a',
                'amitex_algo_{}.xml'.format(gen_name),
                '-s',
                '{}'.format( gen_name)
            ]

            print("executing command: \n",full_cmd)

            systemCall=subprocess.run(full_cmd,stderr=subprocess.PIPE)

            os.chdir(currentDirectory)

            if systemCall.stderr.decode("utf-8"):
                # Amitex process returned error signal
                print("\n\nAmitex process returned error signal:")
                print(systemCall.stderr.decode("utf-8"))

                return False,systemCall.stderr.decode("utf-8")
            
            return True,""

    def postprocessing(self):
        gen_name = self.get_generic_name()
        work_path = self.get_workspace_path()

        if self.loadingType=="elasticity":

            with open(r'{}/{}.std'.format(work_path, gen_name), 'r') as f:
                amitex_results = f.read().split('\n')

            voigt_weights = modified_voigt_weights()

            amitex_results = amitex_results[7:]

            self.C_hom = np.zeros((6, 6))

            for i in range(6):
                step_results = amitex_results[i].split()
                for j in range(6):
                    self.C_hom[i, j] = voigt_weights[i, j]*float(step_results[j+1])

            return write_to_file_Voigt4(
                '{}/{}_C_hom.txt'.format(work_path, gen_name), 
                self.C_hom,
                material=self.material,
                material_tag=self.material_tag,
                loadingType=self.loadingType,
                prec_total=12,
                prec_decimal=8
            )

        if self.loadingType in ["creep","StressTimeSeries"]:

            with open(r'{}/{}.std'.format(work_path, gen_name), 'r') as f:
                amitex_results = f.read().split('\n')

            voigt_weights = np.ones((6,6),np.float64)

            amitex_results = amitex_results[6:]

            self.Time=np.zeros(len(amitex_results)-1)
            self.Stress = np.zeros((len(amitex_results)-1,6))
            self.Strain = np.zeros((len(amitex_results)-1,6))

            if self.material["loadingSpecs"]["loadingType"] =="Relaxation":

                loadingSpecs=self.material["loadingSpecs"]

                nIter_t1    =loadingSpecs["nIter_t1"]
                nIter_t2    =loadingSpecs["nIter_t2"]
                nIter_t3    =loadingSpecs["nIter_t3"]

                nIterPerLoadCase=nIter_t1+nIter_t2+nIter_t3

                # each load case gives us a column of C as a function of time (relaxation test)
                # only the stress values during constant strain (load case #2 are kept)
                self.C_hom=np.zeros((nIter_t2,6,6))

            iTimeStep_C=0
            iCol=0

            changeTimeToMinutes=False

            for iTimeStep in range(len(amitex_results)-1):
                step_results = amitex_results[iTimeStep].split()
                self.Time[iTimeStep]=float(step_results[0])
                for i in range(6):
                    matchobj=re.match(r"(-?[0-9].[0-9]+E)(\+|\-)([0-9]+)",step_results[i+1])
                    if matchobj is None:
                        matchobj=re.match(r"(-?[0-9].[0-9]+)(\+|\-)([0-9]+)",step_results[i+1])
                        
                        step_results[i+1]=matchobj.group(1)+'E'+matchobj.group(2)+matchobj.group(3)
    
                    self.Stress[iTimeStep, i] = voigt_weights[0, i]*float(step_results[i+1])
                for i in range(6,12):
                    self.Strain[iTimeStep, i-6] = voigt_weights[0, i-6]*float(step_results[i+1])


            if self.material["loadingSpecs"]["loadingType"] in ["Creep recovery","StressTimeSeries"]:
                self.C_hom=(self.Stress,self.Strain)


            labelsStress=[
                "stress_xx",
                "stress_yy",
                "stress_zz",
                "stress_xy",
                "stress_xz",
                "stress_yz",
            ]

            labelsStrain=[
                "strain_xx",
                "strain_yy",
                "strain_zz",
                "strain_xy",
                "strain_xz",
                "strain_yz",
            ]

            markersList =["v","s",">","o","x","^"]
            sizeList    =[100,40 ,25 ,10 ,6  ,4  ]

            plt.rcParams.update({'font.size': 16})
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams["font.family"] = "Times New Roman"

            for i in range(6):

                plt.figure(figsize=[10,8],num="Stress vs Time")

                plt.plot(self.Time,self.Stress[:,i], label=labelsStress[i])
                plt.scatter(self.Time,self.Stress[:,i], marker=markersList[i],s=sizeList[i])

                if changeTimeToMinutes:
                    # so plot labels match actual changed units
                    timeUnits="minutes"
                else:
                    timeUnits=self.material["loadingSpecs"]["units"]["time"]

                plt.xlabel("Time ({})".format(timeUnits))
                plt.ylabel(r"$\sigma$ "+"({})".format(self.material["loadingSpecs"]["units"]["stress"]))

                plt.legend()

                plt.savefig(os.path.join(self.workspace_path,'StressVsTime.png'))

                plt.figure(figsize=[10,8],num="Strain vs Time")

                plt.plot(self.Time,self.Strain[:,i], label=labelsStrain[i])
                plt.scatter(self.Time,self.Strain[:,i], marker=markersList[i],s=sizeList[i])

                plt.xlabel("Time ({})".format(timeUnits))
                plt.ylabel(r"$\epsilon$ (m/m)")

                plt.legend()

                plt.savefig(os.path.join(self.workspace_path,'StrainVsTime.png'))

            if showPlotsViscoelasticity:
                plt.show()
            else:
                plt.close('all')

            return write_to_file_Voigt4(
                '{}/{}_C_hom.txt'.format(work_path, gen_name), 
                self.C_hom,
                material=self.material,
                material_tag=self.material_tag,
                loadingType=self.loadingType,
                time=self.Time,
                prec_total=12,
                prec_decimal=8,
                units=self.material["loadingSpecs"]["units"]
            )

        if self.loadingType in ["relaxation"]:

            with open(r'{}/{}_xx.std'.format(work_path, gen_name), 'r') as f:
                amitex_results = f.read().split('\n')

            voigt_weights = np.ones((6,6),np.float64)

            amitex_results = amitex_results[6:]

            self.Time=np.zeros(len(amitex_results)-1)
            self.TimeAll=np.zeros(len(amitex_results)*6-1)
            self.Stress     = np.zeros((len(amitex_results)-1   ,6))
            self.StressAll  = np.zeros((len(amitex_results)*6-1 ,6))
            self.Strain     = np.zeros((len(amitex_results)-1   ,6))
            self.StrainAll  = np.zeros((len(amitex_results)*6-1 ,6))


            loadingSpecs=self.material["loadingSpecs"]

            nIter_t1    =loadingSpecs["nIter_t1"]
            nIter_t2    =loadingSpecs["nIter_t2"]

            nIterPerLoadCase=nIter_t1+nIter_t2

            # each load case gives us a column of C as a function of time (relaxation test)
            # only the stress values during constant strain (load case #2 are kept)
            self.C_hom=np.zeros((nIter_t2,6,6))

            iTimeStep_C=0
            iTimeStepAll=0
            iCol=0

            timeOffset=0

            changeTimeToMinutes=False

            for iDir in range(6):

                loadDirections={
                    0:'xx',
                    1:'yy',
                    2:'zz',
                    3:'xy',
                    4:'xz',
                    5:'yz',
                }

                with open(r'{}/{}_{}.std'.format(work_path, gen_name,loadDirections[iDir]), 'r') as f:
                    amitex_results = f.read().split('\n')
                    amitex_results = amitex_results[6:]

                for iTimeStep in range(len(amitex_results)-1):
                    step_results = amitex_results[iTimeStep].split()
                    self.Time   [iTimeStep]     =float(step_results[0])
                    self.TimeAll[iTimeStepAll]  =float(step_results[0])+timeOffset
                    for i in range(6):
                        matchobj=re.match(r"(-?[0-9].[0-9]+E)(\+|\-)([0-9]+)",step_results[i+1])
                        if matchobj is None:
                            matchobj=re.match(r"(-?[0-9].[0-9]+)(\+|\-)([0-9]+)",step_results[i+1])
                            
                            step_results[i+1]=matchobj.group(1)+'E'+matchobj.group(2)+matchobj.group(3)
        
                        self.Stress[iTimeStep, i] = voigt_weights[0, i]*float(step_results[i+1])
                        self.StressAll[iTimeStepAll, i] = voigt_weights[0, i]*float(step_results[i+1])
                    for i in range(6,12):
                        self.Strain[iTimeStep, i-6] = voigt_weights[0, i-6]*float(step_results[i+1])
                        self.StrainAll[iTimeStepAll, i-6] = voigt_weights[0, i-6]*float(step_results[i+1])
                        
                    iTimeTruncated=iTimeStep%(nIterPerLoadCase+1)

                    # only the stress data during the relaxation phase are kept
                    if nIter_t1<=iTimeTruncated<nIter_t1+nIter_t2:

                        self.C_hom[iTimeStep_C,:,iCol]=self.Stress[iTimeStep]
                        iTimeStep_C+=1
                        iTimeStepAll+=1

                    if iTimeTruncated==nIterPerLoadCase:
                        iTimeStep_C=0
                        iCol+=1

                timeOffset+=self.Time[-1]

            if self.material["loadingSpecs"]["units"]["time"]=="seconds":
                #output needs to be in minutes for Anton's method
                self.Time[iTimeStep]/=60
                self.TimeAll[iTimeStepAll]/=60
                changeTimeToMinutes=True

            labelsStress=[
                "stress_xx",
                "stress_yy",
                "stress_zz",
                "stress_xy",
                "stress_xz",
                "stress_yz",
            ]

            labelsStrain=[
                "strain_xx",
                "strain_yy",
                "strain_zz",
                "strain_xy",
                "strain_xz",
                "strain_yz",
            ]

            markersList =["v","s",">","o","x","^"]
            sizeList    =[100,40 ,25 ,10 ,6  ,4  ]

            plt.rcParams.update({'font.size': 16})
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams["font.family"] = "Times New Roman"

            for i in range(6):

                plt.figure(figsize=[10,8],num="Stress vs Time")

                plt.plot(self.TimeAll,self.StressAll[:,i], label=labelsStress[i])
                plt.scatter(self.TimeAll,self.StressAll[:,i], marker=markersList[i],s=sizeList[i])

                if changeTimeToMinutes:
                    # so plot labels match actual changed units
                    timeUnits="minutes"
                else:
                    timeUnits=self.material["loadingSpecs"]["units"]["time"]

                plt.xlabel("Time ({})".format(timeUnits))
                plt.ylabel(r"$\sigma$ "+"({})".format(self.material["loadingSpecs"]["units"]["stress"]))

                plt.legend()

                plt.savefig(os.path.join(self.workspace_path,'StressVsTime.png'))

                plt.figure(figsize=[10,8],num="Strain vs Time")

                plt.plot(self.TimeAll,self.StrainAll[:,i], label=labelsStrain[i])
                plt.scatter(self.TimeAll,self.StrainAll[:,i], marker=markersList[i],s=sizeList[i])

                plt.xlabel("Time ({})".format(timeUnits))
                plt.ylabel(r"$\epsilon$ (m/m)")

                plt.legend()

                plt.savefig(os.path.join(self.workspace_path,'StrainVsTime.png'))

            if showPlotsViscoelasticity:
                plt.show()
            else:
                plt.close('all')

            return write_to_file_Voigt4(
                '{}/{}_C_hom.txt'.format(work_path, gen_name), 
                self.C_hom,
                material=self.material,
                material_tag=self.material_tag,
                loadingType=self.loadingType,
                time=self.Time,
                prec_total=12,
                prec_decimal=8,
                units=self.material["loadingSpecs"]["units"]
            )

    def get_workspace_path(self):
        return self.workspace_path

    def get_rve_mat_array(self):
        return self.rve_mat_array

    def get_rve_zone_array(self):
        return self.rve_zone_array

    def set_rve_arrays(self, xx, yy, kz, id, mat):
        # id is used in the Craft version

        if mat == 'fiber':
            self.rve_mat_array[kz, yy, xx] = self.mat_ID_fiber_int
            self.rve_zone_array[kz, yy, xx] = self.n_fiber_zones
        elif mat == 'void':
            self.rve_mat_array[kz, yy, xx] = self.mat_ID_void_int
            self.rve_zone_array[kz, yy, xx] = self.n_porosity_zones


class AbaqusInterface(CFRPHInterface):

    def gmsh_setSurfacePeriodicInterpolation(self, vec, surfaces, unicity_list, mesh_order):
        slave_name = '{}p'.format(vec)
        master_name = '{}m'.format(vec)
        phi_list = shape_functions(mesh_order)
        proj = np.array([int(x == vec) for x in ['x', 'y', 'z']])
        true_list = 2*[True]
        eps = 1e-5

        slave_nodes_tags, slave_nodes_coords = gmsh.model.mesh.getNodesForPhysicalGroup(
            2, surfaces[slave_name]['physical_tag'])

        n_slave_nodes = slave_nodes_tags.shape[0]
        slave_nodes = [[]]
        master_nodes = [[], []]

        for i in range(n_slave_nodes):
            slave_node_coord = slave_nodes_coords[3*i:3*(i+1)]
            surf_slave_coord = [slave_node_coord[i]
                                for i in range(3) if not(proj[i])]

            if list(map(lambda x: eps < x < 1-eps, surf_slave_coord)) == true_list:
                slave_nodes[0].append(slave_nodes_tags[i])
                fictive_master_node_coord = slave_node_coord - proj

                _, _, master_nodes_tags, xi, eta, _ = gmsh.model.mesh.getElementByCoordinates(
                    *fictive_master_node_coord, dim=2, strict=False)

                weights = [-phi(xi, eta) for phi in phi_list]

                for j in range(master_nodes_tags.shape[0]):
                    master_nodes[0].append(master_nodes_tags[j])
                    master_nodes[1].append(weights[j])

                # print(zeta)
                # print(master_nodes[i])
                # print('----------------------------------------------------------')

        slave_nodes = np.array(slave_nodes).transpose()
        master_nodes = np.array(master_nodes).transpose()

        # print(slave_nodes)
        # print('----------------------------------------------------------')
        # print(master_nodes)
        # print('----------------------------------------------------------')

        self.set_node_set('inner_surface', slave_name, slave_nodes)
        self.set_node_set('inner_surface', master_name, master_nodes)

        print(slave_name)

    def gmsh_setEdgePeriodicInterpolation(self, vec, edges, unicity_list):
        axis = {'x': 0, 'y': 1, 'z': 2}
        edges_names = list(edges.keys())[4*axis[vec]:4*axis[vec]+4]
        master_name = edges_names[0]
        edges_nodes = dict.fromkeys(edges_names)
        phi_1, phi_2, _ = shape_functions('se2')
        eps = 1e-5

        master_nodes_tags, master_nodes_coords = gmsh.model.mesh.getNodesForPhysicalGroup(
            1, edges[master_name]['physical_tag'])
        master_nodes = np.array([master_nodes_tags]).transpose()
        self.set_node_set('inner_edge', master_name, master_nodes)
        print('MASTER {}'.format(master_name))
        print(master_nodes)
        print('--------------------------------------------------------------')

        for i in range(0, len(edges_names), 4):
            const_coords = [edges_names[i][3], edges_names[i][0]]

            for j in range(i+3, i, -1):
                slave_nodes_tags, slave_nodes_coords = gmsh.model.mesh.getNodesForPhysicalGroup(
                    1, edges[edges_names[j]]['physical_tag'])

                n_slave_nodes = slave_nodes_tags.shape[0]
                slave_nodes = [[], [], []]

                dummy_node_axis = const_coords[j % 2]
                proj = np.array([int(x == dummy_node_axis)
                                 for x in ['x', 'y', 'z']])
                orientation = 1 if edges_names[j][4] == 'm' else -1

                for k in range(n_slave_nodes):
                    slave_node_coord = slave_nodes_coords[3*k:3*(k+1)]

                    if eps < slave_node_coord[axis[vec]] < 1-eps:
                        slave_nodes[0].append(slave_nodes_tags[k])
                        fictive_master_node_coord = slave_node_coord + orientation*proj

                        _, _, master_nodes_tags, xi, _, _ = gmsh.model.mesh.getElementByCoordinates(
                            *fictive_master_node_coord, dim=1, strict=True)

                        # print(master_nodes_tags)
                        # print(xi, eta, zeta)
                        # print(gmsh.model.mesh.getElementProperties(element_type))
                        # print('----------------------------------------------')

                        weights = [phi_1(xi), phi_2(xi)]

                        slave_nodes[1].append(
                            (master_nodes_tags[0], master_nodes_tags[1]))
                        slave_nodes[2].append((weights[0], weights[1]))

                slave_nodes = np.array(slave_nodes).transpose()

                print('SLAVE {}'.format(edges_names[j]))
                print(slave_nodes)
                print('--------------------------------------------------------------')

                self.set_node_set('inner_edge', edges_names[j], slave_nodes)

    def gmsh_getSurfacePeriodicNodes(self, vec, surfaces, unicity_list):
        slave_name = '{}p'.format(vec)
        master_name = '{}m'.format(vec)
        slave_nodes = [[], [], [], []]
        master_nodes = [[], [], [], []]

        for slave_tag in surfaces[slave_name]['entity_tags']:
            master_tag, slave_node_tags, master_node_tags, _ = gmsh.model.mesh.getPeriodicNodes(
                2, slave_tag)
            if len(slave_node_tags) == len(master_node_tags):
                n_nodes = len(slave_node_tags)

                for i in range(n_nodes):
                    slave_node_tag = slave_node_tags[i]
                    master_node_tag = master_node_tags[i]
                    if not(unicity_list[slave_node_tag]) and not(unicity_list[master_node_tag]):

                        if surface_node_check(vec, slave_node_tag, master_node_tag):
                            slave_nodes[0].append(slave_node_tag)
                            master_nodes[0].append(master_node_tag)

                            slave_node_coord, _ = gmsh.model.mesh.getNode(
                                slave_node_tag)
                            master_node_coord, _ = gmsh.model.mesh.getNode(
                                master_node_tag)

                            for j in range(3):
                                slave_nodes[j+1].append(slave_node_coord[j])
                                master_nodes[j+1].append(master_node_coord[j])

                            unicity_list[slave_node_tag] += 1
                            unicity_list[master_node_tag] += 1

            else:
                error_msg = (
                    'The number of nodes contained in the slave surface {} is not'
                    'the same as in the master surface {}.'
                ).format(slave_tag, master_tag)
                print(error_msg)

        slave_nodes = np.array(slave_nodes).transpose()
        master_nodes = np.array(master_nodes).transpose()

        self.set_node_set('inner_surface', slave_name, slave_nodes)
        self.set_node_set('inner_surface', master_name, master_nodes)

        return unicity_list

    def gmsh_getEdgePeriodicNodes(self, vec, edges, unicity_list):
        axis = {'x': 0, 'y': 1, 'z': 2}
        edges_names = list(edges.keys())[4*axis[vec]:4*axis[vec]+4]
        master_name = edges_names[0]
        edges_nodes = dict.fromkeys(edges_names)

        for edge_name in edges_names:
            temp_node_datas = [[], [], [], []]

            for slave_tag in edges[edge_name]['entity_tags']:
                slave_node_tags, node_coord, _ = gmsh.model.mesh.getNodes(
                    1, slave_tag, includeBoundary=True)

                for i in range(len(slave_node_tags)):
                    if not(unicity_list[slave_node_tags[i]]) and edge_node_check(axis[vec], node_coord[3*i:3*(i+1)]):
                        temp_node_datas[0].append(slave_node_tags[i])
                        for j in range(3):
                            temp_node_datas[j+1].append(node_coord[3*i+j])

                        unicity_list[slave_node_tags[i]] += 1

            temp_node_datas = np.array(temp_node_datas).transpose()
            temp_node_datas = temp_node_datas[temp_node_datas[:,
                                                              axis[vec]+1].argsort()]

            edges_nodes[edge_name] = temp_node_datas

        master_nodes = edges_nodes[master_name]

        for edge_name in edges_names:
            if edge_periodicity_check(axis[vec], edges_nodes[edge_name], master_nodes):
                self.set_node_set('inner_edge', edge_name,
                                  edges_nodes[edge_name])
            else:
                error_msg = (
                    'The number of nodes contained in the slave edge {} is not '
                    'the same as in the master edge {}.'
                ).format(edge_name, master_name)
                print(error_msg)

        return unicity_list

    def gmsh_getVertexPeriodicNodes(self, vertices, unicity_list):
        vertices_names = list(vertices.keys())

        for vertex_name in vertices_names:

            for vertex_tag in vertices[vertex_name]['entity_tags']:
                node_tags, node_coord, _ = gmsh.model.mesh.getNodes(
                    0, vertex_tag, returnParametricCoord=False)

                for node_tag in node_tags:
                    if not(unicity_list[node_tag]):
                        self.set_node_set('vertex', vertex_name, np.array(
                            [[node_tag, *node_coord]]))
                        unicity_list[node_tag] += 1

        return unicity_list

    def gmsh_getInnerVolumeNodes(self, unicity_list):
        node_tags, coord, _ = gmsh.model.mesh.getNodes(
            returnParametricCoord=False)

        x_target_coord = np.array([0.75, 0.5, 0.75])
        y_target_coord = np.array([0.5, 0.5, 0.25])
        z_target_coord = np.array([0.25, 0.5, 0.75])

        x_node_tag, y_node_tag, z_node_tag = 1, 1, 1

        first_node_coord = coord[0:3]

        x_dist_min = np.linalg.norm(first_node_coord - x_target_coord)
        y_dist_min = np.linalg.norm(first_node_coord - y_target_coord)
        z_dist_min = np.linalg.norm(first_node_coord - z_target_coord)

        n = len(node_tags)
        n_inner_nodes = 0

        for i in range(n):
            node_tag = node_tags[i]

            if not(unicity_list[node_tag]):
                node_coord = coord[3*i:3*(i+1)]

                x_dist = np.linalg.norm(node_coord-x_target_coord)
                y_dist = np.linalg.norm(node_coord-y_target_coord)
                z_dist = np.linalg.norm(node_coord-z_target_coord)

                if x_dist < x_dist_min:
                    x_dist_min = x_dist
                    x_node_tag = node_tag

                if y_dist < y_dist_min:
                    y_dist_min = y_dist
                    y_node_tag = node_tag

                if z_dist < z_dist_min:
                    z_dist_min = z_dist
                    z_node_tag = node_tag

                n_inner_nodes += 1

        node_tags = np.array([[x_node_tag], [y_node_tag], [z_node_tag]])

        # print('--------------------------------------------------------------')
        # print(x_node_tag)
        # print(gmsh.model.mesh.getNode(x_node_tag)[0])
        # print('--------------------------------------------------------------')
        # print(y_node_tag)
        # print(gmsh.model.mesh.getNode(y_node_tag)[0])
        # print('--------------------------------------------------------------')
        # print(z_node_tag)
        # print(gmsh.model.mesh.getNode(z_node_tag)[0])

        self.set_node_set('inner_volume', get_volume_key(), node_tags)

        return n_inner_nodes

    def write_mesh(self, periodic_mesh_bool, mesh_order):
        gmsh.initialize()
        gmsh.option.setNumber('General.Terminal', 1)
        gmsh.model.add('RVE')

        fibers_dimtags = []
        porosities_dimtags = []
        fiber_phases = {}
        characteristic_length = 1

        for ellipsoid_id in range(1, self.get_n_ellipsoids()+1):
            ellipsoid_features = self.get_ellipsoid(ellipsoid_id)
            mat, rx, ry, rz, xc, yc, zc, quat, _ = ellipsoid_features.values()
            inclusion_tag = gmsh.model.occ.addSphere(0, 0, 0, 1)
            gmsh.model.occ.dilate([(3, inclusion_tag)], 0, 0, 0, rx, ry, rz)
            gmsh.model.occ.affineTransform(
                [(3, inclusion_tag)],
                gmsh_affineTransform([xc, yc, zc], qt.as_rotation_matrix(quat))
            )

            min_radius = min(rx, ry, rz)
            if min_radius < characteristic_length:
                characteristic_length = min_radius

            if mat == 'fiber':
                fibers_dimtags.append((3, inclusion_tag))
                # if self.get_symmetry_class() == 'trans_iso':
                max_radius = max(rx, ry, rz)
                vector_l = np.array(
                    list(map(lambda r: r == max_radius, [rx, ry, rz])))
                fiber_phases[inclusion_tag] = {
                    'fiber_tags': [],
                    'orientation': np.dot(qt.as_rotation_matrix(quat), vector_l)
                }
            elif mat == 'void':
                porosities_dimtags.append((3, inclusion_tag))

        periodic_vectors = [x for x in itertools.product(
            [-1, 0, 1], repeat=3) if x != (0, 0, 0)]

        boxes_dimtags = []
        for v in periodic_vectors:
            box_tag = gmsh.model.occ.addBox(
                *list(map(lambda x: -x, v)), 1, 1, 1)
            boxes_dimtags.append([(3, box_tag)])

        intersected_fibers_dimtags = []
        intersected_porosities_dimtags = []

        for i in range(len(periodic_vectors)):

            for fiber_dimtag in fibers_dimtags:
                intersection = gmsh.model.occ.intersect(
                    boxes_dimtags[i], [fiber_dimtag], removeObject=False, removeTool=False)

                if len(intersection[0]) > 0:
                    gmsh.model.occ.affineTransform(
                        intersection[0], gmsh_affineTransform(periodic_vectors[i]))

                    for dimtag in intersection[0]:
                        intersected_fibers_dimtags.append(dimtag)
                        fiber_phases[fiber_dimtag[1]
                                     ]['fiber_tags'].append(dimtag[1])

            for porosity_dimtag in porosities_dimtags:
                intersection = gmsh.model.occ.intersect(
                    boxes_dimtags[i], [porosity_dimtag], removeObject=False, removeTool=False)

                if len(intersection[0]) > 0:
                    gmsh.model.occ.affineTransform(
                        intersection[0], gmsh_affineTransform(periodic_vectors[i]))

                    for dimtag in intersection[0]:
                        intersected_porosities_dimtags.append(dimtag)

        gmsh.model.occ.remove(boxes_dimtags, recursive=True)

        ver_box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        all_porosities_dimtags = porosities_dimtags + intersected_porosities_dimtags
        if all_porosities_dimtags:
            cut = gmsh.model.occ.cut([(3, ver_box)], all_porosities_dimtags)
            ver_box_dimtag = cut[0]
        else:
            ver_box_dimtag = [(3, ver_box)]

        for fiber_dimtag in fibers_dimtags:
            intersection = gmsh.model.occ.intersect(
                ver_box_dimtag, [fiber_dimtag], removeObject=False)
            if len(intersection[0]) > 0:
                for dimtag in intersection[0]:
                    intersected_fibers_dimtags.append(dimtag)
                    fiber_phases[fiber_dimtag[1]
                                 ]['fiber_tags'].append(dimtag[1])

        all_phases_dimtags = gmsh.model.occ.fragment(
            [(3, ver_box)], intersected_fibers_dimtags)

        gmsh.model.occ.synchronize()

        all_phases_tags = gmsh_getEntitiesTags(all_phases_dimtags[0])
        fiber_tags = gmsh_getEntitiesTags(intersected_fibers_dimtags)
        matrix_tags = gmsh_getMatrixTags(fiber_tags, all_phases_tags)

        fiber_phases_keys = list(
            map(lambda tag: 'fiber_{}'.format(tag), fiber_phases.keys()))

        if self.get_symmetry_class() == 'iso':
            phase_keys = ['matrix', 'fiber']
        elif self.get_symmetry_class() == 'trans_iso':
            phase_keys = ['matrix', *fiber_phases_keys]

        surface_keys = get_surface_keys()
        edge_keys = get_edge_keys()
        vertex_keys = get_vertex_keys()

        default_value = {'entity_tags': None, 'physical_tag': None}

        phase_values = [default_value.copy() for i in range(len(phase_keys))]
        surface_values = [default_value.copy()
                          for i in range(len(surface_keys))]
        edge_values = [default_value.copy() for i in range(len(edge_keys))]
        vertex_values = [default_value.copy() for i in range(len(vertex_keys))]

        phases = dict(zip(phase_keys, phase_values))
        surfaces = dict(zip(surface_keys, surface_values))
        edges = dict(zip(edge_keys, edge_values))
        vertices = dict(zip(vertex_keys, vertex_values))

        phases['matrix']['entity_tags'] = matrix_tags
        physical_tag = gmsh.model.addPhysicalGroup(3, matrix_tags)
        phases['matrix']['physical_tag'] = physical_tag
        gmsh.model.setPhysicalName(3, physical_tag, 'matrix')

        if self.get_symmetry_class() == 'iso':
            phases['fiber']['entity_tags'] = fiber_tags
            physical_tag = gmsh.model.addPhysicalGroup(3, fiber_tags)
            phases['fiber']['physical_tag'] = physical_tag
            gmsh.model.setPhysicalName(3, physical_tag, 'fiber')

        elif self.get_symmetry_class() == 'trans_iso':

            for fiber_phase, phase_datas in fiber_phases.items():
                physical_name = 'fiber_{}'.format(fiber_phase)
                physical_tag = gmsh.model.addPhysicalGroup(
                    3, phase_datas['fiber_tags'])

                phases[physical_name]['entity_tags'] = phase_datas['fiber_tags']
                phases[physical_name]['physical_tag'] = physical_tag

                gmsh.model.setPhysicalName(3, physical_tag, physical_name)
                phases[physical_name]['orientation'] = phase_datas['orientation']

        print(phases)
        self.phases = phases

        for physical_name, physical_datas in surfaces.items():
            physical_datas['entity_tags'] = gmsh_getBoundarySurfaceTags(
                physical_name)
            physical_datas['physical_tag'] = gmsh.model.addPhysicalGroup(
                2, physical_datas['entity_tags'])
            gmsh.model.setPhysicalName(
                2, physical_datas['physical_tag'], physical_name)

        for physical_name, physical_datas in edges.items():
            physical_datas['entity_tags'] = gmsh_getBoundaryEdgeTags(
                physical_name)
            physical_datas['physical_tag'] = gmsh.model.addPhysicalGroup(
                1, physical_datas['entity_tags'])
            gmsh.model.setPhysicalName(
                1, physical_datas['physical_tag'], physical_name)

        for physical_name, physical_datas in vertices.items():
            physical_datas['entity_tags'] = gmsh_getBoundaryVertexTags(
                physical_name)
            physical_datas['physical_tag'] = gmsh.model.addPhysicalGroup(
                0, physical_datas['entity_tags'])
            gmsh.model.setPhysicalName(
                0, physical_datas['physical_tag'], physical_name)

        if periodic_mesh_bool:
            # gmsh_setPeriodicity('x', surfaces)
            # gmsh_setPeriodicity('y', surfaces)
            # gmsh_setPeriodicity('z', surfaces)
            setPeriodic(0)
            setPeriodic(1)
            setPeriodic(2)

        print(characteristic_length)

        # gmsh.fltk.run()

        gmsh.option.setNumber(
            'Mesh.CharacteristicLengthMin', characteristic_length/2)
        gmsh.option.setNumber(
            'Mesh.CharacteristicLengthMax', characteristic_length/2)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize('Netgen')

        if mesh_order == 2:
            gmsh.model.mesh.setOrder(2)
            gmsh.model.mesh.optimize('HighOrder')

        gmsh.write(
            '/home/clemvella/Documents/CFRP_homogenization_TEST/gmsh_before_p.msh')

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        n_nodes = node_tags.shape[0]
        self.set_n_nodes(n_nodes)
        unicity_list = (n_nodes+1)*[0]

        if periodic_mesh_bool:
            unicity_list = self.gmsh_getSurfacePeriodicNodes(
                'x', surfaces, unicity_list)
            unicity_list = self.gmsh_getSurfacePeriodicNodes(
                'y', surfaces, unicity_list)
            unicity_list = self.gmsh_getSurfacePeriodicNodes(
                'z', surfaces, unicity_list)

            unicity_list = self.gmsh_getEdgePeriodicNodes(
                'x', edges, unicity_list)
            unicity_list = self.gmsh_getEdgePeriodicNodes(
                'y', edges, unicity_list)
            unicity_list = self.gmsh_getEdgePeriodicNodes(
                'z', edges, unicity_list)

            unicity_list = self.gmsh_getVertexPeriodicNodes(
                vertices, unicity_list)

            n_inner_nodes = self.gmsh_getInnerVolumeNodes(unicity_list)

            n_retrieved_nodes = sum(unicity_list)

            if n_nodes - n_inner_nodes == n_retrieved_nodes:
                success_msg = (
                    'All {} boundary nodes have been retrieved successfully.'
                ).format(n_retrieved_nodes)
                print(success_msg)
            else:
                error_msg = (
                    'Not all {} boundary nodes have been retrieved. '
                    'Number of boundray nodes retrieved: {}.'
                ).format(n_nodes - n_inner_nodes, n_retrieved_nodes)
                print(error_msg)

        else:
            self.gmsh_setSurfacePeriodicInterpolation(
                'x', surfaces, unicity_list, mesh_order)
            self.gmsh_setSurfacePeriodicInterpolation(
                'y', surfaces, unicity_list, mesh_order)
            self.gmsh_setSurfacePeriodicInterpolation(
                'z', surfaces, unicity_list, mesh_order)

            # self.gmsh_setEdgePeriodicInterpolation('x', edges, unicity_list)
            # self.gmsh_setEdgePeriodicInterpolation('y', edges, unicity_list)
            # self.gmsh_setEdgePeriodicInterpolation('z', edges, unicity_list)
            self.gmsh_getEdgePeriodicNodes('x', edges, unicity_list)
            self.gmsh_getEdgePeriodicNodes('y', edges, unicity_list)
            self.gmsh_getEdgePeriodicNodes('z', edges, unicity_list)

            # for edge_name in get_edge_keys():
            #     self.set_node_set('inner_edge', edge_name, [])

            self.gmsh_getVertexPeriodicNodes(vertices, unicity_list)

            self.gmsh_getInnerVolumeNodes(unicity_list)

        gmsh.write(
            '/home/clemvella/Documents/CFRP_homogenization_TEST/gmsh_after_p.msh')

        for physical_name, physical_datas in surfaces.items():
            gmsh.model.removePhysicalGroups(
                [(2, physical_datas['physical_tag'])])
            gmsh.model.removePhysicalName(physical_name)

        for physical_name, physical_datas in edges.items():
            gmsh.model.removePhysicalGroups(
                [(1, physical_datas['physical_tag'])])
            gmsh.model.removePhysicalName(physical_name)

        for physical_name, physical_datas in vertices.items():
            gmsh.model.removePhysicalGroups(
                [(0, physical_datas['physical_tag'])])
            gmsh.model.removePhysicalName(physical_name)

        gmsh.option.setNumber('Mesh.SaveGroupsOfNodes', 1)
        gmsh.write(self.get_gmsh_inp_filename())

        gmsh.finalize()

    def write_fem(self, periodic_mesh_bool, mesh_order):
        microstructure_name = self.get_microstructure_name()
        inp_filename = self.get_gmsh_inp_filename()
        n_nodes = self.get_n_nodes()
        node_sets = self.node_sets
        symmetry_class = self.get_symmetry_class()
        mat_features = self.material
        phases = self.phases

        abaqus_part_cmd_ = abaqus_part_cmds(
            n_nodes, inp_filename, symmetry_class, phases)

        components = ['11', '22', '33', '12', '13', '23']

        for component in components:
            strain_operator = get_strain_operator(component)
            inp_filename = self.get_abaqus_inp_filename(component)

            with open(inp_filename, 'w') as f:
                f.write(abaqus_heading_cmd(microstructure_name))
                f.write(abaqus_part_cmd_)
                f.write(abaqus_assembly_cmd(n_nodes, node_sets, strain_operator,
                                            microstructure_name, periodic_mesh_bool, mesh_order))
                f.write(abaqus_material_cmd(mat_features))
                f.write(abaqus_step_cmd('my_step', n_nodes,
                                        node_sets, strain_operator))

    def write_homogenized_stiffness_matrix(self):
        filename = self.get_C_hom_filename()

        print('Writing C_hom file ...')
        C_hom = np.loadtxt(filename)

        print("\n Homogenized tensor: \n")
        printVoigt4(C_hom)
        write_to_file_Voigt4(filename, C_hom)

    def preprocessing(self, periodic_mesh_bool=True, mesh_order=1):
        self.write_mesh(periodic_mesh_bool, mesh_order)
        self.write_fem(periodic_mesh_bool, mesh_order)

    def processing(self):
        rootpath = self.get_rootpath()
        microstructure_name = self.get_microstructure_name()
        components = ['11', '22', '33', '12', '13', '23']

        for component in components:
            full_cmd = [
                'abaqus',
                '-job',
                '{}_{}'.format(microstructure_name, component),
                '-interactive'
            ]
            try:  # temporary solution because Abaqus gets stuck
                subprocess.run(full_cmd, cwd=rootpath, timeout=500)
            except:
                pass

    def postprocessing(self):
        rootpath = self.get_rootpath()
        microstructure_name = self.get_microstructure_name()

        voigt_weights = modified_voigt_weights()
        voigt_weights = list(map(str, voigt_weights[[0, 5, 5], [0, 0, 5]]))

        porosity_volume = str(self.get_reached_porosity_volume_fraction())

        full_cmd = [
            'abaqus',
            '-cae',
            '-noGUI',
            'c_hom_computation.py',
            '--',
            '{}/{}'.format(rootpath, microstructure_name),
            *voigt_weights,
            porosity_volume
        ]
        subprocess.run(
            full_cmd, cwd='/home/clemvella/Documents/git_repository/CFRP-homogen/matlab_to_craft')

        self.write_homogenized_stiffness_matrix()

    def set_rve_arrays(self):
        pass

    def get_generic_path(self):
        return '{}/{}'.format(self.get_rootpath(), self.get_microstructure_name())

    def get_nodes_filename(self, name):
        return '{}/{}.txt'.format(self.get_rootpath(), name)

    def get_n_nodes(self):
        return self.n_nodes

    def get_gmsh_inp_filename(self):
        return '{}/{}.inp'.format(self.get_rootpath(), self.get_microstructure_name())

    def get_abaqus_inp_filename(self, component):
        return '{}/{}_{}.inp'.format(self.get_rootpath(), self.get_microstructure_name(), component)

    def get_C_hom_filename(self):
        return '{}_C_hom.txt'.format(self.get_generic_path())

    def set_n_nodes(self, n):
        self.n_nodes = n

    def set_node_set(self, entity_type, set_name, node_tags):
        self.node_sets[entity_type][set_name] = node_tags


class MoriTanakaInterface(CFRPHInterface):
    def __init__(
        self,
        rootpath,
        microstructure_name,
        material,
        material_tag,
        resolution,
        OpenFiberSeg=False,
        origin=None,
        convergence_acceleration=True,
        openMP_threads=1
    ):
        super().__init__(rootpath, microstructure_name, material,
                         resolution, OpenFiberSeg=OpenFiberSeg, origin=origin)

        # paramètre intégration fonction de Green
        Pzeta   =32 #64 #combien de point de Gauss en z
        Qomega  =32 #256#combien de points de Gauss en theta

        self.zeta_csv  = import_hmgnzt_quad(pathMT_library, "zeta3_{}points.csv".format(Pzeta) )
        self.omega_csv = import_hmgnzt_quad(pathMT_library, "omega_{}points.csv".format(Qomega) )

        self.workspace_path   = '{}{}_mat_{}_res={}_origin={}_sym={}'.format(
                rootpath,
                _MoriTanaka_foldername,
                material_tag,
                resolution,
                origin,
                material["fiber"]["behavior"]).replace(" ","").replace("\'","").replace("//","/")

        self.material_tag=material_tag
        create_folder(self.workspace_path)

        if self.material["fiber"]["behavior"]=="iso":
            # no convention modification here
            self.C=generate_isoC_from_E_nu(self.material["fiber"]["young"],self.material["fiber"]["poisson"])

        elif self.material["fiber"]["behavior"]=="trans_iso":
            # no convention modification here
            self.C=generate_trans_isoC_from_E_nu_G(
                        self.get_E_l(),
                        self.get_E_t(),
                        self.get_nu_l(),
                        self.get_nu_t(),
                        self.get_G_l(),
                        self.get_axis()
                        )
        else: #self.material["fiber"]["behavior"]=="none"
            print("not implemented")

        self.C0=generate_isoC_from_E_nu(self.material['matrix']["young"],self.material['matrix']["poisson"])


    def create_inclusions_fromTiffFiles(self):

        if not self.OpenFiberSeg:
            raise RuntimeError("Not implemented for microstructures not from OpenFiberSeg")

        else:

            V_fibers_cropped,V_pores_cropped,V_interface,fiberStruct=self.loadVolumes()

            V_zones, LUT_markerToQuaternion, nZones,AR_LUT,print_e1E2 = compactifySubVolume(
                V_fibers_cropped, fiberStruct, parallelHandle=True,returnAspectRatio=True)

            self.totalVoxelCount=V_zones.shape[0]*V_zones.shape[1]*V_zones.shape[2]
            
            self.MoriTanakaData={}

            self.fiberVoxelCount=0

            for marker in LUT_markerToQuaternion:
                voxelsCurrentMarker=len(np.where(V_zones==marker)[0])

                self.fiberVoxelCount+=voxelsCurrentMarker

                self.MoriTanakaData[marker]={
                    "volumeFraction":voxelsCurrentMarker/self.totalVoxelCount ,
                    "aspectRatio"   :AR_LUT[marker],
                    "R"             :qt.as_rotation_matrix(LUT_markerToQuaternion[marker]),
                    "material"      :"fiber"
                }

            print()


    def createInclusionsFromTxtFile(self):
    
        partnerSet=set([])
        
        self.MoriTanakaData={}

        self.fiberVolumeFraction=0.

        for inclusionNumber,ellDict in self.ellipsoids.items():

            if inclusionNumber not in partnerSet:

                # the total volume is unitary in this description, 
                # and each inclusion's volume is it's volume fraction
                volumeFraction=4./3.*np.pi*ellDict["rx"]*ellDict["ry"]*ellDict["rz"]

                self.fiberVolumeFraction+=volumeFraction

                if ellDict["rx"]!=ellDict["ry"]:
                    raise ValueError("not implemented for ellipsoids with non-circular cross-section")

                if ellDict["mat"] =="fiber":
                    ellDict["C"]=self.C
                else: #voids
                    ellDict["C"]=generate_isoC_from_E_nu(0, 0) #porosity


                aspectRatio=ellDict["rz"]/ellDict["rx"]

                rotationMatrix=qt.as_rotation_matrix(ellDict["quat"])

                self.MoriTanakaData[inclusionNumber]={
                    "volumeFraction":volumeFraction ,
                    "aspectRatio"   :aspectRatio,
                    "R"             :qt.as_rotation_matrix(ellDict["quat"]),
                    "material"      :ellDict["mat"],
                    "C"             :ellDict["C"],
                }

                partnerSet.update(ellDict["partners"])

        print("\n\ttotal inclusions volume fraction: {: >8.4f}".format(self.fiberVolumeFraction))


    def computeMT(self):

        #to ensure dict value ordering doesn't cause issues, keep keys in a list (where order is enforced)
        markerList=list(self.MoriTanakaData.keys())

        # RStatic=np.array(
        #     [[ 0., 0., 0.],
        #      [ 0., 0., 0.],
        #      [ 0., 0., 1.]])

        results = Parallel(n_jobs=num_cores)\
        (delayed(parallelEval)(
            marker,
            len(self.MoriTanakaData),
            0,#iCases
            1,#nCases
            self.MoriTanakaData[marker]["aspectRatio"],
            self.MoriTanakaData[marker]["R"],
            # RStatic,
            self.zeta_csv,
            self.omega_csv,
            self.C0,
            self.MoriTanakaData[marker]["C"]
            ) for marker in markerList)  
        
        T_r={}
        C1_rotated={}

        for index,marker in enumerate(markerList):
            T_r[marker]         =results[index][0]
            C1_rotated[marker]  =results[index][1]

            # print('T_r')    
            # tensF.printVoigt4(T_r[iFam])
            # print("C1_rotated")
            # tensF.printVoigt4(C1_rotated[iFam])


        temp2=(1-self.fiberVolumeFraction)*np.identity(6)


        for marker in markerList:
            temp2+=self.MoriTanakaData[marker]["volumeFraction"]*T_r[marker]
        

        print("Construction tenseur homogénéisé\n")
        C_tilde=self.C0#initialisation du calcul de Mori-Tanaka

        A_rList=[]
        for marker in markerList:
            A_r=np.dot(T_r[marker],np.linalg.inv(temp2))
            
            A_rList.append(A_r)
            #tensF.printVoigt4(A_r)
            C_tilde=C_tilde+self.MoriTanakaData[marker]["volumeFraction"]*np.dot(C1_rotated[marker]-self.C0,A_r)


        self.A_r_mean=np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                valuesList=[A[i,j] for A in A_rList]
                self.A_r_mean[i,j]=np.mean(valuesList)

        for line in C_tilde:
            print("{: >8.3f}\t{: >8.3f}\t{: >8.3f}\t{: >8.3f}\t{: >8.3f}\t{: >8.3f}".format(*line))

        # # # extraire coefficients isotrope transverse
        self.C_hom,alphaMT,betaMT,gammaMT,gamma_pMT,\
            deltaMT,delta_pMT,\
            E_lMT,E_tMT,\
            nu_lMT,nu_tMT,G_lMT = \
                transverse_isotropic_projector(C_tilde,2)

        print("E_lMT={: >8.3f}".format(E_lMT))
        print("E_tMT={: >8.3f}".format(E_tMT))
        print("nu_lMT={: >8.3f}".format(nu_lMT))
        print("nu_lMT={: >8.3f}".format(nu_lMT))
        print("G_lMT={: >8.3f}".format(G_lMT))

        print("Projected C in transverse isotropic space")

        for line in self.C_hom:
            print("{: >8.3f}\t{: >8.3f}\t{: >8.3f}\t{: >8.3f}\t{: >8.3f}\t{: >8.3f}".format(*line))

    def postprocessing(self):
        gen_name = self.get_generic_name()
        work_path = self.workspace_path

        prec_total=8
        prec_decimal=4
        prec="{{: >{}.{}f}}, ".format(prec_total,prec_decimal)
        precString="["+prec*6+"],\n"

        filename='{}/{}_A2mean.txt'.format(work_path, gen_name)

        with open(filename,'w') as f:
            f.write("Resulting localization tensor A2\n\n")

            f.write('\n Usual convention\n')
            for i in range(6):
                f.write(precString.format(*self.A_r_mean[i]))

            f.write("\n\nIsotropic parameters: \n")
            C_iso,alphaIso,betaIso,E,nu = isotropic_projector_Facu(self.A_r_mean)
            outputString="\nalpha\t= \t"+prec+"\nbeta\t= \t"+prec+"\nE\t\t= \t"+prec+"\nnu\t\t= \t"+prec+"\n"
            f.write(outputString.format(alphaIso,betaIso,E,nu))
            
        return write_to_file_Voigt4(
            '{}/{}_C_hom.txt'.format(work_path, gen_name), 
            self.C_hom,
            material=self.material,
            material_tag=self.material_tag,
            modifyConventionBool=False
            )

    def preprocessing(self):
        if self.OpenFiberSeg:
            self.create_inclusions_fromTiffFiles()
        else:
            self.createInclusionsFromTxtFile()

    def processing(self):
        self.computeMT()

    def set_rve_arrays():
        pass



__author__ = "Clément Vella"
__copyright__ = "All rights reserved"
__credits__ = ["Facundo Sosa-Rey, Christophe Geuzaine"]
__license__ = "None"
__version__ = "1.0.1"
__maintainer__ = ""
__email__ = ""
__status__ = "Development"
