# -*- coding: utf-8 -*-

import csv
import re
import xml.etree.ElementTree as et
from xml.dom import minidom
import intervals as it
import numpy as np

from materialProperties import max_vtk_size

import os

def create_folder(_path):
    """Creates Amitex or CRaFT folders
    
    Arguments:
        _path {str} -- Path of the working directory
    """
    if not os.path.isdir(_path):
        os.mkdir(_path)

def raise_exception(msg):
    raise Exception('{}'.format(msg))

def check_material_format(material):

    
    assert 'fiber' in material, (
        'The \'fiber\' key is missing in the \'material\' argument.'
    )
    if material['fiber']['behavior'] not in ['UMAT_viscoelastic','none']:
        assert 'matrix' in material, (
            'The \'matrix\' key is missing in the \'material\' argument.'
        )
        assert 'mat_lib_path' in material, (
            'The \'mat_lib_path\' information is missing in the \'material\' argument.'
        )

        assert material["matrix"]["behavior"] in ["iso","orthotropic","viscoelas_maxwell","UMAT_viscoelastic"],(
            "Not implemented for matrix of symmetry {}".format(material["matrix"]["behavior"])
        )

        if material["matrix"]["behavior"]=="iso":
            assert 'young' in material['matrix'], (
                'The \'young\' key is missing in the matrix properties.'
            )
            assert 'poisson' in material['matrix'], (
                'The \'poisson\' key is missing in the matrix properties.'
            )
            assert len(material['matrix']) == 3, (
                'The matrix material contains too many properties, expected 3 keys, {} were given.'.format(len(material['matrix']))
            )
        elif material["matrix"]["behavior"]=="orthotropic":
            assert 'C' in material['matrix'], (
                'The \'C\' tensor is missing in the matrix properties.'
            )
            assert material['matrix']['C'].shape == (6,6), (
                'The C tensor is the wrong shape, expected (6,6), instead got {}'.format(material['matrix']['C'].shape)
            )
        elif material["matrix"]["behavior"]=="viscoelas_maxwell":
            assert 'kappa_0' in material['matrix'], (
                'The \'kappa_0\' key is missing in the matrix properties.'
            )
            assert 'mu_0' in material['matrix'], (
                'The \'mu_0\' key is missing in the matrix properties.'
            )        
            assert 'chains' in material['matrix'], (
                'The \'chains\' key is missing in the matrix properties.'
            )
            for chain in material['matrix']['chains']:
                assert len(chain) == 3, (
                    'One of the chains in matrix properties is the wrong length, expected 3, instead got {}'.format(len(chain))
                )
        

        assert 'behavior' in material['fiber'], (
            'The \'behavior\' key is missing in the fiber properties.'
        )

        assert "mat_lib_path" in material.keys(), (        
            'The path to material library is missing.'
        )

        if material['fiber']['behavior'] == 'iso':
            assert 'young' in material['fiber'], (
                'The \'young\' key is missing in the fiber properties.'
            )
            assert 'poisson' in material['fiber'], (
                'The \'poisson\' key is missing in the fiber properties.'
            )
            assert len(material['fiber']) == 3, (
                'The fiber material contains too many properties, expected 3 keys, {} were given.'.format(len(material['fiber']))
            )
        
        elif material['fiber']['behavior'] == 'trans_iso':
            assert 'young_l' in material['fiber'], (
                'The \'young_l\' key is missing in the fiber properties.'
            )
            assert 'young_t' in material['fiber'], (
                'The \'young_t\' key is missing in the fiber properties.'
            )
            assert 'poisson_l' in material['fiber'], (
                'The \'poisson_l\' key is missing in the fiber properties.'
            )
            assert 'poisson_t' in material['fiber'], (
                'The \'poisson_t\' key is missing in the fiber properties.'
            )
            assert 'shear_l' in material['fiber'], (
                'The \'shear_l\' key is missing in the fiber properties.'
            )
            assert 'axis' in material['fiber'], (
                'The \'axis\' key is missing in the fiber properties.'
            )
            assert type(material['fiber']['axis']) == int, (
                'The \'axis\' key must contain an integer.\n'
                '0 for the (O, x) axis,\n'
                '1 for the (O, y) axis and\n'
                '2 for the (O, z) axis.'
            )
        
        elif material["fiber"]["behavior"]=="viscoelas_maxwell":
            assert 'kappa_0' in material['fiber'], (
                'The \'kappa_0\' key is missing in the fiber properties.'
            )
            assert 'mu_0' in material['fiber'], (
                'The \'mu_0\' key is missing in the fiber properties.'
            )        
            assert 'chains' in material['fiber'], (
                'The \'chains\' key is missing in the fiber properties.'
            )
            for chain in material['fiber']['chains']:
                assert len(chain) == 3, (
                    'One of the chains in fiber properties is the wrong length, expected 3, instead got {}'.format(len(chain))
                )

def skip_lines(f, skip_counter, skip_offsets):
        """Skips some lines during sequential reading of the file containing
        ellipsoid features
        
        Arguments:
            f {_io.TextIOWrapper} -- File being read
            skip_counter {int} -- Counter selecting the number of lines to be skipped
            skip_offsets {list} -- List containing the number of lines to be skipped
        
        Returns:
            integer -- New counter
        """
        for _ in range(skip_offsets[skip_counter]):
            f.readline()
        
        return skip_counter + 1

def read_input_microstructure(filename):
    """Reads the file (.txt) containing the REV generation datas and the
    the ellipsoids features
    
    Returns:
        list -- REV generation datas and ellispoid features
    """
    skip_counter = 0
    skip_offsets = [25, 9, 4, 6]

    with open(filename, 'r') as f:
        skip_counter = skip_lines(f, skip_counter, skip_offsets)

        line = filtered_readline(f, 'int')
        n_ellipsoids = int(line[0])

        line = filtered_readline(f, 'float')
        first_aspect_ratio = float(line[0])

        line = filtered_readline(f, 'float')
        second_aspect_ratio = float(line[0])

        line = filtered_readline(f, 'float')
        desired_volume_fraction = float(line[0])

        skip_counter = skip_lines(f, skip_counter, skip_offsets)

        line = filtered_readline(f, 'float')
        simulation_time = float(line[0])

        line = filtered_readline(f, 'float')
        reached_fiber_volume_fraction    = float(line[0])
        if len(line)>1:
            reached_porosity_volume_fraction = float(line[1])
        else:
            reached_porosity_volume_fraction=0.


        skip_counter = skip_lines(f, skip_counter, skip_offsets)

        ellipsoids = {}
        line = ellipse_readline(f)

        while line != []:
            ellipse_features = ellipsoid_datas_to_dict(line)
            el_id = int(ellipse_features.pop('id'))
            ellipsoids[el_id] = ellipse_features
            line = ellipse_readline(f)
    
    txt_microstructure_features = {
        'n_ellipsoids'                    : n_ellipsoids,
        'first_aspect_ratio'              : first_aspect_ratio,
        'second_aspect_ratio'             : second_aspect_ratio,
        'desired_volume_fraction'         : desired_volume_fraction,
        'simulation_time'                 : simulation_time,
        'reached_fiber_volume_fraction'   : reached_fiber_volume_fraction,
        'reached_porosity_volume_fraction': reached_porosity_volume_fraction,
        'ellipsoids'                      : ellipsoids
    }

    return txt_microstructure_features

def filtered_readline(f, data_type):
    """Filters the characters of the read line
    
    Arguments:
        f {_io.TextIOWrapper} -- File being read
        data_type {str} -- Type of data to be filtered
    
    Returns:
        list -- List of 'data_type' type characters
    """
    filter = {'int': '\d+', 'float': '\d+(?:\.\d*)'}

    return re.findall(filter[data_type], f.readline())

def ellipse_readline(f):
    """Reads a line
    
    Arguments:
        f {_io.TextIOWrapper} -- File being read
    
    Returns:
        list -- List of characters read
    """
    return f.readline().split()

def ellipsoid_datas_to_dict(line):
    """Formats the ellipsoid features into a dictionary
    
    Arguments:ELLIPSOIDS_f20
        line {list} -- List of the ellispoid features
    
    Returns:
        dict -- Dictionary of the ellipsoid features
    """
    ellipsoid_features = {}
    # 1 id + 3 rayons + 3 positions + 4 quaternions = 11 caractéristiques
    n_features = 11
    n = len(line)
    partners = []

    for i in range(n_features, n):
        partners.append(int(line[i]))
    
    ellipsoid_features['id']       = int(line[0])
    ellipsoid_features['mat']      = 'fiber' if float(line[1]) > 0 else 'void'
    ellipsoid_features['rx']       = abs(float(line[1]))
    ellipsoid_features['ry']       = abs(float(line[2]))
    ellipsoid_features['rz']       = abs(float(line[3]))
    ellipsoid_features['xc']       = float(line[4])
    ellipsoid_features['yc']       = float(line[5])
    ellipsoid_features['zc']       = float(line[6])
    ellipsoid_features['quat']     = np.quaternion(*[float(e) for e in line[7:11]])
    ellipsoid_features['partners'] = partners

    return ellipsoid_features

def prettify(elem):
    """Adds indentations and line breaks to make writing XML files more
    easily readable
    
    Arguments:
        elem {xml.etree.ElementTree.Element} -- [description]
    
    Returns:
        [type] -- [description]
    """
    rough_string = et.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)

    return reparsed.toprettyxml(indent='    ')

def check_z_lim(z_interval):
        """Calculating a valid z-interval for filling the REV arrays
        
        Arguments:
            z_interval {intervals.Interval} -- Interval calculated analytically,
            not necessarily included in [0, 1[
        
        Returns:
            [intervals.Interval] -- Intersection between z_interval[type] and [0, 1[
        """
        return z_interval.intersection(it.closedopen(0, 1))

def check_xy_lim(x_res, y_res, x_list, y_list):
        """Checks if the voxels of coordinates (x_list[i], y_list[i]) are in
        the set [0, nx[×[0, ny[
        
        Arguments:
            x_list {numpy.ndarray} -- x coordinates
            y_list {numpy.ndarray} -- y coordinates
        
        Returns:
            numpy.ndarray, numpy.ndarray -- (new_xx[i], new_yy[i]) in [0, nx[×[0, ny[
        """
        new_xx, new_yy = [], []

        for i in range(len(x_list)):
        
            if (0 <= x_list[i] < x_res) and (0 <= y_list[i] < y_res):
                new_xx.append(x_list[i])
                new_yy.append(y_list[i])
        
        return new_xx, new_yy

def gmsh_getEntitiesTags(outDimTags):
    return list(map(lambda x: x[1], outDimTags))

def get_phase_keys():
    return [
        'matrix',
        'fiber'
    ]

def get_volume_key():
    return 'omega_int'

def get_surface_keys():
    return [
        'xm',
        'xp',
        'ym',
        'yp',
        'zm',
        'zp'
    ]

def get_edge_keys():
    return [
        'ym_zm',
        'zm_yp',
        'yp_zp',
        'zp_ym',
        'zm_xm',
        'xm_zp',
        'zp_xp',
        'xp_zm',
        'xm_ym',
        'ym_xp',
        'xp_yp',
        'yp_xm'
    ]

def get_vertex_keys():
    return [
        'xm_ym_zm',
        'xp_ym_zm',
        'xp_yp_zm',
        'xm_yp_zm',
        'xm_yp_zp',
        'xm_ym_zp',
        'xp_ym_zp',
        'xp_yp_zp'
    ]

def gmsh_getBoundarySurfaceTags(physical_name):
    tags             = []
    eps              = 1e-5
    not_vec          = list(map(lambda x: x != physical_name[0], ['x', 'y', 'z']))
    bounding_corners = np.zeros(6)

    if physical_name[1] == 'm':
        vec = [0, 0, 0]
    else:
        vec = np.logical_not(not_vec)

    for i in range(3):
        bounding_corners[i]   = vec[i] - eps
        bounding_corners[i+3] = vec[i] + not_vec[i] + eps

    for dimTag in gmsh.model.getEntitiesInBoundingBox(*bounding_corners, dim=2):
        tags.append(dimTag[1])
    
    return tags

def gmsh_getBoundaryEdgeTags(physical_name):
    tags             = []
    eps              = 1e-5
    directions_s     = ['x', 'y', 'z']
    directions_i     = [0, 1, 2]
    directions_conv  = dict(zip(directions_s, directions_i))
    start_point      = np.zeros(3)
    bounding_corners = np.zeros(6)
    sum_             = 0

    for surface_name in physical_name.split('_'):
        direction_s = surface_name[0]
        direction_i = directions_conv[direction_s]
        boundary    = surface_name[1]

        if boundary == 'p':
            start_point[direction_i] = 1

        sum_ += direction_i
    
    end_point = start_point.copy()
    end_point[3-sum_] = 1

    for i in range(3):
        bounding_corners[i]   = start_point[i] - eps
        bounding_corners[i+3] = end_point[i] + eps

    for dimTag in gmsh.model.getEntitiesInBoundingBox(*bounding_corners, dim=1):
        tags.append(dimTag[1])
    
    return tags

def gmsh_getBoundaryVertexTags(physical_name):
    tags             = []
    eps              = 1e-5
    directions_s     = ['x', 'y', 'z']
    directions_i     = [0, 1, 2]
    directions_conv  = dict(zip(directions_s, directions_i))
    point            = np.zeros(3)
    bounding_corners = np.zeros(6)

    for surface_name in physical_name.split('_'):
        direction_s = surface_name[0]
        direction_i = directions_conv[direction_s]
        boundary    = surface_name[1]

        if boundary == 'p':
            point[direction_i] = 1

    for i in range(3):
        bounding_corners[i]   = point[i] - eps
        bounding_corners[i+3] = point[i] + eps

    for dimTag in gmsh.model.getEntitiesInBoundingBox(*bounding_corners, dim=0):
        tags.append(dimTag[1])
    
    return tags

def gmsh_getMatrixTags(part_tags, all_tags):
    return [tag for tag in all_tags if tag not in part_tags]

def gmsh_affineTransform(tr, rot=np.eye(3)):
    """Returns the list representation of an affine transformation
    (see gmsh/model/occ/affineTransform in the GMSH doc)
    
    Arguments:
        tr {list} -- Translation vection
    
    Keyword Arguments:
        rot {numpy.ndarray} -- Rotation matrix (default: {np.eye(3, dtype=int)})
    Returns:
        [list] -- see description
    """
    return [*rot[0,:], tr[0], *rot[1,:], tr[1], *rot[2,:], tr[2], 0, 0, 0, 1]

def gmsh_setPeriodicity(vec, surfaces):
    """Sets surface mesh periodicity in the direction of the vector whose name is given
    
    Arguments:
        vec {str} -- 'x' or 'y' or 'z'import vtk
    """
    slave_name = '{}p'.format(vec)
    master_name = '{}m'.format(vec)
    axis = {'x': 0, 'y': 1, 'z': 2}
    eps = 1e-5
    true_list = 6*[True]
    n_periodic_vec = list(map(lambda x: int(x == vec), axis.keys()))
    smin = gmsh.model.getEntitiesForPhysicalGroup(2, surfaces[master_name]['physical_tag'])
    smax = gmsh.model.getEntitiesForPhysicalGroup(2, surfaces[slave_name]['physical_tag'])

    for i in range(len(smin)):
        min_bb = gmsh.model.getBoundingBox(2, smin[i])

        for j in range(len(smax)):
            min_bb_max_clone = list(gmsh.model.getBoundingBox(2, smax[j]))
            min_bb_max_clone[axis[vec]] -= 1
            min_bb_max_clone[axis[vec]+3] -= 1

            if list(map(lambda x, y: abs(x-y)<eps, min_bb_max_clone, min_bb)) == true_list:
                gmsh.model.mesh.setPeriodic(2, [smax[j]], [smin[i]], gmsh_affineTransform(n_periodic_vec))

def setPeriodic(coord):
    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1
    zmin = 0
    zmax = 1
    e = 1e-5
    smin = gmsh.model.getEntitiesInBoundingBox(xmin - e, ymin - e, zmin - e,
                                               (xmin + e) if (coord == 0) else (xmax + e),
                                               (ymin + e) if (coord == 1) else (ymax + e),
                                               (zmin + e) if (coord == 2) else (zmax + e),
                                               2)
    dx = (xmax - xmin) if (coord == 0) else 0
    dy = (ymax - ymin) if (coord == 1) else 0
    dz = (zmax - zmin) if (coord == 2) else 0

    for i in smin:
        bb = gmsh.model.getBoundingBox(i[0], i[1])
        bbe = [bb[0] - e + dx, bb[1] - e + dy, bb[2] - e + dz,
               bb[3] + e + dx, bb[4] + e + dy, bb[5] + e + dz]
        smax = gmsh.model.getEntitiesInBoundingBox(bbe[0], bbe[1], bbe[2],
                                                   bbe[3], bbe[4], bbe[5],
                                                   2)
        for j in smax:
            bb2 = list(gmsh.model.getBoundingBox(j[0], j[1]))
            bb2[0] -= dx; bb2[1] -= dy; bb2[2] -= dz
            bb2[3] -= dx; bb2[4] -= dy; bb2[5] -= dz
            if(abs(bb2[0] - bb[0]) < e and abs(bb2[1] - bb[1]) < e and
               abs(bb2[2] - bb[2]) < e and abs(bb2[3] - bb[3]) < e and
               abs(bb2[4] - bb[4]) < e and abs(bb2[5] - bb[5]) < e):
                trsf = [1, 0, 0, dx, 0, 1, 0, dy, 0, 0, 1, dz, 0, 0, 0, 1]
                gmsh.model.mesh.setPeriodic(2, [j[1]], [i[1]], trsf)

def surface_node_check(vec, slave_node_tag, master_node_tag):
    axis      = {'x': 0, 'y': 1, 'z': 2}
    eps       = 1e-5
    true_list = 2*[True]
    
    slave_coord, _ = gmsh.model.mesh.getNode(slave_node_tag)
    master_coord, _ = gmsh.model.mesh.getNode(master_node_tag)

    surf_slave_coord  = [slave_coord[i] for direction, i in axis.items() if direction != vec]
    surf_master_coord = [master_coord[i] for direction, i in axis.items() if direction != vec]

    if list(map(lambda x, y: abs(x-y)<eps, surf_slave_coord, surf_master_coord)) == true_list:
        periodicity_check = True
    else:
        error_msg = (
            'The surface coordinates of the slave node {} are not consistent '
            'with those of its master node {}.'
        ).format(slave_node_tag, master_node_tag)
        print(error_msg)
        print(list(map(lambda x, y: abs(x-y), surf_slave_coord, surf_master_coord)))
        periodicity_check = False
    
    inner_check = list(map(lambda x: eps < x < 1-eps, surf_slave_coord)) == true_list

    return periodicity_check and inner_check

def edge_node_check(direction, coord):
    eps = 1e-5

    return eps < coord[direction] < 1-eps

def edge_periodicity_check(vec, slave_nodes, master_nodes):
    eps = 1e-5
    n_master_nodes = master_nodes.shape[0]
    periodicity_check = True

    if slave_nodes.shape[0] == n_master_nodes:

        for i in range(n_master_nodes):
            slave_node_tag = int(slave_nodes[i, 0])
            slave_node_coord = slave_nodes[i, vec+1]

            master_node_tag = int(master_nodes[i, 0])
            master_node_coord = master_nodes[i, vec+1]

            if abs(slave_node_coord-master_node_coord) > 1.3e2*eps:
                error_msg = (
                    'The edge coordinates of the slave node {} are not consistent '
                    'with those of its master node {}.'
                ).format(slave_node_tag, master_node_tag)
                print(error_msg)
                print(abs(slave_node_coord-master_node_coord))
                periodicity_check = False
    
    return periodicity_check

def abaqus_heading_cmd(generic_name):
    heading_cmd = (
        '*Heading\n'
        '** Generic name: {}\n'
    ).format(generic_name.upper())
    
    return heading_cmd

def abaqus_orientation_cmd(symmetry_class, phases):
    orientation_cmd = ''

    if symmetry_class == 'trans_iso':
        default_cmd = (
            '*Orientation, name=ORI_{SET_NAME}\n'
            '{ORIENTATION[0]:.6f}, {ORIENTATION[1]:.6f}, {ORIENTATION[2]:.6f}, 1., 0., 0.\n'
            '2, 90.\n'
        )

        fiber_phases = [key for key in phases.keys() if key != 'matrix']

        for fiber_phase in fiber_phases:
            orientation_cmd += default_cmd.format(
                SET_NAME=fiber_phase.upper(),
                ORIENTATION=phases[fiber_phase]['orientation']
            )
    
    return orientation_cmd

def abaqus_section_cmd(symmetry_class, phases):
    if symmetry_class == 'iso':
        default_cmd = (
            '** Section: {PHASE_NAME}_SECTION\n'
            '*Solid Section, elset={PHASE_NAME}, material={PHASE_NAME}\n'
        )

        section_cmd = ''
        
        for phase in phases.keys():
            section_cmd += default_cmd.format(PHASE_NAME=phase.upper())

    elif symmetry_class == 'trans_iso':
        default_cmd = (
            '** Section: {PHASE_NAME}_SECTION\n'
            '*Solid Section, elset={PHASE_NAME}, orientation=ORI_{PHASE_NAME}, material=FIBER\n'
        )

        section_cmd = (
            '** Section: MATRIX_SECTION\n'
            '*Solid Section, elset=MATRIX, material=MATRIX\n'
        )

        fiber_keys = [key for key in phases.keys() if key != 'matrix']

        for fiber_key in fiber_keys:
            section_cmd += default_cmd.format(
                PHASE_NAME=fiber_key.upper()
            )
    
    return section_cmd

def cut_heading(mesh_string):
    occurrence_indexes = [x.start() for x in re.finditer('\n', mesh_string)]
    splitting_index    = occurrence_indexes[1] + 1
    return mesh_string[splitting_index:]

def abaqus_part_cmds(n_nodes, inp_filename, symmetry_class, phases):
    with open(inp_filename, 'r') as f:
        mesh_data = cut_heading(f.read())
    
    orientation_cmd = abaqus_orientation_cmd(symmetry_class, phases)
    section_cmd = abaqus_section_cmd(symmetry_class, phases)
    
    part_cmd = (
        '**\n'
        '** PARTS\n'
        '**\n'
        '*Part, name=RVE\n'
        '{NODES_N_ELEMENTS}'
        '{ORIENTATION_CMD}'
        '{SECTION_CMD}'
        '*End Part\n'
        '*Part, name=DUMMY_PART\n'
        '*NODE\n'
        '{X_DUMMY_NODE_TAG}, -0.5, 0.5, 0.5\n'
        '{Y_DUMMY_NODE_TAG}, 0.5, -0.5, 0.5\n'
        '{Z_DUMMY_NODE_TAG}, 0.5, 0.5, -0.5\n'
        '*End Part\n'
    ).format(
        NODES_N_ELEMENTS=mesh_data,
        ORIENTATION_CMD=orientation_cmd,
        SECTION_CMD=section_cmd,
        X_DUMMY_NODE_TAG=n_nodes+1,
        Y_DUMMY_NODE_TAG=n_nodes+2,
        Z_DUMMY_NODE_TAG=n_nodes+3
    )

    return part_cmd

def abaqus_nset_cmd(n_nodes, node_sets, assembly_instance):    
    default_cmd = (
        '*Nset, nset={SET_TYPE}_{SET_NAME}_{NODE_TAG:.0f}, instance={ASSEMBLY_INSTANCE}\n'
        '{NODE_TAG:.0f}\n'
    )

    nset_cmd = ''

    for set_type, datas in node_sets.items():
        for set_name, node_datas in datas.items():
            for node in node_datas:
                nset_cmd += default_cmd.format(
                    SET_TYPE=set_type.upper(),
                    SET_NAME=set_name.upper(),
                    NODE_TAG=node[0],
                    ASSEMBLY_INSTANCE=assembly_instance.upper()
                )
    
    nset_cmd += (
        '*Nset, nset=X_DUMMY_NODE, instance=DUMMY_INSTANCE\n'
        '{X_DUMMY_NODE_TAG}\n'
        '*Nset, nset=Y_DUMMY_NODE, instance=DUMMY_INSTANCE\n'
        '{Y_DUMMY_NODE_TAG}\n'
        '*Nset, nset=Z_DUMMY_NODE, instance=DUMMY_INSTANCE\n'
        '{Z_DUMMY_NODE_TAG}\n'
    ).format(
        X_DUMMY_NODE_TAG=n_nodes+1,
        Y_DUMMY_NODE_TAG=n_nodes+2,
        Z_DUMMY_NODE_TAG=n_nodes+3
    )

    return nset_cmd

def abaqus_equation_cmd(n_nodes, node_sets, E, periodic_mesh_bool, mesh_order):
    if periodic_mesh_bool:
        default_cmd = (
            '*Equation\n'
            '3\n'
            # Node set, dof, magnitude
            '{SET_TYPE}_{SLAVE_SET_NAME}_{SLAVE_NODE_TAG:.0f}, {DOF_NUMBER}, 1\n'
            '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG:.0f}, {DOF_NUMBER}, -1\n'
            '{DUMMY_NODE_AXIS}_DUMMY_NODE, {DOF_NUMBER}, {DISPL_MAGNITUDE:.6f}\n'
        )
        
        dummy_sets = ['X', 'Y', 'Z']
        eq_cmd = ''

        surface_keys = get_surface_keys()
        edge_keys    = get_edge_keys()
        vertex_keys  = get_vertex_keys()

        for i in range(0, len(surface_keys), 2):
            dummy_node_axis = surface_keys[i+1][0].upper()
            slave_set  = node_sets['inner_surface'][surface_keys[i+1]]
            master_set = node_sets['inner_surface'][surface_keys[i]]
            for j in range(slave_set.shape[0]):
                displ = np.dot(E, slave_set[j, 1:]-master_set[j, 1:])
                for k in range(1, 4):
                    eq_cmd += default_cmd.format(
                        SET_TYPE='INNER_SURFACE',
                        SLAVE_SET_NAME=surface_keys[i+1].upper(),
                        MASTER_SET_NAME=surface_keys[i].upper(),
                        DUMMY_NODE_AXIS=dummy_node_axis,
                        SLAVE_NODE_TAG=slave_set[j, 0],
                        MASTER_NODE_TAG=master_set[j, 0],
                        DOF_NUMBER=k,
                        DISPL_MAGNITUDE=-displ[k-1]
                    )

    else:
        if mesh_order == 2:
            n_interpolation_points = 6
            default_surface_cmd = (
                '*Equation\n'
                '8\n'
                # Node set, dof, magnitude
                '{SET_TYPE}_{SLAVE_SET_NAME}_{SLAVE_NODE_TAG:.0f}, {DOF_NUMBER}, 1\n'
                '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG[0]:.0f}, {DOF_NUMBER}, {MASTER_MAGNITUDE[0]:.6f}\n'
                '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG[1]:.0f}, {DOF_NUMBER}, {MASTER_MAGNITUDE[1]:.6f}\n'
                '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG[2]:.0f}, {DOF_NUMBER}, {MASTER_MAGNITUDE[2]:.6f}\n'
                '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG[3]:.0f}, {DOF_NUMBER}, {MASTER_MAGNITUDE[3]:.6f}\n'
                '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG[4]:.0f}, {DOF_NUMBER}, {MASTER_MAGNITUDE[4]:.6f}\n'
                '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG[5]:.0f}, {DOF_NUMBER}, {MASTER_MAGNITUDE[5]:.6f}\n'
                '{DUMMY_NODE_AXIS}_DUMMY_NODE, {DOF_NUMBER}, {DISPL_MAGNITUDE:.6f}\n'
            )
        
        else:
            n_interpolation_points = 3
            default_surface_cmd = (
                '*Equation\n'
                '5\n'
                # Node set, dof, magnitude
                '{SET_TYPE}_{SLAVE_SET_NAME}_{SLAVE_NODE_TAG:.0f}, {DOF_NUMBER}, 1\n'
                '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG[0]:.0f}, {DOF_NUMBER}, {MASTER_MAGNITUDE[0]:.6f}\n'
                '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG[1]:.0f}, {DOF_NUMBER}, {MASTER_MAGNITUDE[1]:.6f}\n'
                '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG[2]:.0f}, {DOF_NUMBER}, {MASTER_MAGNITUDE[2]:.6f}\n'
                '{DUMMY_NODE_AXIS}_DUMMY_NODE, {DOF_NUMBER}, {DISPL_MAGNITUDE:.6f}\n'
            )

            # default_edge_cmd = (
            #     '*Equation\n'
            #     '4\n'
            #     # Node set, dof, magnitude
            #     '{SET_TYPE}_{SLAVE_SET_NAME}_{SLAVE_NODE_TAG:.0f}, {DOF_NUMBER}, 1\n'
            #     '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG[0]:.0f}, {DOF_NUMBER}, {MASTER_MAGNITUDE[0]:.6f}\n'
            #     '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG[1]:.0f}, {DOF_NUMBER}, {MASTER_MAGNITUDE[1]:.6f}\n'
            #     '{DUMMY_NODE_AXIS}_DUMMY_NODE, {DOF_NUMBER}, {DISPL_MAGNITUDE:.6f}\n'
            # )

        dummy_sets = ['X', 'Y', 'Z']
        eq_cmd = ''

        surface_keys = get_surface_keys()
        edge_keys    = get_edge_keys()
        vertex_keys  = get_vertex_keys()

        for i in range(0, len(surface_keys), 2):
            dummy_node_axis = surface_keys[i+1][0].upper()
            slave_set  = node_sets['inner_surface'][surface_keys[i+1]]
            master_set = node_sets['inner_surface'][surface_keys[i]]
            for j in range(slave_set.shape[0]):
                vec = np.array([int(x == dummy_node_axis) for x in dummy_sets])
                displ = np.dot(E, vec)
                for k in range(1, 4):
                    eq_cmd += default_surface_cmd.format(
                        SET_TYPE='INNER_SURFACE',
                        SLAVE_SET_NAME=surface_keys[i+1].upper(),
                        MASTER_SET_NAME=surface_keys[i].upper(),
                        DUMMY_NODE_AXIS=dummy_node_axis,
                        SLAVE_NODE_TAG=slave_set[j, 0],
                        MASTER_NODE_TAG=master_set[n_interpolation_points*j:n_interpolation_points*(j+1), 0],
                        DOF_NUMBER=k,
                        MASTER_MAGNITUDE=master_set[n_interpolation_points*j:n_interpolation_points*(j+1), 1],
                        DISPL_MAGNITUDE=-displ[k-1]
                    )
        
        # for i in range(0, len(edge_keys), 4):
        #     const_coords = [edge_keys[i][3].upper(), edge_keys[i][0].upper()]
        #     for j in range(i+3, i, -1):
        #         slave_set = node_sets['inner_edge'][edge_keys[j]]
        #         master_set = node_sets['inner_edge'][edge_keys[j-1]]
        #         dummy_node_axis = const_coords[j%2]
        #         for k in range(slave_set.shape[0]):
        #             proj = np.array([int(x == dummy_node_axis) for x in ['x', 'y', 'z']])
        #             orientation = 1 if edge_keys[j][4] == 'm' else -1
        #             displ = orientation*np.dot(E, proj)
        #             for n in range(1, 4):
        #                 eq_cmd += default_edge_cmd.format(
        #                     SET_TYPE='INNER_EDGE',
        #                     SLAVE_SET_NAME=edge_keys[j].upper(),
        #                     MASTER_SET_NAME=edge_keys[j-1].upper(),
        #                     DUMMY_NODE_AXIS=dummy_node_axis,
        #                     SLAVE_NODE_TAG=slave_set[k, 0],
        #                     MASTER_NODE_TAG=slave_set[k, 1],
        #                     DOF_NUMBER=n,
        #                     MASTER_MAGNITUDE=slave_set[k, 2],
        #                     DISPL_MAGNITUDE=-displ[n-1]
        #                 )
    
    for i in range(0, len(edge_keys), 4):
            const_coords = [edge_keys[i][3].upper(), edge_keys[i][0].upper()]
            for j in range(i+3, i, -1):
                slave_set = node_sets['inner_edge'][edge_keys[j]]
                master_set = node_sets['inner_edge'][edge_keys[j-1]]
                dummy_node_axis = const_coords[j%2]
                for k in range(slave_set.shape[0]):
                    displ = np.dot(E, slave_set[k, 1:]-master_set[k, 1:])
                    for n in range(1, 4):
                        eq_cmd += (
                            '*Equation\n'
                            '3\n'
                            # Node set, dof, magnitude
                            '{SET_TYPE}_{SLAVE_SET_NAME}_{SLAVE_NODE_TAG:.0f}, {DOF_NUMBER}, 1\n'
                            '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG:.0f}, {DOF_NUMBER}, -1\n'
                            '{DUMMY_NODE_AXIS}_DUMMY_NODE, {DOF_NUMBER}, {DISPL_MAGNITUDE:.6f}\n'
                        ).format(
                            SET_TYPE='INNER_EDGE',
                            SLAVE_SET_NAME=edge_keys[j].upper(),
                            MASTER_SET_NAME=edge_keys[j-1].upper(),
                            DUMMY_NODE_AXIS=dummy_node_axis,
                            SLAVE_NODE_TAG=slave_set[k, 0],
                            MASTER_NODE_TAG=master_set[k, 0],
                            DOF_NUMBER=n,
                            DISPL_MAGNITUDE=-displ[n-1]
                        )

    for j in range(len(vertex_keys)-1, 0, -1):
            slave_set = node_sets['vertex'][vertex_keys[j]]
            master_set = node_sets['vertex'][vertex_keys[j-1]]
            dummy_node_axis = (
                'X'*(vertex_keys[j][1] != vertex_keys[j-1][1])
                + 'Y'*(vertex_keys[j][4] != vertex_keys[j-1][4])
                + 'Z'*(vertex_keys[j][7] != vertex_keys[j-1][7])
            )
            for k in range(slave_set.shape[0]):
                displ = np.dot(E, slave_set[k, 1:]-master_set[k, 1:])
                for n in range(1, 4):
                    eq_cmd += (
                        '*Equation\n'
                        '3\n'
                        # Node set, dof, magnitude
                        '{SET_TYPE}_{SLAVE_SET_NAME}_{SLAVE_NODE_TAG:.0f}, {DOF_NUMBER}, 1\n'
                        '{SET_TYPE}_{MASTER_SET_NAME}_{MASTER_NODE_TAG:.0f}, {DOF_NUMBER}, -1\n'
                        '{DUMMY_NODE_AXIS}_DUMMY_NODE, {DOF_NUMBER}, {DISPL_MAGNITUDE:.6f}\n'
                    ).format(
                        SET_TYPE='VERTEX',
                        SLAVE_SET_NAME=vertex_keys[j].upper(),
                        MASTER_SET_NAME=vertex_keys[j-1].upper(),
                        DUMMY_NODE_AXIS=dummy_node_axis,
                        SLAVE_NODE_TAG=slave_set[k, 0],
                        MASTER_NODE_TAG=master_set[k, 0],
                        DOF_NUMBER=n,
                        DISPL_MAGNITUDE=-displ[n-1]
                    )
    
    return eq_cmd

def abaqus_assembly_cmd(n_nodes, node_sets, strain_operator, generic_name, periodic_mesh_bool, mesh_order):

    nset_cmd = abaqus_nset_cmd(n_nodes, node_sets, 'RVE_INSTANCE')
    eq_cmd   = abaqus_equation_cmd(n_nodes, node_sets, strain_operator, periodic_mesh_bool, mesh_order)

    assembly_cmd = (
        '**\n'
        '** ASSEMBLY\n'
        '**\n'
        '*Assembly, name=CFRP-RVE\n'
        '*Instance, name=RVE_INSTANCE, part=RVE\n'
        '*End Instance\n'
        '*Instance, name=DUMMY_INSTANCE, part=DUMMY_PART\n'
        '*End Instance\n'
        '{NSET_CMD}'
        '** PBC Constraints:\n'
        '{EQ_CMD}'
        '*End Assembly\n'
    ).format(
        NSET_CMD=nset_cmd,
        EQ_CMD=eq_cmd
    )

    return assembly_cmd

def abaqus_material_cmd(mat_features):
    mat_cmd = (
        '**\n'
        '** MATERIALS\n'
        '**\n'
        '*Material, name=MATRIX\n'
        '*Elastic\n'
        '{YOUNG_MODULUS:.6f}, {POISSON_RATIO:.6f}\n'
    ).format(
        YOUNG_MODULUS=mat_features['matrix']['young'],
        POISSON_RATIO=mat_features['matrix']['poisson']
    )
    
    if mat_features['fiber']['behavior'] == 'iso':
        mat_cmd += (
            '**\n'
            '** MATERIALS\n'
            '**\n'
            '*Material, name=FIBER\n'
            '*Elastic\n'
            '{YOUNG_MODULUS:.6f}, {POISSON_RATIO:.6f}\n'
        ).format(
            YOUNG_MODULUS=mat_features['fiber']['young'],
            POISSON_RATIO=mat_features['fiber']['poisson']
        )
    
    elif mat_features['fiber']['behavior'] == 'trans_iso':
        mat_cmd += (
            '*Material, name=FIBER\n'
            '*Elastic, type=ENGINEERING CONSTANTS\n'
            '{YOUNG_T:.6f}, {YOUNG_T:.6f}, {YOUNG_L:.6f}, {POISSON_L:.6f},  {POISSON_T:.6f},  {POISSON_T:.6f}, {SHEAR_T:.6f}, {SHEAR_L:.6f}\n'
            '{SHEAR_L:.6f},\n'
        ).format(
            YOUNG_T=mat_features['fiber']['young_t'],
            YOUNG_L=mat_features['fiber']['young_l'],
            POISSON_L=mat_features['fiber']['poisson_l'],
            POISSON_T=mat_features['fiber']['poisson_t'],
            SHEAR_L=mat_features['fiber']['shear_l'],
            SHEAR_T=mat_features['fiber']['young_t']/(2*(1+mat_features['fiber']['poisson_l']))
        )

    return mat_cmd

def abaqus_block_bc_cmd(nodes):
    set_name = get_volume_key()

    full_command = (
        '*Boundary\n'
        'INNER_VOLUME_{SET_NAME}_{X_NODE_TAG:.0f}, 1, 1, 0\n'
        'INNER_VOLUME_{SET_NAME}_{Y_NODE_TAG:.0f}, 2, 2, 0\n'
        'INNER_VOLUME_{SET_NAME}_{Z_NODE_TAG:.0f}, 3, 3, 0\n'
    ).format(
        SET_NAME=set_name,
        X_NODE_TAG=nodes[0],
        Y_NODE_TAG=nodes[1],
        Z_NODE_TAG=nodes[2],
    )

    return full_command

def abaqus_dummy_bc_cmd(n_nodes, E):
    n = E.shape[0]
    x_displ = [E[i, 0] != 0 for i in range(n)]
    y_displ = [E[i, 1] != 0 for i in range(n)]
    z_displ = [E[i, 2] != 0 for i in range(n)]

    boundary_cmd = (
        '*Boundary\n'
        'X_DUMMY_NODE, 1, 1, {DISPL_MAGNITUDE[0]:.6f}\n'
        'X_DUMMY_NODE, 2, 2, {DISPL_MAGNITUDE[1]:.6f}\n'
        'X_DUMMY_NODE, 3, 3, {DISPL_MAGNITUDE[2]:.6f}\n'
        'Y_DUMMY_NODE, 1, 1, {DISPL_MAGNITUDE[3]:.6f}\n'
        'Y_DUMMY_NODE, 2, 2, {DISPL_MAGNITUDE[4]:.6f}\n'
        'Y_DUMMY_NODE, 3, 3, {DISPL_MAGNITUDE[5]:.6f}\n'
        'Z_DUMMY_NODE, 1, 1, {DISPL_MAGNITUDE[6]:.6f}\n'
        'Z_DUMMY_NODE, 2, 2, {DISPL_MAGNITUDE[7]:.6f}\n'
        'Z_DUMMY_NODE, 3, 3, {DISPL_MAGNITUDE[8]:.6f}\n'
    ).format(DISPL_MAGNITUDE=[*x_displ, *y_displ, *z_displ])

    return boundary_cmd

def abaqus_step_cmd(step_name, n_nodes, node_sets, strain_operator):
    blocked_nodes = node_sets['inner_volume']['omega_int'][:, 0]

    block_bc_cmd = abaqus_block_bc_cmd(blocked_nodes)
    dummy_bc_cmd = abaqus_dummy_bc_cmd(n_nodes, strain_operator)

    step_string = (
        '** \n'
        '** STEP: UNIQUE_STEP\n'
        '**\n'
        '*Step, name=UNIQUE_STEP, nlgeom=NO\n'
        '*Static\n'
        '1., 1., 1e-05, 1.\n'
        '** \n'
        '** BOUNDARY CONDITIONS\n'
        '** \n'
        '** Name: DUMMY_BC Type: Displacement/Rotation\n'
        '{DUMMY_BC}'
        '** Name: BLOCKED_NODES_BC Type: Symmetry/Antisymmetry/Encastre\n'
        '{BLOCKED_NODES_BC}'
        '** \n'
        '** OUTPUT REQUESTS\n'
        '** \n'
        '*Restart, write, frequency=0\n'
        '** \n'
        '** FIELD OUTPUT: F-Output-1\n'
        '** \n'
        '*Output, field\n'
        '*Node Output\n'
        'U\n'
        '*Element Output, directions=YES\n'
        'E, IVOL, S\n'
        '** \n'
        '** HISTORY OUTPUT: H-Output-1\n'
        '** \n'
        '*Output, history, variable=PRESELECT\n'
        '*End Step'
    ).format(
        DUMMY_BC=dummy_bc_cmd,
        BLOCKED_NODES_BC=block_bc_cmd
    )

    return step_string

def get_strain_operator(component):
    E = np.zeros((3, 3))
    col, row = list(map(int, component))

    E[col-1, row-1] += 0.5
    E[row-1, col-1] += 0.5

    return E

def secondsToText(sec):
    hours = sec//3600
    minutes = (sec - hours*3600)//60
    seconds = sec - hours*3600 - minutes*60
    result = ("{:0>2.0f}h".format(hours) if hours else "") + \
    ("{:0>2.0f}m".format(minutes)) + \
    ("{:0>2.0f}s".format(seconds))
    return result

def modified_voigt_weights():
    ones = np.ones((3, 3))
    weight_matrix = np.zeros((6, 6))

    weight_matrix[:3, :3]   = ones
    weight_matrix[:3, 3:6]  = np.sqrt(2)*ones
    weight_matrix[3:6, :3]  = np.sqrt(2)*ones
    weight_matrix[3:6, 3:6] = 2*ones

    return weight_matrix

def shape_functions(mesh_order):
    if mesh_order == 1:
        phi_1 = lambda xi, eta: 1 - xi - eta
        phi_2 = lambda xi, eta: xi
        phi_3 = lambda xi, eta: eta

        return (phi_1, phi_2, phi_3)
    
    # if mesh_order == 'se2':
    #     phi_1 = lambda xi: 0.5*(1-xi)
    #     phi_2 = lambda xi: 0.5*(1+xi)
    #     phi_3 = None

    #     return (phi_1, phi_2, phi_3)
    
    if mesh_order == 2:
        phi_1 = lambda xi, eta: -(1-xi-eta)*(1-2*(1-xi-eta))
        phi_2 = lambda xi, eta: -xi*(1-2*xi)
        phi_3 = lambda xi, eta: -eta*(1-2*eta)
        phi_4 = lambda xi, eta: 4*xi*(1-xi-eta)
        phi_5 = lambda xi, eta: 4*xi*eta
        phi_6 = lambda xi, eta: 4*eta*(1-xi-eta)

        return (phi_1, phi_2, phi_3, phi_4, phi_5, phi_6)

import subprocess
import json


def hashDict(resolution):
    """if input is dict: make a tuple from dict values so it is hashable as a key to another dict""" 
    if type(resolution)==dict and {"x","y","z"}==set(resolution.keys()):
        return (resolution["x"],resolution["y"],resolution["z"])
    else:
        return resolution

def getCommonPaths(
    rootPath,
    resolution_list,
    origin,
    dict_material_properties,
    OpenFiberSeg=True,
    exclusiveRVEstr=None
    ):

    if OpenFiberSeg:
        identifyerStr="fiberStruct_AMITEX.pickle"
    else:
        identifyerStr="*.txt"

    # finds all files in rootPath that have been segmented with OpenFiberSeg, but homogenized with Amitex at these specifications
    cmd = ["find", rootPath, "-name", identifyerStr, "-type", "f"]
    systemCall = subprocess.run(cmd, stdout=subprocess.PIPE)

    directories=systemCall.stdout.decode("utf-8").split("\n")

    if not OpenFiberSeg:
        directories=[d.split(".txt")[0] for d in directories if \
            "C_hom" not in d and "log_" not in d and "A2mean" not in d]

    if OpenFiberSeg:
        directories=[str(os.path.dirname(p))  for p in directories]        

    unProcessedDirectories={}

    directories.sort()

    if exclusiveRVEstr is not None:
        directories=[d for d in directories if exclusiveRVEstr in d]

    for dir in directories:
        if dir:# skip empty first element in find results
            for resolution in resolution_list:

                if resolution=="all" and not OpenFiberSeg:
                    raise ValueError("Using legacy microstructure generation, resolution cannot be infered from inputs, must be prescribed manually (can't be set to \"all\")")

                if type(resolution) == int:

                    resolution=min(resolution,max_vtk_size)

                elif type(resolution) == dict and\
                        ['x', 'y', 'z'] == list(resolution.keys()):

                    resolution["x"]=min(resolution["x"],max_vtk_size)    
                    resolution["y"]=min(resolution["y"],max_vtk_size)
                    resolution["z"]=min(resolution["z"],max_vtk_size)

                if hashDict(resolution) in list(unProcessedDirectories.keys()):
                    unProcessedDirectories[hashDict(resolution)].append(dir+"/")
                else:
                    unProcessedDirectories[hashDict(resolution)]=[dir+"/"]
                    
            unProcessedDirectories[hashDict(resolution)].sort()

    return unProcessedDirectories

#Generate a tensor, any order, any size
#Value is the default value, commonly 0
def initTensor(value, *lengths):
	list = []
	dim = len(lengths)
	if dim == 1:
		for i in range(lengths[0]):
			list.append(value)
	elif dim > 1:
		for i in range(lengths[0]):
			list.append(initTensor(value, *lengths[1:]))
	return list

def import_hmgnzt_quad( pathMT_library,file_name ):
	"""import guass quadrature points and weigths from csv file"""
	i = 0
	
	with open(pathMT_library+"Points de Gauss/quadratures/" + file_name, 'r',encoding='utf8' ) as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		total_lines = 0
		for line in spamreader:
			total_lines = total_lines + 1 
		
		quad = initTensor(0., total_lines, 2)
		print
		print ("Importing data from", file_name)
		print ("The file contains", total_lines, "lines.")
		csvfile.seek(0)
		i = 0
		for row in spamreader:
			quad[i][0] = float(row[0])
			quad[i][1] = float(row[1])
			i = i + 1
	return quad
