"""
This file is a modified version of the pipe script from the openPMD-api.

Authors: Richard Pausch, Franz Poeschel, Nico Wrobel, Finn-Ole Carstens
License: LGPLv3+
"""

import sys

import openpmd_api as opmd
import numpy as np


class vec3D:
    """
    a helper class to easily handle 3D vectors in python without the need
    to reference another dimension of a numpy array
    """

    x = 0
    y = 0
    z = 0

    def __init__(self, x, y, z):
        """
        initialize with 3 values (x,y,z)

        Arguments:
        x: first value
        y: second value
        z: third value
        """
        self.x = x
        self.y = y
        self.z = z

    def prod(self):
        """
        product of all 3 components

        returns x * y * z
        """
        return self.x * self.y * self.z

    def print(self):
        """
        helper function to print values to screen
        """
        print("x: {}".format(self.x))
        print("y: {}".format(self.y))
        print("z: {}".format(self.z))

    def __truediv__(self, other):
        """
        component wise division using the '/' operator

        Arguments:
        other: float value
               the number by which all 3 components should be divided

        Return:
        vec3D( x/other, y/other, z/other )
        """
        return vec3D(self.x / other, self.y / other, self.z / other)


class addParticles2File:
    def print(self, string):
        """
        helper function that prints information depending on the set
        verbose level

        Arguments:
        string: string
                message to post
        """
        if self.verbose:
            print("\t" * self.tabs + string)

    def __init__(self, filename_out, speciesName="e", verbose=False):
        """
        initialization of manipulation routine

        This class writes particles named speciesName into a new openPMD file
        at filename_out.

        Arguments:
        filename_out: string
                path to bp file to create
        speciesName: string
                short name in PIConGPU for the species to create
        verbose: bool
                True: print output, False: Do not print output to screen
        """
        self.verbose = verbose  # verbose level
        self.tabs = 0  # tab counter for output

        self.timestep = 0  # time step (fixed to 0)
        self.speciesName = speciesName
        self.filename_out = filename_out


        # momentum unit
        self.unitMomentum = 1.0
        # position unit is same as cellSize

        self.has_probeE = False
        self.has_probeB = False
        self.has_momentumPrev1 = False
        self.has_id = False
        self.has_transitionRadiationMask = False

        # extract data type for position
        self.dtype_position = np.dtype("float32")
        # extract data type for positionOffset
        self.dtype_positionOffset = np.dtype("int32")
        # extract data type for momentum
        self.dtype_momentum = np.dtype("float32")
        # extract data type for weighting
        self.dtype_weighting = np.dtype("float32")

        if self.has_probeE:
            # type of E-Field
            self.dtype_probeE = np.dtype("float32")
        self.print("contains probeE = {}".format(self.has_probeE))
        if self.has_probeB:
            # type of B-Field
            self.dtype_probeB = np.dtype("float32")
        self.print("contains probeB {}".format(self.has_probeB))

        if self.has_id:
            # type of particleID
            self.dtype_id = np.dtype("uint64")
        self.print("contains id =  {}".format(self.has_id))

        if self.has_momentumPrev1:
            # type of momentumPrev1
            self.dtype_momentumPrev1 = np.dtype("float32")
        self.print("contains momentumPrev1 =  {}".format(self.has_momentumPrev1))
        if self.has_transitionRadiationMask:
            # type of transitionRadiationMask
            self.dtype_transitionRadiationMask = np.dtype("int8")
        self.print("contains transitionRadiationMask =  {}".format(self.has_transitionRadiationMask))

        self.dtype_patchOffset = np.dtype("uint64")
        self.dtype_patchExtent = np.dtype("uint64")
        self.dtype_patchNum =    np.dtype("uint64")
        self.dtype_patchNumOff = np.dtype("uint64")

        self.cellSize = vec3D(0.5*0.1772e-6, 0.5*0.1772e-6, 0.5*0.1772e-6)
        # particle patch offsets in cells
        off_x = np.array([0], dtype=self.dtype_patchOffset)
        off_y = np.array([0], dtype=self.dtype_patchOffset)
        off_z = np.array([0], dtype=self.dtype_patchOffset)

        # get extent of each GPU in cells (per dimension)
        ext_x = np.array([128], dtype=self.dtype_patchExtent)
        ext_y = np.array([256], dtype=self.dtype_patchExtent)
        ext_z = np.array([128], dtype=self.dtype_patchExtent)

        # extract number of GPUs used in each dimension
        # use np.unique() to reduce patches offset and len() to get number
        # of GPUs per dimension
        self.N_gpus = vec3D(len(np.unique(off_x)), len(np.unique(off_y)), len(np.unique(off_z)))

        # get patch offset
        self.offset = vec3D(off_x, off_y, off_z)
        self.extent = vec3D(ext_x, ext_y, ext_z)

        # get number of particles before each patch (placeholder, might not be needed)
        self.numParticlesOffset = np.array([0], dtype=self.dtype_patchNumOff)
        # get number of particles in each patch
        self.numParticles = np.array([1], dtype=self.dtype_patchNum)


    def addParticles(self, pos, mom, w):
        """
        add particles to the restart file

        Arguments:
        pos - vec3 array
              position in SI units
        mom - vec3 array
              momentum in SI units
        w - float array
            macro particle weighting
        """
        self.N_particles_input = len(w)  # number of particles to add

        # calculate positionOffset (cell location) from given position
        self.positionOffset = vec3D(
            (pos.x / self.cellSize.x).astype(self.dtype_positionOffset),
            (pos.y / self.cellSize.y).astype(self.dtype_positionOffset),
            (pos.z / self.cellSize.z).astype(self.dtype_positionOffset),
        )

        # calculate (in cell) position from given position
        self.position = vec3D(
            (np.mod(pos.x, self.cellSize.x) / self.cellSize.x).astype(self.dtype_position),
            (np.mod(pos.y, self.cellSize.y) / self.cellSize.y).astype(self.dtype_position),
            (np.mod(pos.z, self.cellSize.z) / self.cellSize.z).astype(self.dtype_position),
        )

        # calculate momentum in PIC units from given momentum
        self.momentum = vec3D(
            (mom.x * w / self.unitMomentum).astype(self.dtype_momentum),
            (mom.y * w / self.unitMomentum).astype(self.dtype_momentum),
            (mom.z * w / self.unitMomentum).astype(self.dtype_momentum),
        )

        # weighting
        self.weighting = w.copy().astype(self.dtype_weighting)

        if self.has_probeE:
            # data for witnessed E-Field
            temp_zeros = np.zeros(len(w), dtype=self.dtype_probeE)
            self.probeE = vec3D(temp_zeros, temp_zeros, temp_zeros)

        if self.has_probeB:
            # data for witnessed B-Field
            temp_zeros = np.zeros(len(w), dtype=self.dtype_probeB)
            self.probeB = vec3D(temp_zeros, temp_zeros, temp_zeros)

        if self.has_id:
            # give every particle an ID
            self.id = np.arange(len(w), dtype=self.dtype_id)

        if self.has_momentumPrev1:
            # give momentumPrev1 for all particles ( used for radiation plugin )
            temp_zeros = np.zeros(len(w), dtype=self.dtype_momentumPrev1)
            self.momentumPrev1 = vec3D(temp_zeros, temp_zeros, temp_zeros)

        if self.has_transitionRadiationMask:
            # give every particle a transitionRadiationMask ( used for transition radiation plugin )
            self.transitionRadiationMask = np.zeros(len(w), dtype=self.dtype_transitionRadiationMask)

    def makePatchMask(self):
        """
        calculate particle patches for given particles
        """
        # create empty patch mask (N_GPUs x N_particles)
        self.patch_mask = np.empty((self.N_gpus.prod(), self.N_particles_input), dtype=bool)

        # calculate patch  for each GPU
        for i in np.arange(self.N_gpus.prod()):
            # x direction
            a = np.greater_equal(self.positionOffset.x, self.offset.x[i])
            b = np.less(self.positionOffset.x, self.offset.x[i] + self.extent.x[i])

            # y direction
            c = np.greater_equal(self.positionOffset.y, self.offset.y[i])
            d = np.less(self.positionOffset.y, self.offset.y[i] + self.extent.y[i])

            # z direction:
            e = np.greater_equal(self.positionOffset.z, self.offset.z[i])
            f = np.less(self.positionOffset.z, self.offset.z[i] + self.extent.z[i])

            # combine all 3*2 bools to just give true or false for GPU(i)
            tmp1 = np.logical_and(np.logical_and(a, b), np.logical_and(c, d))
            tmp2 = np.logical_and(tmp1, np.logical_and(e, f))
            self.patch_mask[i, :] = tmp2

        # determine number of particles in all patches
        self.numParticles = np.sum(self.patch_mask, axis=1, dtype=self.dtype_patchNum)
        # calculate number of particles before the patch
        self.numParticlesOffset = np.cumsum(self.numParticles, dtype=self.dtype_patchNumOff) - self.numParticles
        # fix possible negative value for first patch (if number of particles
        # in first patch != 0)
        self.numParticlesOffset[0] = 0

    def writeParticles(self):
        """
        write all particle data to checkpoint with the help of the pipe class
        """
        self.print("make patch mask")
        self.makePatchMask()  # calculate particle patch
        self.N_particles = np.sum(self.numParticles)

        self.createParticleFile()

    def createParticleFile(self):
        """
        create openPMD series and write particles to it
        """
        series = opmd.Series(self.filename_out, opmd.Access.create)
        iteration = series.iterations[self.timestep]
        particle = iteration.particles[self.speciesName]

        # get openPMD records
        position =                  particle["position"]
        position_offset =           particle["positionOffset"]
        momentum =                  particle["momentum"]
        weighting =                 particle["weighting"]

        particlePatch =             particle.particle_patches
        patch_offset =              particlePatch["offset"]
        patch_extent =              particlePatch["extent"]
        patch_numParticles =        particlePatch["numParticles"]
        patch_numParticlesOffset =  particlePatch["numParticlesOffset"]


        # datasets for new data
        dataset_position =          opmd.Dataset(self.dtype_position,       (self.N_particles, ))
        dataset_position_offset =   opmd.Dataset(self.dtype_positionOffset, (self.N_particles, ))
        dataset_momentum =          opmd.Dataset(self.dtype_momentum,       (self.N_particles, ))
        dataset_weighting =         opmd.Dataset(self.dtype_weighting,      (self.N_particles, ))

        dataset_patch_extent =      opmd.Dataset(self.dtype_patchExtent, (self.N_gpus.prod(), ))
        dataset_patch_offset =      opmd.Dataset(self.dtype_patchOffset, (self.N_gpus.prod(), ))

        dataset_patch_numParticles =        opmd.Dataset(self.dtype_patchNum,    (self.N_gpus.prod(), ))
        dataset_patch_numParticlesOffset =  opmd.Dataset(self.dtype_patchNumOff, (self.N_gpus.prod(), ))


        # create new datasets in openPMD
        position["x"].reset_dataset(dataset_position)
        position["y"].reset_dataset(dataset_position)
        position["z"].reset_dataset(dataset_position)

        position_offset["x"].reset_dataset(dataset_position_offset)
        position_offset["y"].reset_dataset(dataset_position_offset)
        position_offset["z"].reset_dataset(dataset_position_offset)

        momentum["x"].reset_dataset(dataset_momentum)
        momentum["y"].reset_dataset(dataset_momentum)
        momentum["z"].reset_dataset(dataset_momentum)

        weighting[opmd.Mesh_Record_Component.SCALAR].reset_dataset(dataset_weighting)

        patch_offset["x"].reset_dataset(dataset_patch_offset)
        patch_offset["y"].reset_dataset(dataset_patch_offset)
        patch_offset["z"].reset_dataset(dataset_patch_offset)

        patch_extent["x"].reset_dataset(dataset_patch_extent)
        patch_extent["y"].reset_dataset(dataset_patch_extent)
        patch_extent["z"].reset_dataset(dataset_patch_extent)

        particlePatch["numParticlesOffset"][opmd.Mesh_Record_Component.SCALAR].reset_dataset(dataset_patch_numParticlesOffset)
        particlePatch["numParticles"][opmd.Mesh_Record_Component.SCALAR].reset_dataset(dataset_patch_numParticles)


        # set unit dimensions of patch
        # for position, they are set by default
        patch_offset.unit_dimension = {
            opmd.Unit_Dimension.L:  1,
        }
        patch_extent.unit_dimension = {
            opmd.Unit_Dimension.L:  1,
        }
        momentum.unit_dimension = {
            opmd.Unit_Dimension.L:  1,
            opmd.Unit_Dimension.M:  1,
            opmd.Unit_Dimension.T:  -1,
        }

        # set units
        position["x"].unit_SI = self.cellSize.x
        position["y"].unit_SI = self.cellSize.y
        position["z"].unit_SI = self.cellSize.z

        position_offset["x"].unit_SI = self.cellSize.x
        position_offset["y"].unit_SI = self.cellSize.y
        position_offset["z"].unit_SI = self.cellSize.z

        momentum["x"].unit_SI = self.unitMomentum
        momentum["y"].unit_SI = self.unitMomentum
        momentum["z"].unit_SI = self.unitMomentum


        # write data
        position["x"][:] = self.position.x
        position["y"][:] = self.position.y
        position["z"][:] = self.position.z

        position_offset["x"][:] = self.positionOffset.x
        position_offset["y"][:] = self.positionOffset.y
        position_offset["z"][:] = self.positionOffset.z

        momentum["x"][:] = self.momentum.x
        momentum["y"][:] = self.momentum.y
        momentum["z"][:] = self.momentum.z

        weighting[opmd.Mesh_Record_Component.SCALAR][:] = self.weighting

        patch_offset["x"].store(0, self.offset.x)
        patch_offset["y"].store(0, self.offset.y)
        patch_offset["z"].store(0, self.offset.z)

        patch_extent["x"].store(0, self.extent.x)
        patch_extent["y"].store(0, self.extent.y)
        patch_extent["z"].store(0, self.extent.z)

        patch_numParticles[opmd.Mesh_Record_Component.SCALAR].store(0, self.numParticles)
        patch_numParticlesOffset[opmd.Mesh_Record_Component.SCALAR].store(0, self.numParticlesOffset)

        # flush and close file
        series.flush()
        series.close()

