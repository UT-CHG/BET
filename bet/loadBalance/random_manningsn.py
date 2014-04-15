# Lindley Graham 4/10/2013
"""
This module contains functions to pull data from a ``fort.61`` and the
:class:`runSet` which controls the running of ADCIRC simulations within a set
of processors allocated by the submission script
"""
import numpy as np
import glob, os, stat, subprocess, shutil
import polysim.pyADCIRC.fort13_management as f13
import polysim.pyADCIRC.fort15_management as f15
from polysim.pyADCIRC.basic import pickleable
import polysim.pyGriddata.table_to_mesh_map as tmm
from polysim.pyGriddata.file_management import copy, mkdir
import polysim.pyADCIRC.plotADCIRC as plot
import polysim.pyADCIRC.prep_management as prep
import polysim.pyADCIRC.output as output
import polysim.run_framework.domain as dom
import scipy.io as sio

def loadmat(save_file, base_dir, grid_dir, save_dir, basis_dir):
    """
    Loads data from ``save_file`` into a
    :class:`~polysim.run_framework.random_manningsn.runSet` object. Reconstructs
    :class:`~polysim.run_framework.random_manningsn.domain`. Fixes dry data if
    it was recorded.

    :param string save_file: local file name
    :param string grid_dir: directory containing ``fort.14``, ``fort.15``, and
        ``fort.22``
    :param string save_dir: directory where ``RF_directory_*`` are
        saved, and where fort.13 is located
    :param string basis_dir: directory where ``landuse_*`` folders are located
    :param string base_dir: directory that contains ADCIRC executables, and
        machine specific ``in.prep#`` files
    :rtype: tuple of :class:`~polysim.run_framework.random_manningsn.runSet` and
        :class:`~polysim.run_framework.random_manningsn.domain` objects
    :returns: (main_run, domain)

    """

    # the lines below are only necessary if you need to update what the
    # directories are when swithcing from euclid to your desktop/laptop
    # assumes that the landuse directory and ADCIRC_landuse directory are in
    # the same directory
    domain = dom.domain(grid_dir)
    domain.update()
    #domain.get_Triangulation()
    domain.set_station_bathymetry()

    main_run = runSet(grid_dir, save_dir, basis_dir, base_dir=base_dir)
    main_run.time_obs = {}
    main_run.ts_data = {}
    main_run.nts_data = {}

    # load the data from at *.mat file
    mdat = sio.loadmat(save_dir+'/'+save_file)
    if mdat.has_key('mann_pts'):
        mann_pts = mdat['mann_pts']
    else:
        mann_pts = None

    for k, v in mdat.iteritems():
        skey = k.split('_')
        if skey[-1] == 'time':
            # check to see if the key is "*_time"
            main_run.time_obs[skey[0]] = v
        elif f15.filetype.has_key(skey[0]) or skey[0] == 'timemax63':
            if len(v.shape) == 2:
                # check to see if key is nts_data
                main_run.nts_data[skey[0]] = v
            elif skey[0] == 'timemax63':
                main_run.nts_data[skey[0]] = v
            else:
                # check to see if key is ts_data
                main_run.ts_data[skey[0]] = v

    if main_run.ts_data.has_key('fort63'):
        main_run.fix_dry_nodes(domain)
        if main_run.ts_data.has_key('fort61'):
            main_run.fix_dry_data(domain)
    if main_run.ts_data.has_key('maxele63'):
        main_run.fix_dry_nodes_nts(domain)

    return (main_run, domain, mann_pts)

def fix_dry_data(ts_data, data):
    """
    Fix dry elevation station data flags

    :param ts_data: time series data
    :param data: :class:`~polysim.run_framework.domain`
    :rtype: dict()
    :returns: ts_data

    """
    mdat = np.ma.masked_equal(ts_data['fort61'], -99999.0)

    for k, v in enumerate(data.stations['fort61']):
        mdat[k-1, :, :] = mdat[k-1, :, :] + v.bathymetry

    ts_data['fort61'] = mdat.filled(0.0)
    return ts_data

def fix_dry_nodes(ts_data, data):
    """
    Fix dry elevation data flags

    :param ts_data: time series data
    :param data: :class:`~polysim.run_framework.domain`
    :rtype: dict()
    :returns: ts_data

    """
    mdat = np.ma.masked_equal(ts_data['fort63'], -99999.0)

    for k, v in data.node.iteritems():
        mdat[k-1, :, :] = mdat[k-1, :, :] + v.bathymetry

    ts_data['fort63'] = mdat.filled(0.0)
    return ts_data

def fix_dry_nodes_nts(nts_data, data):
    """
    Fix dry elevation data flags

    :param nts_data: non time series data
    :param data: :class:`~polysim.run_framework.domain`
    :rtype: dict()
    :returns: nts_data

    """
    mdat = np.ma.masked_equal(nts_data['maxele63'], -99999.0)

    for k, v in data.node.iteritems():
        mdat[k-1, :] = mdat [k-1, :] + v.bathymetry

    nts_data['maxele63'] = mdat.filled(0.0)
    return nts_data

def convert_to_hours(time_obs):
    """
    Converts ``time_obs`` from seconds to hours

    :param time_obs: observation times in seconds
    :rtype: dict()
    :returns: time_obs

    """
    for k in time_obs.iterkeys():
        time_obs[k] = time_obs[k] / (60.0 * 60.0)
    return time_obs

def convert_to_days(time_obs):
    """
    Converts ``time_obs`` from seconds to days

    :param time_obs: observation times in seconds
    :rtype: dict()
    :returns: time_obs

    """
    for k in time_obs.iterkeys():
        time_obs[k] = time_obs[k] / (60.0 * 60.0 * 24.0)
    return time_obs

def convert_to_percent(nts_data, data):
    """
    Converts ``nts_data['tinun63']`` from seconds to percent of RNDAY

    :param nts_data: non-time-series data
    :param data: :class:`~polysim.run_framework.domain`
    :rtype: dict()
    :returns: nts_data

    """
    nts_data['tinun63'] = nts_data['tinun63'] / (60.0 * 60.0 * 24.0 * data.time.rnday)

def concatenate(run_data1, run_data2):
    """
    Combine data from ``run_data1`` and ``run_data2``
    :class:`~polysim.run_framework.random_manningsn.runSet` with another
    :class:`~polysim.run_framework.random_manningsn.runSet` (``other_run``)
    and points from both runs

    To combine several ``run_data``s use::
        run_list = [run1, run2, run3]
        points_list = [points1, points2, points3]
        run_data_list = zip(run_list, points_list)
        reduce(concatenate, run_data_list)

    :param run_data1: (runSet for run1, sample points for run1)
    :type tuple: (:class:`~polysim.run_framework.random_manningsn.runSet`,
        np.array)
    :param run_data2: (runSet for run2, sample points for run2)
    :type tuple: (:class:`~polysim.run_framework.random_manningsn.runSet`,
        np.array)
    :returns: (run_data, points)
    :rtype: tuple

    """
    run1 = run_data1[0]
    points1 = run_data1[1]

    run2 = run_data2[0]
    points2 = run_data2[1]

    # concatenate nontimeseries data
    for k, v in run1.nts_data.iteritems():
        run1.nts_data[k] = np.concatenate((v, run2.nts_data[k]), axis=v.ndim-1)
    # concatenate timeseries data
    for k, v in run1.ts_data.iteritems():
        run1.ts_data[k] = np.concatenate((v, run2.ts_data[k]), axis=v.ndim-1)
    # concatenate time_obes data
    for k, v in run1.time_obs.iteritems():
        run1.time_obs[k] = np.concatenate((v, run2.time_obs[k]), axis=v.ndim-1)
    # concatenate points
    points = np.concatenate((points1, points2), axis=points1.ndim-1)

    run_data = (run1, points)
    return run_data

class runSet(pickleable):
    """
    This class controls the running of :program:`ADCIRC` within the processors
    allocated by the submission script

    grid_dir
        directory containing ``fort.14``, ``fort.15``, and ``fort.22``
    save_dir
        directory where ``RF_directory_*`` are saved, and where fort.13 is
        located
    basis_dir
        directory where ``landuse_*`` folders are located
    base_dir
        directory that contains ADCIRC executables, and machine
        specific ``in.prep#`` files
    num_of_parallel_runs
        size of batch of jobs to be submitted to queue
    script_name
    nts_data
        non timeseries data
    ts_data
        timeseries data
    time_obs
        observation times for timeseries data

    """
    def __init__(self, grid_dir, save_dir, basis_dir,
            num_of_parallel_runs=10, base_dir=None, script_name=None):
        """
        Initialization
        """
        #: string, directory containing ``fort.14``, ``fort.15``, and
        #  ``fort.22*`` 
        self.grid_dir = grid_dir
        self.save_dir = save_dir
        """
        string, directory where ``RF_directory_*`` are saved, and
        where ``fort.13`` is located
        """
        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)
            fort13_file = save_dir.rpartition('/')[0]+'/fort.13'
            copy(fort13_file, save_dir)
        #: string, directory where ``landuse_*`` folders are located
        self.basis_dir = basis_dir
        if base_dir:
            self.base_dir = base_dir
            """
            directory that contains ADCIRC executables, and machine
            specific ``in.prep#`` files
            """
        else:
            self.base_dir = basis_dir.rpartition('/')[0]
        self.prep_dir = basis_dir.rpartition('/')[0]
        #: int, size of batch of jobs to be submitted to queue
        self.num_of_parallel_runs = num_of_parallel_runs
        #: dict() of :class:`np.ndarray`, timeseries data
        self.ts_data = None
        #: dict() of :class:`np.ndarray`, non-timeseries data
        self.nts_data = None
        #: list(), list of ``RF_directory_*/`` names
        self.rf_dirs = None
        #: dict() of :class:`np.array`, time in (s) of observations
        self.time_obs = None
        if script_name:
            #: string, name of the batch bash script
            self.script_name = script_name
        else:
            self.script_name = "run_job_batch.sh"
        super(runSet, self).__init__()

    def initialize_random_field_directories(self, num_procs=12):
        """
        Make directories for parallel funs of random fields

        :rtype: list()
        :returns: list of paths to ``RF_directory_*``

        """
        # Check to see if some of the directories already exist
        rf_dirs = glob.glob(self.save_dir+'/RF_directory_*')
        num_dir = len(rf_dirs)
        # set up all rf_dirs
        if num_dir >= self.num_of_parallel_runs:
            for path in rf_dirs:
                self.setup_rfdir(path, num_procs)
        elif num_dir < self.num_of_parallel_runs:
            for i in xrange(num_dir, self.num_of_parallel_runs):
                rf_dirs.append(self.save_dir+'/RF_directory_'+str(i+1))
                self.setup_rfdir(rf_dirs[i], num_procs)
        self.rf_dirs = rf_dirs
        #PARALLEL: create file containing the list of rf_dirs
        self.update_dir_file(self.num_of_parallel_runs)
        self.write_prep_script(1, self.num_of_parallel_runs)
        self.write_prep_script(2, self.num_of_parallel_runs)
        self.write_prep_script(5, self.num_of_parallel_runs)
        subprocess.call(['./prep_1.sh'], cwd=self.save_dir)
        subprocess.call(['./prep_2.sh'], cwd=self.save_dir)
        return rf_dirs

    def remove_random_field_directories(self):
        """
        Remove directories for parallel funs of random fields

        """
        # Check to see if some of the directories already exist
        rf_dirs = glob.glob(self.save_dir+'/RF_directory_*')
        # remove all rf_dirs
        for rf_dir in rf_dirs:
            shutil.rmtree(rf_dir)

    def setup_rfdir(self, path, num_procs):
        """
        Creates the directory path and copies required files from
        ``self.base_dir`` into path:type path: string

        :param path: folder_name
        :param num_procs: number of processors per :program:`ADCIRC` run

        """
        mkdir(path)
        copy(self.save_dir+'/fort.13', path)
        # crete sybolic links from fort.* files to path
        inputs1 = glob.glob(self.grid_dir+'/fort.1*')
        inputs2 = glob.glob(self.grid_dir+'/fort.2*')
        inputs0 = glob.glob(self.grid_dir+'/fort.01*')
        inputs = inputs0 + inputs1 + inputs2
        if self.grid_dir+'/fort.13' in inputs:
            inputs.remove(self.grid_dir+'/fort.13')
        if not(self.grid_dir+'/fort.019' in inputs):
            if self.grid_dir+'/fort.015' in inputs:
                inputs.remove(self.grid_dir+'/fort.015')
        for fid in inputs:
            if os.path.exists(path+'/'+fid.rpartition('/')[-1]):
                os.remove(path+'/'+fid.rpartition('/')[-1])
            os.symlink(fid, path+'/'+fid.rpartition('/')[-1])
            #copy(fid, path+'/'+fid.rpartition('/')[-1])
        if not(os.path.exists(path+'/adcprep')):
            os.symlink(self.base_dir+'/adcprep', path+'/adcprep')
        prep.write_1(path, num_procs)
        prep.write_2(path, num_procs)
        prep.write_5(path, num_procs)

    def write_run_script(self, num_procs, num_jobs, procs_pnode, TpN,
            screenout=True, num_writers=None):
        """
        Creats a bash script called run_job_batch.sh

        :type num_procs: int
        :param num_procs: number of processors per job
        :type num_jobs: int
        :param num_jobs: number of jobs to run
        :param int procs_pnode: number of processors per node
        :param boolean screenout: flag (True --  write ``ADCIRC`` output to
            screen, False -- write ``ADCIRC`` output to temp file)
        :param int num_writers: number of MPI processes to dedicate soley to
            the task of writing ascii files
        :param int TpN: number of tasks (cores to use) per node (wayness)
        :rtype: string
        :returns: name of bash script for running a batch of jobs within our
            processor allotment

        """
        tmp_file = self.script_name.partition('.')[0]+'.tmp'
        with open(self.base_dir+'/'+self.script_name, 'w') as f:
            f.write('#!/bin/bash\n')
            # change i to 2*i or something like that to no use all of the
            # processors on a node?
            for i in xrange(num_jobs):
                line = 'ibrun -n {:d} -o {:d} '.format(num_procs,
                        num_procs*i*(procs_pnode/TpN))
                line += './padcirc -I {0} -O {0} '.format(self.rf_dirs[i])
                if num_writers:
                    line += '-W '+str(num_writers)+' '
                if not(screenout):
                    line += '> '+tmp_file
                line += ' &\n'
                f.write(line)
            f.write('wait\n')
        curr_stat = os.stat(self.base_dir+'/'+self.script_name)
        os.chmod(self.base_dir+'/'+self.script_name,
                curr_stat.st_mode | stat.S_IXUSR)
        return self.script_name

    def write_prep_script(self, n, num_jobs, screenout=False):
        """
        Creats a bash script to run :program:`adcprep` with ``in.prepn``

        :param int n: n for ``in.prepn`` input to ADCPREP
        :param int num_jobs: number of jobs to run
        :param boolean screenout: flag (True --  write ``ADCPREP`` output to
            screen, False -- write ``ADCPREP`` output to ``prep_o.txt`` file)
        :rtype: string
        :returns: name of bash script for prepping a batch of jobs within our
            processor allotment

        """
        with open(self.save_dir+'/prep_'+str(n)+'.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            line = "parallel '(cd {} && ./adcprep < in.prep"+str(n)
            if not(screenout):
                line += " > prep_o.txt"
            line += ")' :::: dir_list\n"
            f.write(line)
            f.write("wait\n")
        curr_stat = os.stat(self.save_dir+'/prep_'+str(n)+'.sh')
        os.chmod(self.save_dir+'/prep_'+str(n)+'.sh',
                curr_stat.st_mode | stat.S_IXUSR)
        return self.save_dir+'/prep_'+str(n)+'.sh'

    def update_dir_file(self, num_dirs):
        """

        Create a list of RF_dirs for the prep_script to use.

        :param int num_dirs: number of RF_dirs to put in ``dir_list``

        """
        with open(self.save_dir+'/dir_list', 'w') as f:
            for i in xrange(num_dirs-1):
                f.write(self.rf_dirs[i]+'\n')
            f.write(self.rf_dirs[num_dirs-1])

    def save(self, mdict, save_file):
        """
        Save matrices to a ``*.mat`` file for use by ``MATLAB BET`` code and
        :meth:`~polysim.run_framework.random_manningsn.loadmat`

        :param dict() mdict: dictonary of run data
        :param string save_file: file name

        """
        sio.savemat(self.save_dir+'/'+save_file, mdict, do_compression=True)

    def update_mdict(self, mdict):
        """
        Set up references for ``mdict``

        :param dict() mdict: dictonary of run data

        """

        # export nontimeseries data
        for k, v in self.nts_data.iteritems():
            mdict[k] = v
        # export timeseries data
        for k, v in self.ts_data.iteritems():
            mdict[k] = v
        # export time_obs data
        for k, v in self.time_obs.iteritems():
            mdict[k+'_time'] = v

    def concatenate(self, other_run, points1, points2):
        """
        Combine data from this
        :class:`~polysim.run_framework.random_manningsn.runSet` with another
        :class:`~polysim.run_framework.random_manningsn.runSet` (``other_run``)
        and points from both runs

        :param: other_run
        :type other_run: :class:`~polysim.run_framework.random_manningsn.runSet`
        :param points1: sample points for ``self``
        :type points1: np.array
        :param points1: sample points for ``other_run``
        :type points1: np.array
        :returns: (self, points)
        :rtype: tuple

        """

        return concatenate((self, points1), (other_run, points2))

    def run_points(self, data, points, save_file, num_procs=12, procs_pnode=12,
            ts_names=["fort.61"], nts_names=["maxele.63"], screenout=True,
            cleanup_dirs=True, num_writers=None, TpN=12):
        """
        Runs :program:`ADCIRC` for all of the configurations specified by
        ``points`` and returns a dictonary of arrays containing data from
        output files

         Reads in a default Manning's *n* value from self.save_dir and stores
         it in data.manningsn_default
        :param data: :class:`~polysim.run_framework.domain`
        :type points: :class:`np.array` of size (``num_of_basis_vec``,
            ``num_of_random_fields``)
        :param points: containts the weights to be used for each run
        :type save_file: string
        :param save_file: name of file to save ``station_data`` to
        :type num_procs: int or 12
        :param num_procs: number of processors per :program:`ADCIRC`
            simulation
        :param int procs_pnode: number of processors per node, 12 on lonestar,
            and 16 on stampede
        :param list() ts_names: names of ADCIRC timeseries
            output files to be recorded from each run
        :param list() nts_names: names of ADCIRC non timeseries
            output files to be recorded from each run
        :param boolean screenout: flag (True --  write ``ADCIRC`` output to
            screen, False -- write ``ADCIRC`` output to temp file
        :param boolean cleanup_dirs: flag to delete all RF_dirs after run (True
            -- yes, False -- no)
        :param int num_writers: number of MPI processes to dedicate soley to
            the task of writing ascii files. This MUST be < num_procs
        :param int TpN: number of tasks (cores to use) per node (wayness)
        :rtype: (:class:`np.array`, :class:`np.ndarray`, :class:`np.ndarray`)
        :returns: (``time_obs``, ``ts_data``, ``nts_data``)

        .. note:: Currently supports ADCIRC output files ``fort.6*``,
                  ``*.63``, ``fort.7*``, but NOT Hot Start Output
                  (``fort.67``, ``fort.68``)

        """
        # setup and save to shelf
        # set up saving
        if glob.glob(self.save_dir+'/'+save_file):
            os.remove(self.save_dir+'/'+save_file)

        # Save matricies to *.mat file for use by MATLAB or Python
        mdict = dict()
        mdict['mann_pts'] = points
        self.save(mdict, save_file)

        #bv_array = tmm.get_basis_vec_array(self.basis_dir)
        bv_dict = tmm.get_basis_vectors(self.basis_dir)

        # Pre-allocate arrays for various data files
        num_points = points.shape[1]
        # Pre-allocate arrays for non-timeseries data
        nts_data = {}
        self.nts_data = nts_data
        for fid in nts_names:
            key = fid.replace('.', '')
            nts_data[key] = np.zeros((data.node_num, num_points))
        # Pre-allocate arrays for timeseries data
        ts_data = {}
        time_obs = {}
        self.ts_data = ts_data
        self.time_obs = time_obs
        for fid in ts_names:
            key = fid.replace('.', '')
            meas_locs, total_obs, irtype = data.recording[key]
            if irtype == 1:
                ts_data[key] = np.zeros((meas_locs, total_obs, num_points))
            else:
                ts_data[key] = np.zeros((meas_locs, total_obs,
                    irtype, num_points))
            time_obs[key] = np.zeros((total_obs,))

        # Update and save
        self.update_mdict(mdict)
        self.save(mdict, save_file)

        default = data.read_default(path=self.save_dir)

        for k in xrange(0, num_points, self.num_of_parallel_runs):
            if k+self.num_of_parallel_runs >= num_points-1:
                stop = num_points
                step = stop-k
            else:
                stop = k+self.num_of_parallel_runs
                step = self.num_of_parallel_runs
            run_script = self.write_run_script(num_procs, step, procs_pnode,
                    TpN, screenout, num_writers)
            self.write_prep_script(5, step)
            for i in xrange(0, step):
                # generate the Manning's n field
                r_field = tmm.combine_basis_vectors(points[..., i+k], bv_dict,
                          default, data.node_num)
                # create the fort.13 for r_field
                f13.update_mann(r_field, self.rf_dirs[i])
            # do a batch run of python
            #PARALLEL: update file containing the list of rf_dirs
            self.update_dir_file(self.num_of_parallel_runs)
            devnull = open(os.devnull, 'w')
            p = subprocess.Popen(['./prep_5.sh'], stdout=devnull, cwd=
                    self.save_dir)
            p.communicate()
            devnull.close()
            devnull = open(os.devnull, 'w')
            p = subprocess.Popen(['./'+run_script], stdout=devnull, cwd=
                    self.base_dir)
            p.communicate()
            devnull.close()
            # get data
            for i, kk in enumerate(range(k, stop)):
                output.get_data_ts(kk, self.rf_dirs[i], self.ts_data, time_obs,
                        ts_names)
                output.get_data_nts(kk, self.rf_dirs[i], data, self.nts_data,
                        nts_names)
            # Update and save
            self.update_mdict(mdict)
            self.save(mdict, save_file)

        # save data
        self.update_mdict(mdict)
        self.save(mdict, save_file)

        if cleanup_dirs:
            self.remove_random_field_directories()

        return time_obs, ts_data, nts_data

    def make_plots(self, points, domain, save=True, show=False,
            bathymetry=False):
        """
        Plots ``mesh``, ``station_locations``, ``basis_functions``,
        ``random_fields``, ``mean_field``, ``station_data``, and
        save in save_dir/figs

        .. todo:: this uses bv_array everywhere. I might want to change this
                  later when I go to the nested mesh approach

        """
        mkdir(self.save_dir+'/figs')
        domain.get_Triangulation(self.save_dir, save, show)
        domain.plot_bathymetry(self.save_dir, save, show)
        domain.plot_station_locations(self.save_dir, bathymetry, save, show)

        bv_array = tmm.get_basis_vec_array(self.basis_dir)

        self.plot_basis_functions(domain, bv_array, save, show)
        self.plot_random_fields(domain, points, bv_array, save, show)

        self.plot_mean_field(domain, points, bv_array, save, show)
        self.plot_station_data(save, show)

    def plot_basis_functions(self, domain, bv_array, save=True, show=False):
        """
        See :meth:``~polsim.pyADCIRC.plotADCIRC.basis_functions`

        """
        plot.basis_functions(domain, bv_array, self.save_dir, save, show)

    def plot_random_fields(self, domain, points, bv_array, save=True, show=
            False):
        """
        See :meth:`~polsim.rnu_framework.plotADCIRC.random_fields`

        """
        plot.random_fields(domain, points, bv_array, self.save_dir, save, show)

    def plot_mean_field(self, domain, points, bv_array, save=True, show=
        False):
        """
        See :meth:`~polsim.rnu_framework.plotADCIRC.mean_field`

        """
        plot.mean_field(domain, points, bv_array, self.save_dir, save, show)

    def plot_station_data(self, save=True, show=False):
        """
        See :meth:`~polsim.rnu_framework.plotADCIRC.station_data`

        """
        plot.station_data(self.ts_data, self.time_obs, None, self.save_dir,
                save, show)

    def fix_dry_data(self, data):
        """
        Fix dry elevation station data flags

        :param data: :class:`~polysim.run_framework.domain`

        """
        self.ts_data = fix_dry_data(self.ts_data, data)

    def fix_dry_nodes(self, data):
        """
        Fix dry elevation data flags

        :param data: :class:`~polysim.run_framework.domain`

        """
        self.ts_data = fix_dry_nodes(self.ts_data, data)

    def fix_dry_nodes_nts(self, data):
        """
        Fix dry elevation data flags

        :param data: :class:`~polysim.run_framework.domain`

        """
        self.nts_data = fix_dry_nodes_nts(self.nts_data, data)

    def convert_to_hours(self):
        """
        Converts ``self.time_obs`` from seconds to hours

        """
        self.time_obs = convert_to_hours(self.time_obs)

    def convert_to_days(self):
        """
        Converts ``self.time_obs`` from seconds to days

        """
        self.time_obs = convert_to_days(self.time_obs)

    def convert_to_percent(self, data):
        """
        Converts ``self.nts_data['tinun63']`` from seconds to percent of RNDAY

        :param data: :class:`~polysim.run_framework.domain`

        """
        convert_to_percent(self.nts_data, data)




