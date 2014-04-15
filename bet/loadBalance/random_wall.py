# Lindley Graham 4/10/2013
"""
This module contains functions to pull data from ADCIRC output files and the
:class:`runSet` which controls the running of ADCIRC simulations within a set
of processors allocated by the submission script
"""
import numpy as np
import glob, os, subprocess, shutil 
import polysim.pyADCIRC.fort13_management as f13
import polysim.pyADCIRC.fort14_management as f14
import polysim.run_framework.random_manningsn as rmn
import polysim.pyGriddata.table_to_mesh_map as tmm
import polysim.pyADCIRC.plotADCIRC as plot
import polysim.pyADCIRC.output as output
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
    :rtype: tuple of :class:`~polysim.run_framework.random_wall.runSet`,
        :class:`~polysim.run_framework.random_manningsn.domain` objects, and
        two :class:`numpy.array`s
    :returns: (main_run, domain, mann_pts, wall_pts)

    """
    main_run, domain, mann_pts = rmn.loadmat(save_file, base_dir, grid_dir,
            save_dir, basis_dir)
    
       # load the data from at *.mat file
    mdat = sio.loadmat(save_dir+'/'+save_file)
    if mdat.has_key('wall_pts'):
        wall_pts = mdat['wall_pts']
    else:
        wall_pts = None
    points = mdat['points']
    
    return (main_run, domain, mann_pts, wall_pts, points)

class runSet(rmn.runSet):
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
            num_of_parallel_runs = 10, base_dir = None, script_name = None): 
        """
        Initialization
        """
        super(runSet, self).__init__(grid_dir, save_dir, basis_dir, 
            num_of_parallel_runs, base_dir, script_name)
        
    def run_points(self, data, wall_points, mann_points, save_file, 
            num_procs = 12, procs_pnode = 12, ts_names = ["fort.61"], 
            nts_names = ["maxele.63"], screenout = True, s_p_wall = None,
            num_writers = None, TpN = 12):
        """
        Runs :program:`ADCIRC` for all of the configurations specified by
        ``wall_points`` and ``mann_points`` and returns a dictonary of arrays
        containing data from output files. Assumes that the number of
        ``wall_points`` is less than the number of ``mann_points``. Runs
        batches of :program:`PADCIRC` as a double for loop with the
        :program:`ADCPREP` prepping the ``fort.14`` file on the exterior loop
        and the ``fort.13`` file on the interior loop.

         Reads in a default Manning's *n* value from self.save_dir and stores
         it in data.manningsn_default                                                                   
        :param data: :class:`~polysim.run_framework.domain`
        :type wall_points: :class:`np.array` of size (5, ``num_of_walls``)
        :param wall_points: containts the box_limits, and wall_height for each
            wall [ximin, xmax, ymin, ymax, wall_height]
        :type mann_points: :class:`np.array` of size (``num_of_basis_vec``,
            ``num_of_random_fields``), ``num_of_random_fields`` MUST be a
            multiple of ``num_of_walls``. The ith wall will be associated with
            the ith set of i*(num_of_random_fields/num_of_walls) mann_points
        :param mann_points: containts the weights to be used for each run
        :type save_file: string
        :param save_file: name of file to save mdict to 
        :type num_procs: int or 12
        :param num_procs: number of processors per :program:`ADCIRC`
            simulation, 12 on lonestar, and 16 on stamped
        :param int procs_pnode: number of processors per node
        :param list() ts_names: names of ADCIRC timeseries
            output files to be recorded from each run
        :param list() nts_names: names of ADCIRC non timeseries
            output files to be recorded from each run
        :param boolean screenout: flag (True --  write ``ADCIRC`` output to
            screen, False -- write ``ADCIRC`` output to temp file
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
        mdict['mann_pts'] = mann_points 
        mdict['wall_pts'] = wall_points 
 
        self.save(mdict, save_file)

        #bv_array = tmm.get_basis_vec_array(self.basis_dir)
        bv_dict = tmm.get_basis_vectors(self.basis_dir)

        # Pre-allocate arrays for various data files
        num_points = mann_points.shape[1]
        num_walls = wall_points.shape[1]
        if s_p_wall == None:
            s_p_wall = num_points/num_walls*np.ones(num_walls, dtype = int)
       
        # store the wall points with the mann_points as points
        mdict['points'] = np.vstack((np.repeat(wall_points, s_p_wall,1),
            mann_points))

        # Pre-allocate arrays for non-timeseries data
        nts_data = {}
        self.nts_data = nts_data
        for fid in nts_names:
            key = fid.replace('.','')
            nts_data[key] =  np.zeros((data.node_num, num_points))        
        # Pre-allocate arrays for timeseries data
        ts_data = {}
        time_obs = {}
        self.ts_data = ts_data
        self.time_obs = time_obs
        for fid in ts_names:
            key = fid.replace('.','')
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

        default = data.read_default(path = self.save_dir)

        for w in xrange(num_walls):
            # set walls
            wall_dim = wall_points[..., w]
            data.read_spatial_grid()
            data.add_wall(wall_dim[:4], wall_dim[-1])
            # update wall and prep all
            for rf_dir in self.rf_dirs:
                os.remove(rf_dir+'/fort.14')
                shutil.copy(self.grid_dir+'/fort.14', rf_dir)
                f14.update(data, path = rf_dir)
            #PARALLEL: update file containing the list of rf_dirs
            self.update_dir_file(self.num_of_parallel_runs)
            devnull = open(os.devnull, 'w')
            p = subprocess.Popen(['./prep_2.sh'], stdout = devnull, cwd = self.save_dir)
            p.communicate()
            devnull.close()
            #for k in xrange(w*s_p_wall, (w+1)*s_p_wall-1, self.num_of_parallel_runs):
            for k in xrange(sum(s_p_wall[:w]), sum(s_p_wall[:w+1]), self.num_of_parallel_runs):
                if k+self.num_of_parallel_runs >= num_points-1:
                    stop = num_points
                    step = stop-k
                else:
                    stop = k+self.num_of_parallel_runs
                    step = self.num_of_parallel_runs
                run_script = self.write_run_script(num_procs, step,
                        procs_pnode, TpN, screenout, num_writers)
                self.write_prep_script(5, step)
                for i in xrange(0, step):
                    # generate the Manning's n field
                    r_field = tmm.combine_basis_vectors(mann_points[..., i+k], bv_dict,
                              default, data.node_num)
                    # create the fort.13 for r_field
                    f13.update_mann(r_field, self.rf_dirs[i])
                # do a batch run of python
                #PARALLEL: update file containing the list of rf_dirs
                self.update_dir_file(self.num_of_parallel_runs)
                devnull = open(os.devnull, 'w')
                p = subprocess.Popen(['./prep_5.sh'], stdout = devnull, cwd = self.save_dir)
                p.communicate()
                devnull.close()
                devnull = open(os.devnull, 'w')
                p = subprocess.Popen(['./'+run_script], stdout = subprocess.PIPE,
                        cwd = self.base_dir) 
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

        return time_obs, ts_data, nts_data

    def run_nobatch(self, data, wall_points, mann_points, save_file, 
            num_procs = 12, procs_pnode = 12, ts_names = ["fort.61"], 
            nts_names = ["maxele.63"], screenout = True, 
            num_writers = None, TpN = 12):
        """
        Runs :program:`ADCIRC` for all of the configurations specified by
        ``wall_points`` and ``mann_points`` and returns a dictonary of arrays
        containing data from output files. Runs batches of :program:`PADCIRC`
        as a single for loop and preps both the ``fort.13`` and fort.14`` in
        the same step.

         Reads in a default Manning's *n* value from self.save_dir and stores
         it in data.manningsn_default                                                                   
        :param data: :class:`~polysim.run_framework.domain`
        :type wall_points: :class:`np.array` of size (5, ``num_of_walls``)
        :param wall_points: containts the box_limits, and wall_height for each
            wall [ximin, xmax, ymin, ymax, wall_height]
        :type mann_points: :class:`np.array` of size (``num_of_basis_vec``,
            ``num_of_random_fields``), ``num_of_random_fields`` MUST be the
            same as ``num_of_walls``. The ith wall will be associated with
            the ith field specifed by mann_points
        :param mann_points: containts the weights to be used for each run
        :type save_file: string
        :param save_file: name of file to save mdict to 
        :type num_procs: int or 12
        :param num_procs: number of processors per :program:`ADCIRC`
            simulation, 12 on lonestar, and 16 on stamped
        :param int procs_pnode: number of processors per node
        :param list() ts_names: names of ADCIRC timeseries
            output files to be recorded from each run
        :param list() nts_names: names of ADCIRC non timeseries
            output files to be recorded from each run
        :param boolean screenout: flag (True --  write ``ADCIRC`` output to
            screen, False -- write ``ADCIRC`` output to temp file
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
        mdict['mann_pts'] = mann_points 
        mdict['wall_pts'] = wall_points 
 
        self.save(mdict, save_file)

        #bv_array = tmm.get_basis_vec_array(self.basis_dir)
        bv_dict = tmm.get_basis_vectors(self.basis_dir)

        # Pre-allocate arrays for various data files
        num_points = mann_points.shape[1]
        num_walls = wall_points.shape[1]
        if num_points != num_points:
            print "Error: num_walls != num_points"
            quit()

        # store the wall points with the mann_points as points
        mdict['points'] = np.vstack((wall_points, mann_points))

        # Pre-allocate arrays for non-timeseries data
        nts_data = {}
        self.nts_data = nts_data
        for fid in nts_names:
            key = fid.replace('.','')
            nts_data[key] =  np.zeros((data.node_num, num_points))        
        # Pre-allocate arrays for timeseries data
        ts_data = {}
        time_obs = {}
        self.ts_data = ts_data
        self.time_obs = time_obs
        for fid in ts_names:
            key = fid.replace('.','')
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

        default = data.read_default(path = self.save_dir)

        for k in xrange(0, num_points, self.num_of_parallel_runs):
            if k+self.num_of_parallel_runs >= num_points-1:
                stop = num_points
                step = stop-k
            else:
                stop = k+self.num_of_parallel_runs
                step = self.num_of_parallel_runs
            run_script = self.write_run_script(num_procs, step,
                    procs_pnode, TpN, screenout, num_writers)
            self.write_prep_script(5, step)
            # set walls
            wall_dim = wall_points[..., k]
            data.read_spatial_grid()
            data.add_wall(wall_dim[:4], wall_dim[-1])
            # update wall and prep all
            for rf_dir in self.rf_dirs:
                os.remove(rf_dir+'/fort.14')
                shutil.copy(self.grid_dir+'/fort.14', rf_dir)
                f14.update(data, path = rf_dir)
            #PARALLEL: update file containing the list of rf_dirs
            self.update_dir_file(self.num_of_parallel_runs)
            devnull = open(os.devnull, 'w')
            p = subprocess.Popen(['./prep_2.sh'], stdout = devnull, cwd = self.save_dir)
            p.communicate()
            devnull.close()
            for i in xrange(0, step):
                # generate the Manning's n field
                r_field = tmm.combine_basis_vectors(mann_points[..., i+k], bv_dict,
                          default, data.node_num)
                # create the fort.13 for r_field
                f13.update_mann(r_field, self.rf_dirs[i])
            # do a batch run of python
            #PARALLEL: update file containing the list of rf_dirs
            self.update_dir_file(self.num_of_parallel_runs)
            devnull = open(os.devnull, 'w')
            p = subprocess.Popen(['./prep_5.sh'], stdout = devnull, cwd = self.save_dir)
            p.communicate()
            devnull.close()
            devnull = open(os.devnull, 'w')
            p = subprocess.Popen(['./'+run_script], stdout = devnull,
                    cwd = self.base_dir) 
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

        return time_obs, ts_data, nts_data
    
    def make_plots(self, wall_points, mann_points, domain, save = True, show = False, 
                   bathymetry = False):
        """
        Plots ``mesh``, ``station_locations``, ``basis_functions``,
        ``random_fields``, ``mean_field``, ``station_data``, and
        save in save_dir/figs 
        
        .. todo:: this uses bv_array everywhere. I might want to change this
                  later when I go to the nested mesh approach

        """
        super(runSet, self).make_plots(mann_points, domain, save, show, 
                   bathymetry)
        self.plot_walls(domain, wall_points, save, show)

    def plot_walls(self, domain, wall_points, save = True, 
        show = False):
        """
        Plots the walls with dimenstions described in ``wall_points`` and saves
        the plots in ``self.save_dir``
        
        :param domain: :class:`~polysim.run_framework.domain`
        :param wall_points: containts the box_limits, and wall_height for each
            wall [ximin, xmax, ymin, ymax, wall_height]
        :type wall_points: :class:`np.array` of size (5, ``num_of_walls``)
        :param boolean save: flag for whether or not to save plots
        :param boolean show: flag for whether or not to show plots

        """
        plot_walls(self, domain, wall_points, save, show)

        
def plot_walls(run_set, domain, wall_points, save = True, 
        show = False):
    """
    Plots the walls with dimenstions described in ``wall_points`` and saves
    the plots in ``self.save_dir``
    
    :param domain: :class:`~polysim.run_framework.domain`
    :param wall_points: containts the box_limits, and wall_height for each
        wall [ximin, xmax, ymin, ymax, wall_height]
    :type wall_points: :class:`np.array` of size (5, ``num_of_walls``)
    :param boolean save: flag for whether or not to save plots
    :param boolean show: flag for whether or not to show plots

    """
    num_walls = wall_points.shape[1]
    for w in xrange(num_walls):
        # set walls
        wall_dim = wall_points[..., w]
        domain.read_spatial_grid()
        domain.add_wall(wall_dim[:4], wall_dim[-1])
        plot.bathymetry(domain, run_set.save_dir, save, show)

