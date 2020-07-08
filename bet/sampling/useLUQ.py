# Copyright (C) 2014-2020 The BET Development Team
"""
The module contains a class for interfacing between BET and LUQ.
"""

import numpy as np
import bet.sample as sample
import bet.util as util
import logging

class missing_module(Exception):
    """
    Exception for when a module cannot be imported.
    """

def myModel(inputs, times):
    """
    Example for interfacing a time series model with LUQ.
    :param inputs: Parameter values at which to evaluate the model.
    :type inputs: :class:`numpy.ndarray` of shape (num_inputs, num_params)
    :param times: Times at which to output results.
    :type times: :class:`numpy.ndarray` of shape (num_times, )
    :return: Time series data
    :rtype: :class:`numpy.ndarray` of shape (num_inputs, num_times)
    """
    try:
        from luq.dynamical_systems import Selkov
    except ImportError:
        raise missing_module("luq cannot be imported")
    ics = np.ones(inputs.shape)
    # Solve systems
    phys = Selkov()
    return phys.solve(ics=ics, params=inputs, t_eval=times)


class useLUQ:
    """
    Wrappers for interfacing BET with LUQ. Allows for the simple creation of `bet.sample.discretization` objects
    from LUQ output.
    """

    def __init__(self, predict_set, lb_model, times, obs_set=None):
        """
        Initialize the object.
        :param predict_set: Sample set defining input prediction samples.
        :type predict_set: :class:`bet.sample.sample_set`
        :param obs_set: Sample set defining input observation samples.
        :type obs_set: :class:`bet.sample.sample_set`
        :param lb_model: Interface to a time-dependent model takes an input of an array of parameter values and an array
        of times for evaluation as arguments. See an example with `myModel`, above.
        :param times: Times at which to output the model.
        :type times: :class:`numpy.ndarray` with shape (num_times, )
        """

        self.predict_set = predict_set
        self.obs_set = obs_set
        self.lb_model = lb_model
        self.times = times
        self.predicted_time_series = None
        self.obs_time_series = None
        self.learn = None

    def save(self, savefile):
        """
        Save the object to a Pickle file.
        :param savefile: Name of file to save to.
        :type savefile: str
        """
        util.save_object(save_set=self, file_name=savefile, globalize=True)

    def get_predictions(self):
        """
        Evaluate the model for the predicted time series.
        """
        self.predicted_time_series = self.lb_model(self.predict_set.get_values(), self.times)

    def get_obs(self):
        """
        Evaluate the model for the predicted time series.
        """
        self.obs_time_series = self.lb_model(self.obs_set.get_values(), self.times)

    def set_observed_time_series(self, obs_time_series):
        """
        Set observed time series data manually.
        :param obs_time_series: time series data
        :type obs_time_series:
        :return: :class:`numpy.ndarray` with shape (num_obs, num_times)
        """
        self.obs_time_series = obs_time_series

    def initialize(self, predicted_time_series=None, obs_time_series=None, times=None):
        """
        Initialize the LUQ object. This can be used manually if time series are pre-computed.

        :param predicted_time_series: Time series solutions for predicted values.
        :type predicted_time_series: :class:`numpy.ndarray` of shape (num_predict_samples, num_times)
        :param obs_time_series: Time series solutions for predicted values.
        :type obs_time_series: :class:`numpy.ndarray` of shape (num_obs_samples, num_times)
        :param times: Times at which the series are output.
        :type times: :class:`numpy.ndarray` with shape (num_times, )
        """
        try:
            from luq.luq import LUQ
        except ImportError:
            raise missing_module("luq cannot be imported")

        if predicted_time_series is None:
            predicted_time_series = self.predicted_time_series
        if obs_time_series is None:
            obs_time_series = self.obs_time_series
        if times is None:
            times = self.times

        self.learn = LUQ(predicted_time_series, obs_time_series, times)

    def setup(self):
        """
        Setup LUQ object all at once.
        """
        self.get_predictions()
        self.get_obs()
        self.initialize(self.predicted_time_series, self.obs_time_series, self.times)

    def clean_data(self, **kwargs):
        """
        Wrapper for `luq.luq.LUQ.clean_data`
        """
        self.learn.clean_data(**kwargs)

    def dynamics(self, **kwargs):
        """
        Wrapper for `luq.luq.LUQ.dynamics`
        """
        self.learn.dynamics(**kwargs)

    def learn_qois_and_transform(self, **kwargs):
        """
        Wrapper for `luq.luq.LUQ.learn_qois_and_transform`
        """
        self.learn.learn_qois_and_transform(**kwargs)

    def make_disc(self):
        """
        Construct `bet.sample.discretization` objects for predict and obs sets.
        :return: predict_disc, obs_disc
        :rtype: `bet.sample.discretization`, `bet.sample.discretization` or None if no observation set.
        """
        out_dim = self.learn.num_pcs[0]

        predict_output = sample.sample_set(out_dim)
        predict_output.set_region_local(self.learn.predict_labels)
        predict_output.set_cluster_maps(self.learn.predict_maps)

        obs_output = sample.sample_set(out_dim)
        obs_output.set_region_local(self.learn.obs_labels)
        obs_output.set_cluster_maps(self.learn.obs_maps)

        # Prediction discretization
        disc1 = sample.discretization(input_sample_set=self.predict_set,
                                      output_sample_set=predict_output,
                                      output_observed_set=obs_output)
        disc1.local_to_global()

        # Observation discretization
        if self.obs_set is None:
            disc2 = None
        else:
            disc2 = sample.discretization(input_sample_set=self.obs_set,
                                          output_sample_set=obs_output)
            disc2.local_to_global()

        return disc1, disc2

    def local_to_global(self):
        """
        Dummy function for saving.
        """
        pass







