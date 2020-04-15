import numpy as np
import bet.sample as sample
import bet.util as util
from luq.luq import LUQ


def myModel(inputs, times):
    from luq.dynamical_systems import Selkov
    ics = np.ones(inputs.shape)
    # Solve systems
    phys = Selkov()
    return phys.solve(ics=ics, params=inputs, t_eval=times)


class useLUQ:
    def __init__(self, predict_set, obs_set, lb_model, times):
        self.predict_set = predict_set
        self.obs_set = obs_set
        self.lb_model = lb_model
        self.times = times
        self.predicted_time_series = None
        self.obs_time_series = None
        self.learn = None

    def save(self, savefile):
        util.save_object(save_set=self, file_name=savefile, globalize=True)

    def get_predictions(self):
        self.predicted_time_series = self.lb_model(self.predict_set.get_values(), self.times)

    def get_obs(self):
        self.obs_time_series = self.lb_model(self.obs_set.get_values(), self.times)

    def setup(self):
        self.get_predictions()
        self.get_obs()
        self.learn = LUQ(self.predicted_time_series, self.obs_time_series, self.times)

    def clean_data(self, **kwargs):
        self.learn.clean_data(**kwargs)

    def dynamics(self, **kwargs):
        self.learn.dynamics(**kwargs)

    def learn_qois_and_transform(self, **kwargs):
        self.learn.learn_qois_and_transform(**kwargs)

    def make_disc(self):
        out_dim = self.learn.num_pcs[0]
        out_num_predict = self.learn.predicted_time_series.shape[0]
        out_num_obs = self.learn.observed_time_series.shape[0]

        predict_output = sample.sample_set(out_dim)
        predict_output.set_region_local(self.learn.predict_labels)
        predict_output.set_cluster_maps(self.learn.predict_maps)

        obs_output = sample.sample_set(out_dim)
        obs_output.set_region_local(self.learn.obs_labels)
        obs_output.set_cluster_maps(self.learn.obs_maps)

        disc1 = sample.discretization(input_sample_set=self.predict_set,
                                      output_sample_set=predict_output,
                                      output_probability_set=obs_output)

        disc2 = sample.discretization(input_sample_set=self.obs_set,
                                      output_sample_set=obs_output)

        return disc1, disc2

    def local_to_global(self):
        pass







