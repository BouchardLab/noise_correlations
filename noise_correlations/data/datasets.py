import os, glob

import numpy as np

from .drifting_bar import ori, sweeplist

"""Based on Pratik Sachdeva's notebook."""

class BlancheCRCNSpvc3_cat(object):
    def __init__(self, folder):
        spike_path = os.path.join(folder, 'spike_data')
        stimulus_file = os.path.join(folder, 'stimulus_data/drifting_bar.din')
        spike_files = glob.glob(os.path.join(spike_path, 't*.spk'))
        self.neurons = np.array([os.path.splitext(os.path.split(f)[1])[0] for f
                                 in spike_files])
        self.angles = np.array(ori, dtype=int)
        self.trial_labels = np.array(sweeplist, dtype=int)
        self.spike_times = {neuron: np.fromfile(f, dtype='int64') for neuron, f
                            in zip(self.neurons, spike_files)}
        stim_array = np.fromfile(stimulus_file, dtype='int64')
        self.stim_times = stim_array[0::2]
        self.stim_labels = stim_array[1::2]


    def trial_design_matrices(self):
        """Return the total spike counts per trial.
        stimuli.

        Returns
        -------
        trial_angles : ndarray (n_trial)
            Angles for each bin.
        X_1h : ndarray (n_trial, n_angles)
            One-hot vector for the angle in each trial.
        X_cos : ndarray (n_trial, 2)
            Cosine and sine of the angle for each trial.
        Y : ndarray (n_trials, n_neurons)
            Spike count for each trial for each neuron."""

        n_trials = self.trial_labels.size
        n_neurons = self.neurons.size
        n_angles = self.angles.size
        # Transition indices
        transition_idxs = np.argwhere(self.stim_labels[1:] !=
                                      self.stim_labels[:-1]).ravel() + 1
        transition_idxs = np.insert(transition_idxs, 0, 0)
        transition_idxs = np.append(transition_idxs, self.stim_labels.size)

        # Trial boundaries
        stim_time_bounds = np.zeros((n_trials, 2))
        for idx in range(transition_idxs.size - 1):
            bounds = [self.stim_times[transition_idxs[idx]],
                      self.stim_times[transition_idxs[idx+1]-1]]
            stim_time_bounds[idx] = bounds

        # design matrices
        trial_angles = -1 * np.ones(n_trials, dtype=int)
        X_1h = np.zeros((n_trials, n_angles), dtype=int)
        X_cos = np.zeros((n_trials, 2))

        # response matrix
        Y = -1 * np.ones((n_trials, n_neurons), dtype=int)

        # iterate over trials
        for trial_idx, time_bound in enumerate(stim_time_bounds):

            # extract angle for current trial
            stim_idx = self.trial_labels[trial_idx]
            angle = self.angles[stim_idx]
            trial_angles[trial_idx] = angle
            X_1h[trial_idx, stim_idx] = 1
            X_cos[trial_idx] = [np.cos(np.deg2rad(angle)),
                                np.sin(np.deg2rad(angle))]

            # response matrix
            Y[trial_idx, :] = [np.count_nonzero(
                (self.spike_times[neuron] >= time_bound[0]) &
                (self.spike_times[neuron] < time_bound[1])
            ) for neuron in self.neurons]
        return trial_angles, X_1h, X_cos, Y


    def bin_spikes(self, bin_ms):
        """Return the total spike counts per bin. No bins will have mixed
        stimuli.

        Parameters
        ----------
        bin_ms : float
            Bin length in milliseconds.

        Returns
        -------
        bin_angles : ndarray (n_bins)
            Angles for each bin.
        X_1h : ndarray (n_bins, n_angles)
            One-hot vector for the angle in each bin.
        X_cos : ndarray (n_bins, 2)
            Cosine and sine of the angle for each bin.
        Y : ndarray (n_bins, n_neurons)
            Spike count for each bin for each neuron."""

        n_trials = self.trial_labels.size
        n_neurons = self.neurons.size
        n_angles = self.angles.size
        # Transition indices
        start_idxs = np.argwhere(self.stim_labels[1:] !=
                                 self.stim_labels[:-1]).ravel() + 1
        start_idxs = np.insert(start_idxs, 0, 0)
        start_idxs = np.append(start_idxs, self.stim_labels.size)

        bins_per_trial = np.zeros(n_trials, dtype=int)
        for ii, idx in enumerate(start_idxs[:-1]):
            start_t = self.stim_times[idx]
            if ii < start_idxs.size - 2:
                end_t = self.stim_times[start_idxs[ii + 1]]
            else:
                end_t = self.stim_times[-1]
            dt_ms = (end_t - start_t) / 1e3
            bins_per_trial[ii] = int(np.floor(dt_ms / bin_ms))
        n_bins = bins_per_trial.sum()

        # design matrices
        bin_angles = -1 * np.ones(n_bins, dtype=int)
        X_1h = np.zeros((n_bins, n_angles), dtype=int)
        X_cos = np.zeros((n_bins, 2))

        # response matrix
        Y = -1 * np.ones((n_bins, n_neurons), dtype=int)
        bin_idx = 0
        for ii, idx in enumerate(start_idxs[:-1]):
            stim_label = self.stim_labels[idx]
            angle = self.angles[stim_label]

            start_t = self.stim_times[idx]
            end_t = start_t + bin_ms * 1e3
            stim_times = np.append(self.stim_times, self.stim_times[-1] + 1)
            while ((end_t < self.stim_times[-1]) and
                   (end_t < stim_times[start_idxs[ii + 1]])):
                Y[bin_idx, :] = [np.count_nonzero(
                    (self.spike_times[neuron] >= start_t) &
                    (self.spike_times[neuron] < end_t)
                ) for neuron in self.neurons]
                bin_angles[bin_idx] = angle
                X_1h[bin_idx, stim_label] = 1
                X_cos[bin_idx] = [np.cos(np.deg2rad(angle)),
                                  np.sin(np.deg2rad(angle))]
                start_t = end_t
                end_t = start_t + bin_ms * 1e3
                bin_idx += 1
                #print(end_t, self.stim_times[-1])

        return bin_angles, X_1h, X_cos, Y
