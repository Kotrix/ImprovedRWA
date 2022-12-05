import numpy as np


class RWA:
    """
    Implementation of Improved Robust Weighted Averaging described in [1].
    All parameters are set to default as proposed in [1] and they should be fine in most cases for EEG signals.
    Consider adjusting parameters when working with other types of signals (e.g., MEG, ECG, etc.)

    [1] Kotowski K., Stapor K., Leski J. Improved robust weighted averaging for event-related potentials in EEG
     https://doi.org/10.1016/j.bbe.2019.09.002

    :param ksi: preset parameter in the stopping condition of the algorithm. The algorithm is stopped when the norm of
        the difference between current and previous weights is less than ksi.
    :param m: weighting exponent parameter. Generally, a larger 'm' results in a smaller influence of the dissimilarity
        measures. Must be in range (1, inf).
    :param max_iter: maximum number of iterations
    :param min_sigma: a small positive constant added to the value of dissimilarity metric to avoid division by zero.
        Default value set to a value of least significant bit of the digital resolution of the BioSemi EEG amplifier.
    :param c: sensitivity parameter for rejection of outlying epochs. The epochs with weights smaller than 1/(c * N)
        will be rejected, where N is number of epochs.
    :parama init_v: initial value of the result array. If None, result is initialized with samplewise median as in [1].
    """
    def __init__(self, ksi: float = 1e-5, m: float = 2.0, max_iter: int = 1000, min_sigma: float = 31e-9,
                 c: float = 100, init_v: np.array = None):
        self.done_iter = 0
        self.W = None

        assert m > 1
        self._m = m

        assert max_iter > 1
        self._max_iter = max_iter

        assert ksi > 0
        self._ksi = ksi

        assert min_sigma > 0
        self._min_sigma = min_sigma

        assert c > 0
        self._c = c

        self._init_v = init_v

    @staticmethod
    def _compute_v(w: np.array, channel_data: np.array) -> np.array:
        return np.expand_dims(w.dot(channel_data) / np.sum(w), 0)

    @staticmethod
    def _samplewise_median(channel_data: np.array) -> np.array:
        return np.expand_dims(np.median(channel_data, axis=0), 0)

    def run(self, mne_epochs_data: np.array):
        """
        Calculate Improved RWA for EEG data from mne.Epochs object
        :param mne_epochs_data: epochs data as stored in mne.Epochs object
        :return: robustly weighted evoked signal
        """
        number_of_epochs, number_of_channels, signal_length = np.shape(mne_epochs_data)

        result = np.zeros((number_of_channels, signal_length))
        for channel_index in range(number_of_channels):
            channel_data = mne_epochs_data[:, channel_index, :].reshape(number_of_epochs, signal_length)
            result[channel_index, :] = self._run_for_one_channel(channel_data)
        return result

    def _run_for_one_channel(self, channel_data):
        n1, n2 = channel_data.shape  # n1 - number of epochs, n2 - samples per epoch
        v = self._samplewise_median(channel_data) if self._init_v is None else self._init_v  # init result
        w = np.zeros(n1)  # init vector of weights
        u = np.ones_like(w)  # init vector of correlations
        m_exponent = 1.0 / (1 - self._m)
        min_weight = 1.0 / (self._c * n1)

        for i in range(self._max_iter):
            old_w = w

            sigma = np.sum(np.abs(channel_data - v), axis=1)
            sigma += self._min_sigma  # robustness to local minima
            w = np.power(sigma, m_exponent)
            w *= u  # robustness to corrupted epochs
            w /= np.sum(w)
            w[w < min_weight] = 0  # robustness to strong outliers
            w /= np.sum(w)

            w_m = np.power(w, self._m)
            v = self._compute_v(w_m, channel_data)

            delta_w = np.linalg.norm(w - old_w)
            if i > 0 and delta_w <= self._ksi:
                break

            r = np.corrcoef(v, channel_data)  # (n_epochs+1, n_epochs+1)
            u = r[0, 1:]  # (1, n_epochs)
            u = (u+1) / 2  # scale correlations to (0, 1)
        else:
            print(f"ksi ({self._ksi}) not reached ({delta_w})!")

        # store execution details
        self.done_iter = i + 1
        self.W = w

        v = self._compute_v(w, channel_data)[0]  # final regularization

        return v
