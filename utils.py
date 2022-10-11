import datetime as dt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from statsmodels.tsa.arima_process import ArmaProcess


class SyntheticData(object):
    """Synthetic data utility. Generates regression or ARMA datasets
    with known underlying stochastic impulses on treatment periods.
    """
    REGRESSION = 'regression'
    ARMA = 'arma'
    def __init__(
        self,
        n_samples=150,
        n_treatment=30,
        n_controls=50,
        n_informative_controls=5,
        impulse=0.02,
        bias=0.0,
        noise=0.0
    ):
        self.n_samples = n_samples
        self.n_treatment = n_treatment
        self.n_controls = n_controls
        self.n_informative_controls = n_informative_controls
        self.impulse = impulse
        self.bias = bias
        self.noise = noise

    @staticmethod
    def _make_regression_base(
        n_samples=150,
        n_treatment=30,
        n_controls=50,
        n_informative_controls=5,
        impulse=0.02,
        **make_regression_kwargs
        ):
        """Support method for self.make_regression.
        """
        kwargs = dict(
            n_samples=n_samples,
            n_features=n_controls,
            n_informative=n_informative_controls
        )
        kwargs.update(make_regression_kwargs)
        X, y = make_regression(**kwargs)
        y += (abs(y.min()) + 1)
        A = np.full(n_treatment, impulse) * (1 + np.random.random(n_treatment) - 0.5)
        y[-n_treatment:] += A*y[-n_treatment:].mean()
        #y[-n_treatment:] += impulse*y[-n_treatment:].mean()
        return X, y, None

    def make_regression(self, as_timeseries=True, **make_regression_kwargs):
        """Generate mock data for running conformal inference. The method is built
        over `sklearn.datasets.make_regression`.
        - Pass `make_regression` kwargs inside any extra kwargs.
        - Add time indices by passing as_timeseries=True (default).
        """
        X, y, _ = self._make_regression_base(
            n_samples=self.n_samples,
            n_treatment=self.n_treatment,
            n_controls=self.n_controls,
            n_informative_controls=self.n_informative_controls,
            impulse=self.impulse,
            **make_regression_kwargs
        )
        if not as_timeseries:
            return X, y, self.n_treatment
        # add dates
        X, y, slices = self._add_dates(X, y, self.n_treatment)
        return X, y, slices

    @staticmethod
    def _add_dates(X, y, n_treatment):
        """Adds dates (freq=D) to X, y series. Returns pandas DF.
        """
        n_samples = y.size
        # add dates
        ix = pd.date_range(end=dt.date.today(), periods=(n_samples+1), freq='D')[:-1]
        X = pd.DataFrame(X, index=ix)
        y = pd.DataFrame(y, index=ix)
        # calculate slices
        slices = {
            'treatment': slice(ix[-n_treatment], ix[-1]),
            'blackouts': []
        }
        return X, y, slices

    def make_arma(self, as_timeseries=True, **arma_kwargs):
        """Generate mock data for running Bayesian STS causal inference. The method
        is built over a `statsmodels.tsa.arima_process.ArmaProcess`.
        - Pass ArmaProcess kwargs any extra kwargs.
        - Add time indices by passing as_timeseries=True (default).
        """
        kwargs = {
        # see https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.html
            'ar': np.array([1, -.5, .25]),
            'ma': np.array([1, -.5, .5])
        }
        kwargs.update(arma_kwargs)
        arma_process = ArmaProcess(**kwargs)
        X = []
        for i in range(self.n_controls):
            mu, sigma = np.random.normal(self.bias, self.noise, size=2)
            s = (
                arma_process.generate_sample(nsample=self.n_samples)
                + np.random.normal(mu, abs(sigma), size=self.n_samples)
            )
            X.append(s)
        X = np.array(X).T
        # change this so we can get uninformative controls
        betas = np.r_[
            np.ones(shape=self.n_informative_controls),
            np.zeros(shape=(self.n_controls - self.n_informative_controls))]
        np.random.shuffle(betas)
        assert X.shape[1] == betas.size
        y = np.matmul(X, betas)
        y += abs(y.min()) + 1.0
        # add impulse
        _ = (self.n_samples - self.n_treatment)
        y[_:] += y[_:]*self.impulse
        if not as_timeseries:
            return X, y, self.n_treatment
        # add dates
        X, y, slices = self._add_dates(X, y, self.n_treatment)
        return X, y, slices
