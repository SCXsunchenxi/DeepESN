# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as linfit
from IPython.display import clear_output


class ESN:
    def __init__(self, _input_nodes=1, _reservoir_nodes=50, _output_nodes=1, _optimal=True, _k_percentage_optimal=0.35):
        """
        Echo State Network Object.
        """
        self.input_nodes = _input_nodes
        self.reservoir_nodes = _reservoir_nodes
        self.output_nodes = _output_nodes

        self.w_in = self.new_w_in()
        self.w_res = self.new_w_res()
        self.w_back = self.new_w_in(_input_nodes=self.output_nodes,
                                    _reservoir_nodes=self.reservoir_nodes,
                                    optimal=_optimal, k_percentage_optimal=_k_percentage_optimal)
        self.w_out = np.zeros((self.reservoir_nodes, self.output_nodes))
        self.is_trained = False

    def new_w_in(self, _input_nodes=-1, _reservoir_nodes=-1, optimal=True, k_percentage_optimal=0.35):
        """
        Create new Reservoirs input connection given the input and reservoir nodes. Default values reservoirs and input nodes.
        """
        if _input_nodes == -1:
            _input_nodes = self.input_nodes
        if _reservoir_nodes == -1:
            _reservoir_nodes = self.reservoir_nodes
        if optimal:
            in_res = 2 * np.random.randint(low=0, high=2, size=(_input_nodes * _reservoir_nodes)) - 1
            in_res[0:int(k_percentage_optimal * _input_nodes * _reservoir_nodes)] = 0
            np.random.shuffle(in_res)
            in_res = np.reshape(in_res[..., np.newaxis], (_input_nodes, _reservoir_nodes))
            return in_res
        return np.random.randint(low=-1, high=2, size=(_input_nodes, _reservoir_nodes))

    def new_w_res(self, _reservoir_nodes=-1):
        """
        Create new Reservoirs interconnection given the reservoir nodes. Default value reservoirs nodes.
        """
        if _reservoir_nodes == -1:
            _reservoir_nodes = self.reservoir_nodes
        w_res = np.random.normal(0, 1, (_reservoir_nodes, _reservoir_nodes))
        return w_res / max(np.absolute(np.linalg.eigvals(w_res)))

    def normalize_input(self, u):
        """
        Normalize input data.
        """
        return (u - np.mean(u, axis=0)) / np.std(u, axis=0)

    def reservoir_state(self, u, leaky_rate=1, input_gain=0.1, spectral_radius=1, feedback_gain=0, d=[-1], bias=True):
        """
        Compute the reservoirs state given the set of parameters.
        """
        comp = d == [-1]
        if np.all(comp):
            d = np.zeros((self.output_nodes, u.shape[1]))
        state = np.zeros((self.w_res.shape[0], u.shape[1]))
        for i in range(u.shape[1]):
            if i == 0:
                state[:, i] = leaky_rate * np.tanh(input_gain * self.w_in.T @ u[:, i])
            else:
                state[:, i] = (1 - leaky_rate) * state[:, i - 1]
                state[:, i] += leaky_rate * np.tanh(
                    input_gain * self.w_in.T @ u[:, i] + spectral_radius * self.w_res.T @ state[:,
                                                                                          i - 1] + feedback_gain * self.w_back.T @ d[
                                                                                                                                   :,
                                                                                                                                   i - 1])
        # Add one bias node for a better amplitude in the linear fit
        if bias:
            state = np.concatenate((state, np.ones((1, state.shape[1]))), axis=0)
        return state

    def fit(self, reservoir_state, y_train, _omit_states=10):
        """
        End-to-End model fit using Tikhonov Regularization to generalize to unseen data.
        """
        # Remove the first element of the resevoir since does not consider the previous state of the reservoir
        regression = linfit.BayesianRidge(n_iter=3000, tol=1e-6, verbose=True, fit_intercept=False, normalize=True)
        for i in range(y_train.shape[0]):
            regression.fit(reservoir_state[:, _omit_states:].T, y_train[i, _omit_states:])
            if i == 0:
                self.w_out = regression.coef_
                self.w_out = self.w_out[np.newaxis, ...]
            else:
                self.w_out = np.concatenate((self.w_out, regression.coef_[np.newaxis, ...]), axis=0)
        self.is_trained = True

    def predict(self, reservoir_state):
        """
        Use the ESN architecture to make a prediction for a given reservoir state.
        The System should have been trained earlier.
        """
        if not self.is_trained:
            print('Reservoir is not trained.')
            return None
        result = np.zeros((self.w_out.shape[0], reservoir_state.shape[1]))
        for i in range(self.w_out.shape[0]):
            result[i, :] = self.w_out[i].T @ reservoir_state
        return result

    def nrmse(self, y_real, y_prediction):
        """
        Normalized Root Mean Square Error.
        """
        return np.mean(np.sqrt(((y_real - y_prediction) ** 2) / np.var(y_real)))

    def mse(self, y_real, y_prediction):
        """
        Mean Square Error.
        """
        return np.mean((y_real - y_prediction) ** 2)

    def weights_visualization(self, input_weights=True, reservoir_weights=True, back_weights=False, out_weights=False,
                              display_in='rows'):
        """
        Make a plot of the different given connection weights for visualization. The parameter display_in must be 'rows' or 'cols'.
        """
        amount = 0
        count = 0
        if input_weights:
            amount += 1
        if reservoir_weights:
            amount += 1
        if back_weights:
            amount += 1
        if out_weights:
            amount += 1

        if display_in == 'rows':
            fig, ax = plt.subplots(nrows=amount)
        elif display_in == 'cols':
            fig, ax = plt.subplots(ncols=amount)
        else:
            print('non valid display_in parameter, must be rows or cols')
            return

        if input_weights:
            if amount > 1:
                ax[count].plot(range(self.w_in.size), np.reshape(self.w_in, (self.w_in.size)))
                ax[count].set_title('Input weights')
                count += 1
            else:
                ax.plot(range(self.w_in.size), np.reshape(self.w_in, (self.w_in.size)))
                ax.set_title('Input weights')

        if reservoir_weights:
            if amount > 1:
                ax[count].plot(range(self.w_res.size), np.reshape(self.w_res, (self.w_res.size)))
                ax[count].set_title('Reservoir weights')
                count += 1
            else:
                ax.plot(range(self.w_res.size), np.reshape(self.w_res, (self.w_res.size)))
                ax.set_title('Reservoir weights')

        if back_weights:
            if amount > 1:
                ax[count].plot(range(self.back_weights.size), np.reshape(self.back_weights, (self.back_weights.size)))
                ax[count].set_title('Back weights')
                count += 1
            else:
                ax.plot(range(self.back_weights.size), np.reshape(self.back_weights, (self.back_weights.size)))
                ax.set_title('Back weights')

        if out_weights:
            if amount > 1:
                ax[count].plot(range(self.w_out.size), np.reshape(self.w_out, (self.w_out.size)))
                ax[count].set_title('Out weights')
            else:
                ax.plot(range(self.w_out.size), np.reshape(self.w_out, (self.w_out.size)))
                ax.set_title('Out weights')
        plt.show()

    def reservoir_state_visualization(self, reservoir_state, interval=[0, -1]):
        """
        Visualize the reservoirs dynamics where the nodes values for time step k
        are concatenated. Interval default: plot all reservoir. 
        """
        if interval[1] == -1:
            interval[1] = reservoir_state.size

        plt.title('Reservoirs Dynamics')
        plt.plot(range(reservoir_state.size), np.reshape(reservoir_state.T, reservoir_state.size))
        plt.xlim(interval)
        plt.show()

    def grid_search(self, u, y, leaky_range, input_gain_range, spectral_radius_range, feedback_gain_range, _d=[-1]):
        """
        Grid search for optimal parameter search.
        """
        comp = _d == -1
        if np.all(comp):
            _d = np.zeros_like(y)

        best_parameters_nrmse = []
        best_parameters_mse = []
        lowest_nrmse = 10
        lowest_mse = 10
        count = 0
        for i in leaky_range:
            for j in input_gain_range:
                for k in spectral_radius_range:
                    for l in feedback_gain_range:
                        count += 1
                        state = self.reservoir_state(u, leaky_rate=i, input_gain=j, spectral_radius=k, feedback_gain=l,
                                                     d=_d)
                        self.fit(state, y)
                        y_fit = self.predict(state)

                        nrmsError = self.nrmse(y, y_fit)
                        if nrmsError < lowest_nrmse:
                            lowest_nrmse = nrmsError
                            best_parameters_nrmse.insert(0, [nrmsError, i, j, k, l])

                        msError = self.mse(y, y_fit)
                        if msError < lowest_mse:
                            lowest_mse = msError
                            best_parameters_mse.insert(0, [msError, i, j, k, l])

                        # if count%100==0:
                        clear_output(wait=True)
                        print('{}/{}  -->  NRMSE: {} MSE: {}  -->  params:({},{},{},{})'.format(count,
                                                                                                leaky_range.size * input_gain_range.size * spectral_radius_range.size * feedback_gain_range.size,
                                                                                                round(nrmsError, 4),
                                                                                                round(msError, 4), i, j,
                                                                                                k, l))
                        print('best  NRMSE: {} MSE: {}  --  params:({},{},{},{})'.format(round(lowest_nrmse, 4),
                                                                                         round(lowest_mse, 4),
                                                                                         best_parameters_nrmse[0][1],
                                                                                         best_parameters_nrmse[0][2],
                                                                                         best_parameters_nrmse[0][3],
                                                                                         best_parameters_nrmse[0][4]))
        return best_parameters_nrmse, best_parameters_mse


if __name__ == '__main__':
    res = ESN()
    res.weights_visualization()
