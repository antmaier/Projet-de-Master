from networkx.classes import DiGraph
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from abc import ABC, abstractmethod
from itertools import pairwise
from copy import deepcopy


class TranslocationModel(ABC):
    """A translocation model defined by a kinetic scheme.

    The kinetic scheme is a directed graph where the nodes are the states of
    the system and the edges are the physical/chemical reactions, jumping from 
    one state to another.
    The nodes have a 'probability' attribute, which is the probability of the 
    steady-state system to be in this state. The directed edges have a 'rate' 
    attribute, which is the rate of the reaction. Edges may have other 
    attributes, such as 'ATP' or 'position' indicating the amount the 
    corresponding physical quantity is modified when this reaction occurs.

    A TranslocationModel also has physical parameters:
        - atp_adp_ratio: The ATP/ADP ratio.
        - equilibrium_atp_adp_ratio: The equilibrium ATP/ADP concentration ratio.
        - K_d_atp: The protomer-ATP dissociation constant.
        - K_d_adp: The protomer-ADP dissociation constant.
        - k_DT: Effective ADP->ATP exchange rate.
        - k_TD: Effective ATP->ADP exchange rate.
        - k_h: The ATP hydrolysis rate.
        - k_s: The ATP synthesis rate.
    In all generality, a reaction is distinct from its reaction rate. Therefore,
    reaction rates are class attributes/properties defined at the 
    TranslocationModel class level, and the 'rate' property of the edges are 
    functions that return the correct reaction rate. The syntax to access a 
    the rate of an edge is thus `edge['rate']()`, with the parenthesis at the 
    end. 
    This is useful for example when a reaction on the kinetic scheme has its 
    rate modified but some others are left inchanged, so that the physical
    quantity (the class attribute/property) is left unchanged, and one 
    assigns a new function to the 'rate' attribute of the corresponding edge
    in the kinetic scheme. An exemple of such a situation is when we apply an
    external force on the HSP100, which adds a Boltzmann factor to the 
    translocation reaction rate.
    This implementation choice also implies that when a physical parameter is 
    modified, the kinetic scheme is dynamically updated.

    The translocation model is a stochastic process. It can be simulated using
    the Gillespie algorithm, to simulates the evolution of a single particle on
    the kinetic scheme.

    TranslocationModel is an abstract class, a model is a subclass of it. 
    It must implement the _construct_kinetic_scheme method, which constructs 
    the kinetic scheme of the model, and define its own parameters before 
    calling the __super__ constructor. The kinetic scheme is then automatically 
    constructed in the TranslocationModel class.

    The models already implemented are:
    - Sequential Clockwise/2-Residue Step (SC2R)
    - Random Protomer Concertina Locomotion (RPCL)
    and their variant:
    - Defective: one protomer has a reduced hydrolysis rate
    - NonIdeal: all the states of the kinetic scheme are ocnnected to a new 
        state called 'out' representing any state not considered in the model
        decription.

    There are two types of physical parameters: free and constrained. The free
    parameters are self-explanatory, they can be freely set by the user. 
    Constrained parameters are constrained by a physical equation (e.g. a
    thermodynamic loop). 
    Free parameters are class attributes, whereas constrained parameters are
    class properties (without setter) where the core of the getter computes the 
    constrained value from the current value of the free parameters.

    Remark for code maintainers:
    The way it is implemented right now, the probabilities are computed each
    time a 'probability' node attribute is called. If the models become more 
    complex, it may be a costly operation. It may be better to store the 
    probabilities somewhere and update them automatically when any physical 
    parameter is modified. For the models we have right now, it is not a problem.
    """

    def __init__(self, atp_adp_ratio: float = 10,):
        # (Equilibrium) ATP/ADP concentration ratio
        self.atp_adp_ratio = atp_adp_ratio
        self.equilibrium_atp_adp_ratio = 1
        # Protomer-ATP/ADP dissociation constants
        self.K_d_atp = 1
        self.K_d_adp = 1
        # Effective ADP->ATP exchange rate (! not at equilibrium, has to be set 
        # to constrain all the degree of freedom of the ATP/ADP exchange model)
        self.k_DT = 1
        # One could instead define k_[on/off]_[atp/adp], the binding/unbinding
        # rates of ATP/ADP to/from the protomer, remove K_d_atp and K_d_adp
        # and then k_DT and k_TD would be determined by these rates and the 
        # ATP/ADP ratio at- and off-equilibrium. The way it is implemented now
        # has the advantage of having a free parameter less, but has the 
        # disadvantage of being less physically intuitive, since modifying the
        # ATP/ADP ratio does not directly modify k_DT.

        # ATP hydrolysis/synthesis rates
        self.k_h = 1
        self.k_s = 1

        self.kinetic_scheme = self._construct_kinetic_scheme()

    @property
    def k_TD(self) -> float:
        """Effective ATP->ADP exchange rate.

        It is constrained by the ATP/ADP exchange model.
        """
        return self.k_DT * self.K_d_atp / self.K_d_adp / self.atp_adp_ratio

    def gillespie(
        self,
        max_steps: int | None = 1000,
        max_time: float | None = None,
        initial_state: None | str = None,
        n_simulations: int = 1,
        cumulative_sums: str | list[str] | None = None,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Simulate the stochastic system using the Gillespie algorithm.

        Simulate the stochastic evolution of a single particle evoluting on 
        the kinetic scheme using the Gillespie algorithm, unitl one of the
        stopping conditions is met (max_steps or max_time).

        Args:
            max_steps: The maximum number of simulation steps.
            max_time: The maximum time of the simulation.
            initial_state: The initial state of the system. If None, a random
                initial state is chosen based on the steady-state probabilities.
            n_simulations: The number of simulations to do. If > 1, the returned
                object is a list of dataframes, one for each simulation.
            cumulative_sums: The edge attributes to compute the cumulative sum
                of. Each simulation result dataframe then contains new columns
                named 'attribute_name' with the cumulative sum of the specified
                edge attributes at each step, with the step time. It adds a
                row at the beginning with step time 0 and 0 value at each
                column. 
                If None, each simulation returns a dataframe with the step time 
                and the edge taken at each step with all its attributes.

        Returns:
            A dataframe (or a list of dataframes if n_simulations > 1) with the
            columns 'time', 'edge' and the elements of 'cumulative_sums'.
            The 'edge' column contains tuples of the form 
            (state_from, state_to, attributes), where attributes is a dict of 
            the form 'attribute': value.
        """
        if max_steps and max_steps < 1:
            raise ValueError(
                "The number of steps must be strictly greater than 0.")
        if max_time and max_time <= 0:
            raise ValueError(
                "The maximum time must be strictly greater than 0.")
        if initial_state and initial_state not in self.kinetic_scheme.nodes():
            raise ValueError("The initial state does not belong to the graph.")
        if n_simulations < 1:
            raise ValueError(
                "The number of simulations must be strictly greater than 0.")

        results = []
        for _ in range(n_simulations):
            result = {'time': [], 'edge': []}

            time = 0
            if initial_state:
                state = initial_state
            else:
                sorted_nodes = sorted(self.kinetic_scheme.nodes())
                probabilities = [
                    self.kinetic_scheme.nodes[node]['probability']()
                    for node in sorted_nodes]
                state = np.random.choice(sorted_nodes, p=probabilities)
            # Stop if at least one of the conditions is met
            # (max_steps or max_time)
            # max_steps is checked here at the for loop, max_time is checked
            # at each step within the loop
            for _ in range(max_steps):
                # Each step the system starts in the current state and then
                # after a sojourn time given by an exponential distribution
                # with parameter 'the sum of the leaving rates of the current
                # state', it jumps to the next state sampled with probability
                # proportional to the leaving rates.
                if True:
                    out_edges = list(
                        self.kinetic_scheme.out_edges(state, data=True))
                    total_rate = sum(
                        [attributes['rate']() for _, _, attributes in out_edges])
                    sojourn_time = np.random.exponential(1/total_rate)
                    chosen_edge_i = np.random.choice(
                        list(range(len(out_edges))),
                        p=[attributes['rate']()/total_rate
                           for _, _, attributes in out_edges])
                    chosen_edge = out_edges[chosen_edge_i]

                    time += sojourn_time
                    state = chosen_edge[1]
                # Alternative implementation where each reaction is run in
                # parallel, and the first one to occur is chosen
                else:
                    out_edges = list(
                        self.kinetic_scheme.out_edges(state, data=True))
                    betas = [1/attributes['rate']()
                             for _, _, attributes in out_edges]
                    sojourn_times = np.random.exponential(betas)
                    chosen_edge_i = np.argmin(sojourn_times)
                    sojourn_time = sojourn_times[chosen_edge_i]
                    chosen_edge = out_edges[chosen_edge_i]

                    time += sojourn_time
                    state = chosen_edge[1]

                if time > max_time:
                    break

                result['time'].append(time)
                result['edge'].append(chosen_edge)
            result = pd.DataFrame(result)
            if cumulative_sums:
                # TODO understand why if very few steps, there is a warning
                # Use ´warnings.filterwarnings("error", category=FutureWarning)´
                # to catch the warning
                # (good luck)
                self._compute_cumulative_sums(result, cumulative_sums)
            results.append(result)

        if n_simulations == 1:
            out = results[0]
        else:
            out = results

        return out

    def average_velocity(self) -> float:
        """Return the average velocity of the translocation model.

        The average velocity is the sum of rate times the probability of initial
        state times the displacement.
        """
        velocity = 0
        for u, v, displacement in self.kinetic_scheme.edges(data='position',
                                                            default=0):
            velocity += (displacement
                         * self.kinetic_scheme.nodes[u]['probability']()
                         * self.kinetic_scheme.edges[u, v]['rate']())
        return velocity

    def atp_consumption_rate(self) -> float:
        """Return the ATP consumption rate of the translocation model."""
        r = 0
        for u, v, atp in self.kinetic_scheme.edges(data='ATP',
                                                   default=0):
            r += (atp
                  * self.kinetic_scheme.nodes[u]['probability']()
                  * self.kinetic_scheme.edges[u, v]['rate']())
        return r

    def normalize_average_velocity(self, inplace: bool = True
                                   ) -> DiGraph | None:
        """Normalize the average velocity of the translocation model.

        Update all the rates so that the average velocity is 1.
        Useful to compare different models.
        """
        if self.average_velocity() == 0:
            raise ValueError("The average velocity is null, cannot normalize.")

        average_velocity = self.average_velocity()  # Velocity before normalization
        kinetic_scheme = (self.kinetic_scheme if inplace
                          else self.kinetic_scheme.copy())
        for edge in kinetic_scheme.edges():
            kinetic_scheme.edges[edge]['rate'] = (
                lambda old_rate=kinetic_scheme.edges[edge]['rate']:
                    old_rate() / average_velocity)
        return kinetic_scheme

    def analytical_attribute_stats(
        self,
        edge_attribute: str,
        times: float | list[float],
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        """Return mean, std and confidence interval of the edge attribute.

        Return the mean, standard deviation (std) and confidence interval
        (CI) at the specified confidence level of the specified edge attribute. 
        The confidence interval (CI) is computed with gaussian approximation.

        Args:
            edge_attribute: The edge attribute to compute the statistics of.
            times: The time(s) to compute the statistics at.
            confidence_level: The confidence level of the confidence interval. 
                Only relevant if confidence_interval is True.

        Returns:
            Pandas dataframe with the columns 'time', 'mean', 'std',
            'lower_bound', 'upper_bound'.
        """
        if isinstance(times, float):
            times = [times]
        times = np.array(times)

        mean = 0
        var = 0
        for u, v, value in self.kinetic_scheme.edges(data=edge_attribute,
                                                     default=0):
            p = self.kinetic_scheme.nodes[u]['probability']()
            k = self.kinetic_scheme.edges[u, v]['rate']()
            mean += value * p * k
            var += value**2 * p * k
        mean = times * mean
        var = times * var
        std = np.sqrt(var)
        q_lower, q_upper = norm.interval(confidence_level)
        lower_bound = mean + q_lower * std
        upper_bound = mean + q_upper * std
        return pd.DataFrame({
            'time': times,
            'mean': mean,
            'std': std,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })

    def empirical_attribute_stats(
        self,
        edge_attribute: str,
        times: float | list[float],
        confidence_level: float = 0.95,
        n_simulations: int = 100,
    ) -> pd.DataFrame:
        """Return mean, std and confidence interval of the edge attribute.

        Return the empirical mean, standard deviation (std) and confidence 
        interval (CI) at the specified confidence level, of the specified edge 
        attribute (e.g. 'position'), at each time in 'times', as well as samples
        of sojourn times spent at the same value of the edge attribute.
        The confidence interval (CI) is computed with gaussian approximation.

        Args:
            edge_attribute: The edge attribute to compute the statistics of.
            times: The time(s) to compute the statistics at. For sojourn times,
                simulations are done until the maximum time.
            confidence_level: The confidence level of the confidence interval. 
            n_simulations: The number of simulations to do to compute the
                empirical statistics.

        Returns:
            [DataFrame['time', 'mean', 'std', 'lower_bound', 'upper_bound'],
             DataFrame['sojourn_time']]
        """
        if isinstance(times, float):
            times = [times]
        if n_simulations < 2:
            raise ValueError(
                "The number of simulations must be at least 2.")

        trajectories = self.gillespie(
            max_time=max(times),
            n_simulations=n_simulations,
            cumulative_sums=edge_attribute,
        )
        # We keep only the edge_attribute and time columns since all stats
        # depend on these two columns only
        for i in range(len(trajectories)):
            trajectories[i] = trajectories[i].loc[:, ['time', edge_attribute]]

        # Get edge_attribute sojourn times (i.e. time spent at a value of
        # edge_attribute)
        sojourn_times = []
        for trajectory in trajectories:
            # We keep only the times where the value of edge_attribute changes
            # and compute the time it took to change
            # Concatenate all the sojourn times of all simulations
            sojourn_times.append(
                trajectory
                .loc[
                    np.invert(
                        np.isclose(
                            trajectory[edge_attribute].diff().fillna(1), 
                            0))]
                .loc[:, 'time']
                .diff()
                .dropna())
        sojourn_times = pd.concat(sojourn_times)

        # We keep only the edge_attribute and time columns and we fill
        # the dataframe at all times defined above with the last valid value
        for i in range(len(trajectories)):
            trajectories[i] = (trajectories[i]
                               .loc[:, ['time', edge_attribute]]
                               .set_index('time')
                               .reindex(times, method='ffill')
                               .reset_index())
        # We compute the statistics at each time
        statistics = (pd.concat(trajectories)
                      .groupby('time')[edge_attribute]
                      .agg(['mean', 'std']))
        q_lower, q_upper = norm.interval(confidence_level)
        statistics['lower_bound'] = (
            statistics['mean'] + q_lower * statistics['std'])
        statistics['upper_bound'] = (
            statistics['mean'] + q_upper * statistics['std'])
        statistics.reset_index(inplace=True)
        statistics['time'] = times
        return statistics, sojourn_times

    @abstractmethod
    def _construct_kinetic_scheme(
        self,
        kinetic_scheme: DiGraph | None = None
    ) -> DiGraph:
        """Construct the kinetic scheme of the translocation model.

        The kinetic scheme is a directed graph where the nodes are the states of
        the system and the edges are the reactions. The nodes have a 'probability'
        attribute, which is the probability of the steady-state system to be in
        this state. The directed edges have a 'rate' attribute, which is the rate
        of the reaction. Edges may have other attributes indicating the evolution of
        other physical quantities, such as 'ATP' or 'position'.
        """
        pass

    def _compute_probabilities(self) -> dict[str, float]:
        """Compute the steady-state probabilities of the kinetic scheme.

        Return a dictionary of form 'node_name': probability.
        """
        sorted_nodes = sorted(self.kinetic_scheme.nodes())
        # Update matrix M of the system \dot{p} = M p
        """M = (nx.adjacency_matrix(self.kinetic_scheme, 
                                 nodelist=sorted_nodes, 
                                 weight='rate')
             .toarray()
             .T)"""
        M = np.zeros((len(sorted_nodes), len(sorted_nodes)))
        for node in sorted_nodes:
            for neighbor, _, attributes in self.kinetic_scheme.in_edges(
                node, data=True
            ):
                M[sorted_nodes.index(node), sorted_nodes.index(neighbor)] = (
                    attributes['rate']())
        M = M - np.diag(M.sum(axis=0))
        # To solve we add the constraint that the sum of probabilities is 1
        # and we take advantage of the fact that the kernel of M has dimension 1
        M[-1, :] = np.ones(M.shape[1])
        b = np.zeros(M.shape[0])
        b[-1] = 1
        # Probabilities are then given by the solution of the linear system
        probabilities = np.linalg.solve(M, b)
        return {node: probability
                for node, probability in zip(sorted_nodes, probabilities)}

    def _compute_cumulative_sums(
        self,
        result: pd.DataFrame,
        cumulative_sums: str | list[str], # TODO change to "edge_attributes"
    ) -> None:
        """Compute the cumulative sum of the specified edge attributes.

        The post-processing consists in computing the cumulative sum of the
        specified edge attributes at each step. 
        The result is updated in-place.

        Args:
            result: A dataframe with the columns 'time' and 'edge'. The
                'edge' column contains tuples of the form 
                (state_from, state_to, attributes), where attributes is a dict
                of the form 'attribute': value.
            cumulative_sums: (list of) 'edge_attribute' for which to compute the
                cumulative sum.
        """
        if isinstance(cumulative_sums, str):
            cumulative_sums = [cumulative_sums]
        for edge_attribute in cumulative_sums:
            result[edge_attribute] = (
                result['edge']
                .apply(lambda edge: edge[2].get(edge_attribute))
                .cumsum())
        # Add a row at the beginning with time 0 and value 0 at each column
        result.loc[-1] = [0] + [None] + [0] * len(cumulative_sums)
        result.index += 1
        result.sort_index(inplace=True)
        # Fill None values in the cumulative sum columns with last valid value
        result.ffill(inplace=True)


class SC2R(TranslocationModel):
    """Sequential Clockwise/2-Residue Step, 1-Loop translocation model.

    Physical parameters:
        k_up: Translocation up rate.
        k_down: Translocation down rate.
    """

    def __init__(self, atp_adp_ratio: float = 10) -> None:
        self.k_up = 1  # Translocation up rate
        super().__init__(atp_adp_ratio)

    @property
    def k_down(self) -> float:
        """Translocation down rate, constrained by the detailed balance."""
        return (
            self.k_h * self.k_up * self.k_DT
            / (self.k_s * self.k_TD)
            * (self.equilibrium_atp_adp_ratio / self.atp_adp_ratio)
        )

    def _construct_kinetic_scheme(self, kinetic_scheme: DiGraph | None = None
                                  ) -> DiGraph:
        if not kinetic_scheme:
            kinetic_scheme = DiGraph()
        kinetic_scheme.add_nodes_from([
            ('TTT', {'probability':
                     lambda: self._compute_probabilities()['TTT']}),
            ('DTT', {'probability':
                     lambda: self._compute_probabilities()['DTT']}),
            ('TTD', {'probability':
                     lambda: self._compute_probabilities()['TTD']})
        ])
        kinetic_scheme.add_edges_from([
            ('TTT', 'DTT', {'rate': lambda: self.k_h, 'ATP': -1}),
            ('DTT', 'TTT', {'rate': lambda: self.k_s, 'ATP': 1}),
            ('DTT', 'TTD', {'rate': lambda: self.k_up, 'position': 2}),
            ('TTD', 'DTT', {'rate': lambda: self.k_down, 'position': -2}),
            ('TTD', 'TTT', {'rate': lambda: self.k_DT}),
            ('TTT', 'TTD', {'rate': lambda: self.k_TD})
        ])
        return kinetic_scheme


class RPCL(TranslocationModel):
    """Random Protomer Concertina Locomotion translocation model.

    Physical parameters:
        k_[up/down]_[extend/contract]: Translocation rates.
    """

    def __init__(self, atp_adp_ratio: float = 10, n_protomers: int = 6) -> None:
        self.n_protomers = n_protomers
        self.k_up_contract = 1  # Upward contraction translocation rate
        self.k_down_extend = 1  # Downward extension translocation rate
        self.k_up_extend = 1  # Upward extension translocation rate
        super().__init__(atp_adp_ratio)

    @property
    def k_h_bar(self) -> float:
        """Effective ATP hydrolysis rate.

        Each protomer contributes k_h to the effective hydrolysis rate.
        """
        return self.n_protomers * self.k_h

    @property
    def k_down_extend_bar(self) -> float:
        """Effective downward extension translocation rate.

        Each protomer contributes k_down_extend to the effective 
        translocation rate.
        """
        return self.n_protomers * self.k_down_extend

    @property
    def k_down_contract(self) -> float:
        """Downward contraction translocation rate, constrained by the detailed balance."""
        return ((self.k_h * self.k_up_extend * self.k_DT
                 * self.k_up_contract)
                / (self.k_s * self.k_TD * self.k_down_extend)
                * (self.equilibrium_atp_adp_ratio / self.atp_adp_ratio))

    def _construct_kinetic_scheme(self, kinetic_scheme: DiGraph | None = None
                                  ) -> DiGraph:
        if not kinetic_scheme:
            kinetic_scheme = DiGraph()
        kinetic_scheme.add_nodes_from([
            ('contracted-ATP', {'probability':
                          lambda: self._compute_probabilities()['contracted-ATP']}),
            ('contracted-ADP', {'probability':
                          lambda: self._compute_probabilities()['contracted-ADP']}),
            ('extended-ADP', {'probability':
                              lambda: self._compute_probabilities()['extended-ADP']}),
            ('extended-ATP', {'probability':
                              lambda: self._compute_probabilities()['extended-ATP']})
        ])
        kinetic_scheme.add_edges_from([
            ('contracted-ATP', 'contracted-ADP',
             {'rate': lambda: self.k_h_bar, 'ATP': -1}),
            ('contracted-ADP', 'contracted-ATP', {'rate': lambda: self.k_s, 'ATP': 1}),
            ('contracted-ADP', 'extended-ADP', {
                'rate': lambda: self.k_up_extend,
                'position': 2*(self.n_protomers-1)}),
            ('extended-ADP', 'contracted-ADP', {
                'rate': lambda: self.k_down_contract,
                'position': -2*(self.n_protomers-1)}),
            ('extended-ADP', 'extended-ATP', {'rate': lambda: self.k_DT}),
            ('extended-ATP', 'extended-ADP', {'rate': lambda: self.k_TD}),
            ('extended-ATP', 'contracted-ATP', {
                'rate': lambda: self.k_up_contract}),
            ('contracted-ATP', 'extended-ATP', {
                'rate': lambda: self.k_down_extend_bar})
        ])
        return kinetic_scheme


class NonIdealSC2R(SC2R):
    """Non-ideal Sequential Clockwise/2-Residue Step translocation model.

    Non-ideal in the sense that at each state, there is a probability to
    leave the main loop, to a effective state 'out'.
    All k_in rates are similar, and remains only one free parameter k_out, 
    chosen to be the rate of leaving a reference state.
    """

    def __init__(self) -> None:
        self.reference_state = 'TTT'  # State from which the out rate is computed
        self.k_out = 1
        self.k_in = 1
        super().__init__()

    def _k_out(self, state: str) -> float:
        """Compute state out rate, constrained by the detailed balance.

        The constraint is computed based on the main loop.
        """
        rate = self.k_out
        try:
            path = nx.shortest_path(
                self._main_loop, state, self.reference_state)
        except nx.NetworkXNoPath:
            pass
        else:
            for u, v in pairwise(path):
                rate *= (self._main_loop.edges[u, v]['rate']()
                         / self._main_loop.edges[v, u]['rate']())
        return rate

    def _construct_kinetic_scheme(self, kinetic_scheme: DiGraph | None = None
                                  ) -> DiGraph:
        if not kinetic_scheme:
            kinetic_scheme = DiGraph()
        # Construct main loop
        kinetic_scheme = super()._construct_kinetic_scheme(kinetic_scheme)
        self._main_loop = deepcopy(kinetic_scheme)  # Used to compute _k_out
        # Add 'out' state
        kinetic_scheme.add_node(
            'out',
            probability=lambda: self._compute_probabilities()['out'])
        for node in kinetic_scheme.nodes:
            if node != 'out':
                kinetic_scheme.add_edges_from([
                    (node, 'out', {
                     'rate': lambda node=node: self._k_out(node)}),
                    ('out', node, {'rate': lambda: self.k_in})])
        return kinetic_scheme


class NonIdealRPCL(RPCL):
    """Non-ideal RPCL translocation model.

    Non-ideal in the sense that at each state, there is a probability to
    leave the main loop, to a effective state 'out'.
    All k_in rates are similar, and remains only one free parameter k_out, 
    chosen to be the rate of leaving a reference state.
    """

    def __init__(self) -> None:
        self.reference_state = 'contracted-ATP'  # State from which the out rate is computed
        self.k_out = 1
        self.k_in = 1
        super().__init__()

    def _k_out(self, state: str) -> float:
        """Compute state out rate, constrained by the detailed balance.

        The constraint is computed based on the main loop.
        """
        rate = self.k_out
        try:
            path = nx.shortest_path(
                self._main_loop, state, self.reference_state)
        except nx.NetworkXNoPath:
            pass
        else:
            for u, v in pairwise(path):
                rate *= (self._main_loop.edges[u, v]['rate']()
                         / self._main_loop.edges[v, u]['rate']())
        return rate

    def _construct_kinetic_scheme(self, kinetic_scheme: DiGraph | None = None
                                  ) -> DiGraph:
        if not kinetic_scheme:
            kinetic_scheme = DiGraph()
        # Construct main loop
        kinetic_scheme = super()._construct_kinetic_scheme(kinetic_scheme)
        self._main_loop = deepcopy(kinetic_scheme)  # Used to compute _k_out
        # Add 'out' state
        kinetic_scheme.add_node(
            'out',
            probability=lambda: self._compute_probabilities()['out'])
        for node in kinetic_scheme.nodes:
            if node != 'out':
                kinetic_scheme.add_edges_from([
                    (node, 'out', {
                     'rate': lambda node=node: self._k_out(node)}),
                    ('out', node, {'rate': lambda: self.k_in})])
        return kinetic_scheme


class DefectiveSC2R(SC2R):
    """Sequential Clockwise/2-Residue Step with one defective protomer.

    Single-loop-like translocation model with one defective protomer. The 
    defective protomer has an hydrolysis rate that is defect_factor times 
    smaller than the other protomers.

    The states are defined by the ADP/ATP-state of the protomer and the 
    position of the defective protomer, e.g. DTT(T)TT, 
    where T means 'ATP-state', D means 'ADP-state', (X) means 'protomer in 
    X-state is defective'.
    Ignoring the defective protomer, the possible states are:
        - All protomers in ATP-state;
        - All protomers in ATP-state, except the first one in ADP-state;
        - All protomers in ATP-state, except the last one in ADP-state.
    Then the defective protomer can be at any position. 
    The total number of states is then 3*n_protomers.
    """

    def __init__(
            self,
            defect_factor: float = 0.1,
            atp_adp_ratio: float = 10,
            n_protomers: int = 6
    ) -> None:
        """Initialize the defective translocation model.

        Args:
            defect_factor: The factor by which the defective protomer hydrolysis
                rate is smaller than the other protomers hydrolysis rate.
            atp_adp_ratio: The ATP/ADP ratio.
            n_protomers: The number of protomers.
        """
        self.defect_factor = defect_factor
        self.n_protomers = n_protomers
        super().__init__(atp_adp_ratio)

    @property
    def k_h_defect(self) -> float:
        """ATP hydrolysis rate of the defective protomer."""
        return self.defect_factor * self.k_h

    @property
    def k_down(self) -> float:
        """Translocation down rate, constrained by the detailed balance."""
        return (
            (self.k_h**5 * self.k_h_defect)**(1/6) * self.k_up * self.k_DT
            / (self.k_s * self.k_TD)
            * (self.equilibrium_atp_adp_ratio / self.atp_adp_ratio)
        )

    def probabilities_defect_ignored(self) -> dict[str, float]:
        """Compute the total probabilities to be in each defect-ignored state.

        Defect-ignored states mean that we do not differenciate the states
        where the defective protomer is at different positions, e.g. 
        DTT(T)TT, DTTT(T)T and DTTTT(T) are all considered as DTTTTT.
        """
        probabilities = self._compute_probabilities()
        probabilities_defect_ignored = {}
        for state in probabilities:
            state_defect_ignored = state.replace('(', '').replace(')', '')
            if state_defect_ignored in probabilities_defect_ignored:
                probabilities_defect_ignored[state_defect_ignored] += (
                    probabilities[state])
            else:
                probabilities_defect_ignored[state_defect_ignored] = (
                    probabilities[state])
        return probabilities_defect_ignored

    def _construct_kinetic_scheme(self, kinetic_scheme: DiGraph | None = None
                                  ) -> DiGraph:
        if not kinetic_scheme:
            kinetic_scheme = DiGraph()
        for i in range(self.n_protomers):
            states = ['T'*self.n_protomers,
                      'D' + 'T'*(self.n_protomers-1),
                      'T'*(self.n_protomers-1) + 'D']
            for state in states:
                state = state[:i] + '(' + state[i] + ')' + state[i+1:]
                kinetic_scheme.add_node(
                    state,
                    probability=lambda state=state:
                        self._compute_probabilities()[state]
                )

        def add_defect_parenthesis(state: str, i: int) -> str:
            return state[:i] + '(' + state[i] + ')' + state[i+1:]

        for state in kinetic_scheme.nodes():
            state_defect_ignored = state.replace('(', '').replace(')', '')
            defect_index = state.find('(')
            if state_defect_ignored[0] == 'D':
                next_state = state_defect_ignored[1:] + 'D'
                next_state = add_defect_parenthesis(
                    next_state,
                    (defect_index - 1) % self.n_protomers)
                kinetic_scheme.add_edges_from([
                    (state, next_state, {'rate': lambda: self.k_up,
                                         'position': 2}),
                    (next_state, state, {'rate': lambda: self.k_down,
                                         'position': -2})
                ])
            elif state_defect_ignored[-1] == 'D':
                next_state = state_defect_ignored[:-1] + 'T'
                next_state = add_defect_parenthesis(next_state, defect_index)
                kinetic_scheme.add_edges_from([
                    (state, next_state, {'rate': lambda: self.k_DT}),
                    (next_state, state, {'rate': lambda: self.k_TD})
                ])
            elif 'D' not in state_defect_ignored:
                next_state = 'D' + state_defect_ignored[1:]
                next_state = add_defect_parenthesis(next_state, defect_index)
                rate = ((lambda: self.k_h_defect)
                        if defect_index == 0
                        else lambda: self.k_h)
                kinetic_scheme.add_edges_from([
                    (state, next_state, {'rate': rate,
                                         'ATP': -1}),
                    (next_state, state, {'rate': lambda: self.k_s,
                                         'ATP': 1})
                ])
            else:
                raise ValueError("Invalid state.")
        return kinetic_scheme


class DefectiveRPCL(RPCL):
    """RPCL model with one protomer with defective hydrolysis.

    The defective protomer has an hydrolysis rate that is defect_factor times 
    smaller than the other protomers.

    The states in defect-free loops are denoted by:
        'contracted-ATP', 'contracted-ADP', 'extended-ATP', 'extended-ADP'
    and in the defective loop by:
        'contracted-ADP-defect', 'extended-ATP-defect', 'extended-ADP-defect'

    There are n_protomers-1 defect-free loops and one defective loop. For 
    simplicity of the code, we aggregate all the defect-free loops together in a
    single loop with an effective hydrolysis rate k_h_bar = (n_protomers-1)*k_h, 
    and an effective downward extension translocation rate 
    k_down_extend_bar = (n_protomers-1)*k_down_extend_bar.
    """

    def __init__(
            self,
            defect_factor: float = 0.1,
            atp_adp_ratio: float = 10,
            n_protomers: int = 6
    ) -> None:
        """Initialize the defective translocation model.

        Args:
            defect_factor: The factor by which the defective protomer hydrolysis
                rate is smaller than the other protomers hydrolysis rate.
        """
        self.defect_factor = defect_factor
        super().__init__(atp_adp_ratio, n_protomers)

    @property
    def k_h_defect(self) -> float:
        """ATP hydrolysis rate of the defective protomer."""
        return self.defect_factor * self.k_h

    @property
    def k_h_bar(self) -> float:
        """Effective ATP hydrolysis rate for defect-free loop.

        Each defect-free protomer contributes k_h to the effective hydrolysis 
        rate.
        """
        return (self.n_protomers - 1) * self.k_h

    @property
    def k_down_extend_bar(self) -> float:
        """Effective downward extension translocation rate.

        Each defect-free protomer contributes k_down_extend to the 
        effective translocation rate.
        """
        return (self.n_protomers - 1) * self.k_down_extend

    # No need to redefine k_down_contract, the detailed balance constraint is 
    # the same as in the defect-free case.

    def probabilities_defect_ignored(self) -> dict[str, float]:
        """Compute the total probabilities to be in each defect-ignored state.

        Defect-ignored mean that we do not differenciate the defect-free 
        protomers from the defective one.
        The states are then:
            'contracted-ATP', 'contracted-ADP', 'extended-ATP', 'extended-ADP'
        """
        probabilities = self._compute_probabilities()
        probabilities_defect_ignored = {
            'contracted-ATP': probabilities['contracted-ATP'],
            'contracted-ADP': (probabilities['contracted-ADP']
                         + probabilities['contracted-ADP-defect']),
            'extended-ATP': (probabilities['extended-ATP']
                             + probabilities['extended-ATP-defect']),
            'extended-ADP': (probabilities['extended-ADP']
                             + probabilities['extended-ADP-defect'])
        }
        return probabilities_defect_ignored

    def _construct_kinetic_scheme(self, kinetic_scheme: DiGraph | None = None
                                  ) -> DiGraph:
        # Defect-free loop
        kinetic_scheme = super()._construct_kinetic_scheme(kinetic_scheme)
        # Defective loop
        kinetic_scheme.add_nodes_from([
            ('contracted-ADP-defect',
             {'probability':
                 lambda: self._compute_probabilities()['contracted-ADP-defect']}),
            ('extended-ATP-defect',
             {'probability':
                 lambda: self._compute_probabilities()['extended-ATP-defect']}),
            ('extended-ADP-defect',
             {'probability':
                 lambda: self._compute_probabilities()['extended-ADP-defect']})
        ])
        kinetic_scheme.add_edges_from([
            ('contracted-ATP', 'contracted-ADP-defect', {  # Defective hydrolysis
                'rate': lambda: self.k_h_defect,
                'ATP': -1}),
            ('contracted-ADP-defect', 'contracted-ATP', {  # Defective synthesis
                'rate': lambda: self.k_s,
                'ATP': 1}),
            ('contracted-ADP-defect', 'extended-ADP-defect', {
                'rate': lambda: self.k_up_extend,
                'position': 2*(self.n_protomers-1)}),
            ('extended-ADP-defect', 'contracted-ADP-defect', {
                'rate': lambda: self.k_down_contract,
                'position': -2*(self.n_protomers-1)}),
            ('extended-ADP-defect', 'extended-ATP-defect', {  # ADP->ATP exchange
                'rate': lambda: self.k_DT}),
            ('extended-ATP-defect', 'extended-ADP-defect', {  # ATP->ADP exchange
                'rate': lambda: self.k_TD}),
            ('extended-ATP-defect', 'contracted-ATP', {
                'rate': lambda: self.k_up_contract}),
            ('contracted-ATP', 'extended-ATP-defect', {
                'rate': lambda: self.k_down_extend})
        ])
        return kinetic_scheme
