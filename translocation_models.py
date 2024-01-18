from networkx.classes import DiGraph
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from abc import ABC, abstractmethod


class TranslocationModel(ABC):
    """A translocation model defined by a kinetic scheme.
    
    The kinetic scheme is a directed graph where the nodes are the states of
    the system and the edges are the reactions. The nodes have a 'probability'
    attribute, which is the probability of the steady-state system to be in
    this state. The directed edges have a 'rate' attribute, which is the rate
    of the reaction. Edges may have other attributes indicating the evolution of
    other physical quantities, such as 'ATP' or 'position'.
    'rate' and 'probability' attributes are functions that return the value of
    the rate/probability dynamically. This means that if the quantity depends on
    a physical parameter the value of the quantity will be updated when the 
    physical parameter is modified.

    The translocation model is a stochastic process. It can be simulated using
    the Gillespie algorithm. The stochastic process is a single particle
    evoluting on the kinetic scheme. 

    We are interested in the position of the particle over time, for various
    models such as 'Sequential Clockwise/2-Residue Step' or 'Disc-Spiral'.

    A model is a subclass of TranslocationModel. It must implement the 
    _construct_kinetic_scheme method, which constructs the kinetic scheme of
    the model, and define its own parameters before calling the __super__
    constructor. The kinetic scheme is then automatically constructed in the 
    TranslocationModel class, accessible in the kinetic_scheme attribute.

    Physical parameters:
        atp_adp_ratio: The ATP/ADP ratio.
        equilibrium_atp_adp_ratio: The equilibrium ATP/ADP concentration ratio.
        K_d_atp: The protomer-ATP dissociation constant.
        K_d_adp: The protomer-ADP dissociation constant.
        k_DT: Effective ADP->ATP exchange rate.
        k_TD: Effective ATP->ADP exchange rate.
        k_h: The ATP hydrolysis rate.
        k_s: The ATP synthesis rate.


    Remark for code maintainers:
    The way it is implemented right now, the probabilities are computed each
    time they are needed. If the models become more complex, it may be a costly
    operation. It may be better to store the probabilities somewhere and update
    them automatically when any physical parameter is modified.
    For the models we have right now, it is not a problem.
    """

    def __init__(self, atp_adp_ratio: float = 10,): 
        self._atp_adp_ratio = atp_adp_ratio
        self.equilibrium_atp_adp_ratio = 1
        # Protomer-ATP/ADP dissociation constants
        self.K_d_atp = 1
        self.K_d_adp = 1
        # Effective ADP->ATP exchange rate
        self.k_DT = 1
        # ATP hydrolysis/synthesis rates
        self.k_h = 1
        self.k_s = 1
        
        self.kinetic_scheme = self._construct_kinetic_scheme() # None # TODO abstract attribute
        
    @property
    def atp_adp_ratio(self) -> float:
        """ATP/ADP concentration ratio."""
        return self._atp_adp_ratio
    
    @atp_adp_ratio.setter
    def atp_adp_ratio(self, value: float) -> None:
        if value < 0:
            raise ValueError("The ATP/ADP ratio must be nonnegative.")
        self._atp_adp_ratio = value

    @property
    def k_TD(self) -> float:
        """Effective ATP->ADP exchange rate.
        
        It is constrained by the ATP/ADP exchange model.
        """
        return self.k_DT * self.K_d_atp / self.K_d_adp / self.atp_adp_ratio                    
    
    def gillespie(
        self, 
        n_steps: int = 1000, 
        initial_state: None | str = None,
        n_simulations: int = 1,
        cumulative_sums: str | list[str] | None = None,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Simulate the stochastic system using the Gillespie algorithm.

        Simulate the stochastic evolution of a single particle evoluting on 
        the kinetic scheme using the Gillespie algorithm.

        Args:
            n_steps: The number of simulation steps.
            initial_state: The initial state of the system. If None, a random
                initial state is chosen.
            n_simulations: The number of simulations to do. If > 1, the returned
                object is a list of dataframes, one for each simulation.
            cumulative_sums: The edge attributes to compute the cumulative sum
                of. Each simulation result dataframe then contains new columns
                named 'attribute_name' with the cumulative sum of the specified
                edge attributes at each step, with the timestamp. It adds a
                row at the beginning with timestamp 0 and 0 value at each
                column. 
                If None, each simulation returns a dataframe with the timestamp 
                and the edge taken at each step with all its attributes.

        Returns:
            A dataframe (or a list of dataframes if n_simulations > 1) with the
            columns 'timestamp', 'edge' and the elements of 'cumulative_sums'.
            The 'edge' column contains tuples of the form 
            (state_from, state_to, attributes), where attributes is a dict of 
            the form 'attribute': value.
        """
        if n_steps < 1:
            raise ValueError(
                "The number of steps must be strictly greater than 0.")
        if initial_state and initial_state not in self.kinetic_scheme.nodes():
            raise ValueError("The initial state does not belong to the graph.")
        if n_simulations < 1:
            raise ValueError(
                "The number of simulations must be strictly greater than 0.")

        results = []
        for _ in range(n_simulations):
            result = {'timestamp': [], 'edge': []}

            time = 0
            if initial_state:
                state = initial_state
            else:
                sorted_nodes = sorted(self.kinetic_scheme.nodes())
                probabilities = [
                    self.kinetic_scheme.nodes[node]['probability']() 
                    for node in sorted_nodes]
                state = np.random.choice(sorted_nodes, p=probabilities)
            for _ in range(n_steps):
                # Each step the system starts in the current state and then 
                # after a sojourn time given by an exponential distribution
                # with parameter the sum of the leaving rates of the current 
                # state, it goes to the next state chosen with probability 
                # proportional to the leaving rates.
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

                result['timestamp'].append(time)
                result['edge'].append(chosen_edge)
            result = pd.DataFrame(result)
            if cumulative_sums:
                self._compute_cumulative_sums(result, cumulative_sums)
            results.append(result)

        if n_simulations == 1:
            out = results[0]
        else:
            out = results

        return out

    def plot_position_evolution(
        self, 
        trajectory: pd.DataFrame | list[pd.DataFrame],
        time_unit: str = "a.u.", 
        position_unit: str = "Residues", 
        title: str | None = None,
        kinetic_scheme_image_path: str | None = None, 
        ax: mpl.axes.Axes | None = None,
    ) -> mpl.axes.Axes:
        """Plot the evolution of the position.

        Plot the expected position given by the average velocity.

        Args:
            trajectory: Position and timestamp at every changement of the 
                position (in residues, not steps). It must have these two
                columns: 'timestamp' and 'position'. It can also be a list of
                dataframes, one for each simulation. In this case, every 
                trajectory is plotted on the same axes.
            time_unit: Unit of the time (x-)axis
            position_unit: Unit of the position (y-)axis
            title: The title of the plot. If None, no title is added.
            kinetic_scheme_image_path: If given, will add the image of the
                kinetic scheme on the plot.
            ax: The axes where to plot. If None, a new axes is created.

        Returns:
            The axes with the plot.
        """
        if not ax:
            _, ax = plt.subplots()
        
        # Single or multiple trajectories handled the same way in a list
        trajectories = (
            trajectory 
            if isinstance(trajectory, list) 
            else [trajectory])
        for i, trajectory in enumerate(trajectories):
            label = "From Gillespie algorithm" if i == 0 else None
            ax.step(trajectory['timestamp'], trajectory['position'], 
                    where="post", label=label, color='C0')
        ax.plot(trajectories[0]['timestamp'], 
                trajectories[0]['timestamp'] * self.average_velocity(), 
                label="From average velocity", color='C3')
        ax.set_xlabel("Time" + " [" + time_unit + "]")
        ax.set_ylabel("Position" + " [" + position_unit + "]")
        ax.legend()
        if kinetic_scheme_image_path:
            img = np.asarray(Image.open(kinetic_scheme_image_path))
            sub_ax = ax.inset_axes([0.55, 0., 0.44, 0.44])
            sub_ax.imshow(img)
            sub_ax.axis('off')

        if title:
            ax.set_title(title)

        return ax
    
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
    
    def normalize_average_velocity(self, inplace: bool = True
    ) -> DiGraph | None: 
        """Normalize the average velocity of the translocation model.

        Update all the rates so that the average velocity is 1.
        Useful to compare different models.
        """
        if self.average_velocity() == 0:
            raise ValueError("The average velocity is null, cannot normalize.")
        
        average_velocity = self.average_velocity() # Velocity before normalization
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
            timestamps: float | list[float],
            confidence_level: float = 0.95,

    ) -> pd.DataFrame:
        """Return mean, std and confidence interval of the edge attribute.
        
        Return the mean, standard deviation (std) and confidence interval
        (CI) at the specified confidence level of the specified edge attribute. 
        The confidence interval (CI) is computed with gaussian approximation.

        Args:
            edge_attribute: The edge attribute to compute the statistics of.
            timestamps: The timestamp(s) to compute the statistics at.
            confidence_level: The confidence level of the confidence interval. 
                Only relevant if confidence_interval is True.
        
        Returns:
            Pandas dataframe with the columns 'timestamp', 'mean', 'std',
            'lower_bound', 'upper_bound'.
        """
        if isinstance(timestamps, float):
            timestamps = [timestamps]
        
        mean = 0
        var = 0
        for u, v, value in self.kinetic_scheme.edges(data=edge_attribute, 
                                                     default=0):
            p = self.kinetic_scheme.nodes[u]['probability']()
            k = self.kinetic_scheme.edges[u, v]['rate']()
            mean += value * p * k
            var += value**2 * p * k
        mean = timestamps * mean
        var = timestamps * var
        std = np.sqrt(var)
        lower_bound, upper_bound = norm.interval(confidence_level, mean, std)
        return pd.DataFrame({
            'timestamp': timestamps,
            'mean': mean,
            'std': std,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })

    def empirical_attribute_stats(
        self,
        edge_attribute: str,
        confidence_level: float = 0.95,
        n_steps: int = 100,
        n_simulations: int = 100,
    ) -> pd.DataFrame:
        """Return mean, std and confidence interval of the edge attribute.
        
        Return the empirical mean, standard deviation (std) and confidence 
        interval (CI) at the specified confidence level of the specified edge 
        attribute (e.g. 'position'). The confidence interval (CI) is computed 
        with gaussian approximation.
        The statistics is computed at all timestamps corresponding to a step 
        in any simulation, up to the end of the shortest simulation.

        Args:
            edge_attribute: The edge attribute to compute the statistics of.
            confidence_level: The confidence level of the confidence interval. 
            n_steps: The number of simulation steps for each simulation.
            n_simulations: The number of simulations to do to compute the
                empirical statistics.
        
        Returns:
            Pandas dataframe with the columns 'timestamp', 'mean', 'std',
            'lower_bound', 'upper_bound'.
        """
        if n_simulations < 2:
            raise ValueError(
                "The number of simulations must be at least 2.")
        
        trajectories = self.gillespie(
            n_steps=n_steps,
            n_simulations=n_simulations,
            cumulative_sums=edge_attribute,
        )
        # We take all timestamps of each simulation up to the end of the 
        # shortest simulation
        max_time = min([trajectory['timestamp'].iat[-1] 
                        for trajectory in trajectories])
        timestamps = pd.concat(trajectories)['timestamp'].unique()
        timestamps = timestamps[timestamps <= max_time]
        # We keep only the edge attribute and timestamp columns and we fill
        # the dataframe at all timestamp above with the last valid value
        for i in range(len(trajectories)):
            trajectories[i] = (trajectories[i]
                               .loc[:, ['timestamp', edge_attribute]]
                               .set_index('timestamp')
                               .reindex(timestamps, method='ffill')
                               .reset_index())
        # We compute the statistics at each timestamp
        statistics = (pd.concat(trajectories)
                      .groupby('timestamp')[edge_attribute]
                      .agg(['mean', 'std']))
        q_lower, q_upper = norm.interval(confidence_level)
        statistics['lower_bound'] = (
            statistics['mean'] + q_lower * statistics['std'])
        statistics['upper_bound'] = (
            statistics['mean'] + q_upper * statistics['std'])
        statistics.reset_index(inplace=True)
        statistics['timestamp'] = timestamps
        return statistics

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
        cumulative_sums: str | list[str],
    ) -> None:
        """Compute the cumulative sum of the specified edge attributes.

        The post-processing consists in computing the cumulative sum of the
        specified edge attributes at each step. 
        The result is updated in-place.

        Args:
            result: A dataframe with the columns 'timestamp' and 'edge'. The
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
        # Add a row at the beginning with timestamp 0 and value 0 at each column
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
        self.k_up = 1 # Translocation up rate
        super().__init__(atp_adp_ratio)
        #self.kinetic_scheme = self._construct_kinetic_scheme()
    
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
    

class SC2R2Loops(SC2R):
    """Sequential Clockwise/2-Residue Step, 2-Loops translocation model."""

    def __init__(self, atp_adp_ratio: float = 10) -> None:
        super().__init__(atp_adp_ratio)
        #self.kinetic_scheme = self._construct_kinetic_scheme()
    
    def _construct_kinetic_scheme(self, kinetic_scheme: DiGraph | None = None
    ) -> DiGraph:
        if not kinetic_scheme:
            kinetic_scheme = DiGraph()
        kinetic_scheme = super()._construct_kinetic_scheme(kinetic_scheme)
        kinetic_scheme.add_node(
            'DTD', probability = lambda: self._compute_probabilities()['DTD'])
        kinetic_scheme.add_edges_from([
            ('DTT', 'DTD', {'rate': lambda: self.k_TD}),
            ('DTD', 'DTT', {'rate': lambda: self.k_DT}),
            ('DTD', 'TTD', {'rate': lambda: self.k_s, 'ATP': 1}),
            ('TTD', 'DTD', {'rate': lambda: self.k_h, 'ATP': -1})
        ])
        return kinetic_scheme


class DiscSpiral(TranslocationModel):
    """Disc-Spiral translocation model.
    
    Physical parameters:
        k_[extended/flat]_to_[flat/extended]_[up/down]
    """

    def __init__(self, atp_adp_ratio: float = 10, n_protomers: int = 6) -> None:
        self.n_protomers = n_protomers
        self.k_extended_to_flat_up = 1 # Spiral->disc up translocation rate
        self.k_flat_to_extended_down = 1 # Disc->spiral down translocation rate
        self.k_flat_to_extended_up = 1 # Disc->spiral up translocation rate
        super().__init__(atp_adp_ratio)
        #self.kinetic_scheme = self._construct_kinetic_scheme()
    
    @property
    def k_h_bar(self) -> float:
        """Effective ATP hydrolisis rate.
        
        Each protomer contributes k_h to the effective hydrolisis rate.
        """
        return self.n_protomers * self.k_h

    @property
    def k_flat_to_extended_down_bar(self) -> float:
        """Effective disc->spiral down translocation rate.
        
        Each protomer contributes k_flat_to_extended_down to the effective 
        translocation rate.
        """
        return self.n_protomers * self.k_flat_to_extended_down
    
    @property
    def k_extended_to_flat_down(self) -> float:
        """Spiral->disc down translocation rate, constrained by the detailed balance."""
        return ((self.k_h * self.k_flat_to_extended_up * self.k_DT 
                 * self.k_extended_to_flat_up)
                / (self.k_s * self.k_TD * self.k_flat_to_extended_down)
                * (self.equilibrium_atp_adp_ratio / self.atp_adp_ratio))

    def _construct_kinetic_scheme(self, kinetic_scheme: DiGraph | None = None
    ) -> DiGraph:
        if not kinetic_scheme:
            kinetic_scheme = DiGraph()
        kinetic_scheme.add_nodes_from([
            ('flat-ATP', {'probability': 
                        lambda: self._compute_probabilities()['flat-ATP']}),
            ('flat-ADP', {'probability':
                        lambda: self._compute_probabilities()['flat-ADP']}),
            ('extended-ADP', {'probability': 
                        lambda: self._compute_probabilities()['extended-ADP']}),
            ('extended-ATP', {'probability': 
                        lambda: self._compute_probabilities()['extended-ATP']})
        ])
        kinetic_scheme.add_edges_from([ # ⤴⤵⤷↳↱ 
            ('flat-ATP', 'flat-ADP', {'rate': lambda: self.k_h_bar, 'ATP': -1}),
            ('flat-ADP', 'flat-ATP', {'rate': lambda: self.k_s, 'ATP': 1}),
            ('flat-ADP', 'extended-ADP', { # k_⤴
                'rate': lambda: self.k_flat_to_extended_up,
                'position': 2*(self.n_protomers-1)}),
            ('extended-ADP', 'flat-ADP', { # k_↳
                'rate': lambda: self.k_extended_to_flat_down,
                'position': -2*(self.n_protomers-1)}),
            ('extended-ADP', 'extended-ATP', {'rate': lambda: self.k_DT}),
            ('extended-ATP', 'extended-ADP', {'rate': lambda: self.k_TD}),
            ('extended-ATP', 'flat-ATP', { # k_↱
                'rate': lambda: self.k_extended_to_flat_up}),
            ('flat-ATP', 'extended-ATP', { # k_⤵
                'rate': lambda: self.k_flat_to_extended_down_bar})
        ])
        return kinetic_scheme


class DefectiveSC2R(SC2R):
    """Sequential Clockwise/2-Residue Step with one defective protomer.
    
    Single-loop-like translocation model with one defective protomer. The 
    defective protomer has an hydrolisis rate that is defect_factor times 
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
            defect_factor: The factor by which the defective protomer hydrolisis
                rate is smaller than the other protomers hydrolisis rate.
            atp_adp_ratio: The ATP/ADP ratio.
            n_protomers: The number of protomers.
        """
        self.defect_factor = defect_factor
        self.n_protomers = n_protomers
        super().__init__(atp_adp_ratio)
        # Redundant kinetic_scheme construction, but more explicit
        #self.kinetic_scheme = self._construct_kinetic_scheme()

    @property
    def k_h_defect(self) -> float:
        """ATP hydrolisis rate of the defective protomer."""
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


class DefectiveDiscSpiral(DiscSpiral):
    """Disc-spiral model with one protomer with defective hydrolisis.
    
    The defective protomer has an hydrolisis rate that is defect_factor times 
    smaller than the other protomers.

    The states in defect-free loops are denoted by:
        'flat-ATP', 'flat-ADP', 'extended-ATP', 'extended-ADP'
    and in the defective loop by:
        'flat-ADP-defect', 'extended-ATP-defect', 'extended-ADP-defect'

    There are n_protomers-1 defect-free loops and one defective loop. For 
    simplicity of the code, we aggregate all the defect-free loops together in a
    single loop with an effective hydrolisis rate k_h_bar = (n_protomers-1)*k_h, 
    and an effective disc->spiral down translocation rate 
    k_flat_to_extended_down_bar = (n_protomers-1)*k_flat_to_extended_down.
    """

    def __init__(
            self, 
            defect_factor: float = 0.1, 
            atp_adp_ratio: float = 10, 
            n_protomers: int = 6
    ) -> None:
        """Initialize the defective translocation model.
        
        Args:
            defect_factor: The factor by which the defective protomer hydrolisis
                rate is smaller than the other protomers hydrolisis rate.
        """
        self.defect_factor = defect_factor
        super().__init__(atp_adp_ratio, n_protomers)
        # Redundant kinetic_scheme construction, but more explicit
        #self.kinetic_scheme = self._construct_kinetic_scheme() 

    @property
    def k_h_defect(self) -> float:
        """ATP hydrolisis rate of the defective protomer."""
        return self.defect_factor * self.k_h
    
    @property
    def k_s_defect(self) -> float:
        """ATP synthesis rate of the defective protomer."""
        return self.defect_factor * self.k_s
    
    @property
    def k_h_bar(self) -> float:
        """Effective ATP hydrolisis rate for defect-free loop.
        
        Each defect-free protomer contributes k_h to the effective hydrolisis 
        rate.
        """
        return (self.n_protomers - 1) * self.k_h
    
    @property
    def k_flat_to_extended_down_bar(self) -> float:
        """Effective disc->spiral down translocation rate.
        
        Each defect-free protomer contributes k_flat_to_extended_down to the 
        effective translocation rate.
        """
        return (self.n_protomers - 1) * self.k_flat_to_extended_down
    
    # No need to redefine k_extended_to_flat_down, the detailed balance 
    # constraint is the same as in the defect-free case.

    def probabilities_defect_ignored(self) -> dict[str, float]:
        """Compute the total probabilities to be in each defect-ignored state.
        
        Defect-ignored mean that we do not differenciate the defect-free 
        protomers from the defective one.
        The states are then:
            'flat-ATP', 'flat-ADP', 'extended-ATP', 'extended-ADP'
        """
        probabilities = self._compute_probabilities()
        probabilities_defect_ignored = {
            'flat-ATP': probabilities['flat-ATP'],
            'flat-ADP': (probabilities['flat-ADP'] 
                         + probabilities['flat-ADP-defect']),
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
            ('flat-ADP-defect', 
             {'probability': 
                 lambda: self._compute_probabilities()['flat-ADP-defect']}),
            ('extended-ATP-defect', 
             {'probability':
                 lambda: self._compute_probabilities()['extended-ATP-defect']}),
            ('extended-ADP-defect', 
             {'probability': 
                 lambda: self._compute_probabilities()['extended-ADP-defect']})
        ])
        kinetic_scheme.add_edges_from([
            ('flat-ATP', 'flat-ADP-defect', { # Defective hydrolisis
                'rate': lambda: self.k_h_defect, 
                'ATP': -1}),
            ('flat-ADP-defect', 'flat-ATP', { # Defective synthesis
                'rate': lambda: self.k_s_defect, 
                'ATP': 1}),
            ('flat-ADP-defect', 'extended-ADP-defect', { # k_⤴
                'rate': lambda: self.k_flat_to_extended_up,
                'position': 2*(self.n_protomers-1)}),
            ('extended-ADP-defect', 'flat-ADP-defect', { # k_↳
                'rate': lambda: self.k_extended_to_flat_down,
                'position': -2*(self.n_protomers-1)}),
            ('extended-ADP-defect', 'extended-ATP-defect', { # ADP->ATP exchange
                'rate': lambda: self.k_DT}),
            ('extended-ATP-defect', 'extended-ADP-defect', { # ATP->ADP exchange
                'rate': lambda: self.k_TD}),
            ('extended-ATP-defect', 'flat-ATP', { # k_↱
                'rate': lambda: self.k_extended_to_flat_up}),
            ('flat-ATP', 'extended-ATP-defect', { # k_⤵
                'rate': lambda: self.k_flat_to_extended_down})
        ])
        return kinetic_scheme


