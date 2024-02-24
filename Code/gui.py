import ipywidgets as widgets
from ipywidgets import Label, HTML, FloatLogSlider, IntSlider, HBox
from sys import maxsize


# TODO update docstrings
# TODO round values in constrained parameters
class GUI():
    """A GUI for the translocation model experiments.

    The GUI is a GridspecLayout of widgets. It contains widgets to control the 
    physical parameters of the translocation model and the parameters of the
    experimental setup. 
    Each row is either a description, or a parameter. Each parameter row 
    contains a symbol, a value, and a description.
    Some parameters are constrained by other parameters. In this case, the value
    is displayed and cannot be directly, and its value is dynamically updated
    when the other parameters change.

    The GUI is accessible via the 'grid' attribute. All the parameter widgets
    are accessible via the 'parameters' attribute, a dictionary of the form
    {symbol: widget}.

    The number of rows of the grid has to be specified at initialization. You 
    can access the number of rows for each section of the grid in the docstring
    of the corresponding method, e.g. Gui.add_general_parameters.__doc__.

    Example:
        gui = Gui(12)
        gui.add_general_parameters(0)
        gui.add_SC2R_parameters(9)
        IPython.display.display(gui.grid)
    """

    def __init__(self):
        self.parameters = {}
        self.gui = widgets.VBox()

    def add_general_parameters(self) -> None:
        """Add write/read interface for general physical parameters (9 rows).

        General parameters are the physical parameters common to all models.
        
        Add inplace the parameters to the given grid and the 
        {symbol: widget_value} parameters dictionary.
        
        The interface is 9 rows long starting at the given row and expanding 
        downwards. 
        Each parameter row contains a symbol, a value, and a description. 
        The rows are:
        0: "General physical parameters"
        1: ATP/ADP concentration ratio (atp_adp_ratio)
        2: Equilibrium ATP/ADP concentration ratio (equilibrium_atp_adp_ratio)
        3: Protomer-ATP dissociation constant (K_d_atp)
        4: Protomer-ADP dissociation constant (K_d_adp)
        5: Effective ADP->ATP exchange rate (k_DT)
        6: Effective ATP->ADP exchange rate (k_TD) (constrained by ATP<->ADP 
            exchange model)
        7: ATP hydrolysis rate (k_h)
        8: ATP synthesis rate (k_s)
        """
        self.gui.children += (HTML(value="<b>General Physical Parameters</b>"),)

        self._add_parameter(
            'atp_adp_ratio', "[ATP]/[ADP]", "ATP/ADP concentration ratio", 
            value=10.0)
        
        self._add_parameter(
            'equilibrium_atp_adp_ratio', "([ATP]/[ADP])|eq.", 
            "Equilibrium ATP/ADP concentration ratio", value=0.01)
        
        self._add_parameter(
            'K_d_atp', "K_d^ATP", "Protomer-ATP dissociation constant", 
            value=0.1)
        
        self._add_parameter(
            'K_d_adp', "K_d^ADP", "Protomer-ADP dissociation constant")
        
        self._add_parameter('k_DT', "k_DT", "Effective ADP->ATP exchange rate")
        
        def compute_k_TD():
            return (self.parameters['k_DT'].value 
                    * self.parameters['K_d_atp'].value 
                    / self.parameters['K_d_adp'].value 
                    / self.parameters['atp_adp_ratio'].value)
        self._add_parameter(
            'k_TD', "k_TD", "Effective ATP->ADP exchange rate "\
                "(constrained by ATP<->ADP exchange model)", 
            value=compute_k_TD(), type='constrained')
        def change_k_TD(_): 
            self.parameters['k_TD'].value = str(compute_k_TD())
        self.parameters['k_DT'].observe(change_k_TD, names='value')
        self.parameters['K_d_atp'].observe(change_k_TD, names='value')
        self.parameters['K_d_adp'].observe(change_k_TD, names='value')
        self.parameters['atp_adp_ratio'].observe(change_k_TD, names='value')

        self._add_parameter('k_h', "k_h", "ATP hydrolysis rate")

        self._add_parameter('k_s', "k_s", "ATP synthesis rate", value=0.1)

    def add_sc2r_parameters(self) -> None:
        """Add write/read interface for SC2R parameters (3 rows).

        Add inplace the parameters to the given grid and the 
        {symbol: widget_value} parameters dictionary.

        The parameters dictionary must contain the general parameters.

        The interface is 3 rows long starting at the given row and expanding 
        downwards. 
        Each parameter row contains a symbol, a value, and a description.
        The rows are:
        0: "SC2R parameters"
        1: Translocation up rate (k_up)
        2: Translocation down rate (k_down) (constrained by the detailed balance)
        """
        self.gui.children += (HTML(
            value="<b>SC2R Model Physical Parameters</b>"),)
        
        self._add_parameter('k_up', "k_↑", "Translocation up rate")
        
        def compute_k_down():
            return (
                (self.parameters['k_h'].value 
                 * self.parameters['k_up'].value 
                 * self.parameters['k_DT'].value)
                / (self.parameters['k_s'].value 
                   * float(self.parameters['k_TD'].value)) # Because k_TD is a Label
                * (self.parameters['equilibrium_atp_adp_ratio'].value 
                   / self.parameters['atp_adp_ratio'].value)
            )
        self._add_parameter(
            'k_down', "k_↓", 
            "Translocation down rate (constrained by the detailed balance)",
            value=compute_k_down(), type='constrained')
        def change_k_down(_):
            self.parameters['k_down'].value = str(compute_k_down())
        self.parameters['k_up'].observe(change_k_down, names='value')
        self.parameters['atp_adp_ratio'].observe(change_k_down, names='value')
        self.parameters['equilibrium_atp_adp_ratio'].observe(change_k_down, 
                                                             names='value')
        self.parameters['k_h'].observe(change_k_down, names='value')
        self.parameters['k_s'].observe(change_k_down, names='value')
        self.parameters['k_DT'].observe(change_k_down, names='value')

    def add_disc_spiral_parameters(self) -> None:
        """Add write/read interface for Disc-Spiral parameters (8 rows).

        Add inplace the parameters to the given grid and the 
        {symbol: widget_value} parameters dictionary.

        The parameters dictionary must contain the general parameters.

        The gris is 8 rows long starting at the given row expanding downwards. 
        Each parameter row contains a symbol, a value, and a description.
        The rows are:
        k_[extended/flat]_to_[flat/extended]_[up/down]
        0: "Disc-Spiral Model"
        1: Number of protomers (n_protomers)
        2: Effective ATP hydrolysis rate (k_h_bar)
        3: Spiral->disc up translocation rate (k_extended_to_flat_up)
        4: Disc->spiral down translocation rate (k_flat_to_extended_down)
        5: Effective disc->spiral down translocation rate (k_flat_to_extended_down_bar)
        6: Disc->spiral up translocation rate (k_flat_to_extended_up)
        7: Spiral->disc down translocation rate (k_extended_to_flat_down) (
            constrained by the detailed balance)
        """
        self.gui.children += (HTML(
            value="<b>Disc-Spiral Model Physical Parameters</b>"),)
        
        self._add_parameter(
            'n_protomers', "n_protomers", "Number of protomers", value=6, 
            type='int', min=1, max=10)
        
        def compute_k_h_bar(): 
            return (self.parameters['n_protomers'].value 
                    * self.parameters['k_h'].value)
        k_h_bar = self._add_parameter(
            'k_h_bar', "ꝁ_h", "Effective ATP hydrolysis rate", 
            value=compute_k_h_bar(), type='constrained')
        def change_k_h_bar(_):
            self.parameters['k_h_bar'].value = str(compute_k_h_bar())
        self.parameters['n_protomers'].observe(change_k_h_bar, names='value')
        self.parameters['k_h'].observe(change_k_h_bar, names='value')

        self._add_parameter(
            'k_extended_to_flat_up', "k_⮫", 
            "Spiral->disc up translocation rate")
        
        self._add_parameter(
            'k_flat_to_extended_down', r"k_⮯", 
            "Disc->spiral down translocation rate")
        
        def compute_k_flat_to_extended_down_bar():
            return (self.parameters['n_protomers'].value 
                    * self.parameters['k_flat_to_extended_down'].value)
        k_flat_to_extended_down_bar = self._add_parameter(
            'k_flat_to_extended_down_bar', "ꝁ_⮯", 
            "Effective disc->spiral down translocation rate", 
            value=compute_k_flat_to_extended_down_bar(), type='constrained')
        def change_k_flat_to_extended_down_bar(_):
            self.parameters['k_flat_to_extended_down_bar'].value = str(
                compute_k_flat_to_extended_down_bar())
        self.parameters['n_protomers'].observe(
            change_k_flat_to_extended_down_bar, names='value')
        self.parameters['k_flat_to_extended_down'].observe(
            change_k_flat_to_extended_down_bar, names='value')

        self._add_parameter(
            'k_flat_to_extended_up', r"k_⮭", 
            "Disc->spiral up translocation rate")
        
        def compute_k_extended_to_flat_down():
            return ((self.parameters["k_h"].value 
                     * self.parameters['k_flat_to_extended_up'].value 
                     * self.parameters["k_DT"].value 
                     * self.parameters['k_extended_to_flat_up'].value)
                    / (self.parameters["k_s"].value 
                       * float(self.parameters["k_TD"].value) 
                       * self.parameters['k_flat_to_extended_down'].value)
                    * (self.parameters["equilibrium_atp_adp_ratio"].value 
                       / self.parameters["atp_adp_ratio"].value))
        self._add_parameter(
            'k_extended_to_flat_down', r"k_⮩", "Spiral->disc down "\
                "translocation rate (constrained by the detailed balance)",
            value=compute_k_extended_to_flat_down(), type='constrained')
        def change_k_extended_to_flat_down(_):
            self.parameters['k_extended_to_flat_down'].value = str(
                compute_k_extended_to_flat_down())
        self.parameters['k_h'].observe(change_k_extended_to_flat_down, 
                                       names='value')
        self.parameters['k_s'].observe(change_k_extended_to_flat_down, 
                                       names='value')
        self.parameters['k_DT'].observe(change_k_extended_to_flat_down, 
                                        names='value')
        self.parameters['k_extended_to_flat_up'].observe(
            change_k_extended_to_flat_down, names='value')
        self.parameters['k_flat_to_extended_down'].observe(
            change_k_extended_to_flat_down, names='value')
        self.parameters['k_flat_to_extended_up'].observe(
            change_k_extended_to_flat_down, names='value')
        self.parameters['atp_adp_ratio'].observe(
            change_k_extended_to_flat_down, names='value')
        self.parameters['equilibrium_atp_adp_ratio'].observe(
            change_k_extended_to_flat_down, names='value')

    def add_defect_sc2r_parameters(self, row: int) -> None:
        pass # TODO

    def _add_parameter(
        self,
        var_name: str,
        symbol: str,
        desc: str,
        value: float = 1.0,
        type: str = 'float',
        min: int = None,
        max: int = None,
    ) -> None:
        """Add a parameter to the gui and widget to the parameters dictionary.

        If type is 'float', the parameter is a FloatLogSlider with min/max 
        exponents. If type is 'int', the parameter is an IntSlider with min/max
        values. If type is 'constrained', the parameter is an HTML widget. The
        dynamical update of the constrained parameter is not handled by this
        method.

        Args:
            var_name: The name of the parameter attribute in the model.
            symbol: The LaTeX symbol of the parameter.
            desc: A description of the parameter.
            value: The initial value of the parameter.
            type: The type of the parameter. Either 'float', 'int' or 
                'constrained'.
            min: The minimum value. If type is 'float', min is the minimum 
                exponent, not min value. If type is not 'float', the min value 
                is the minimum value.
            max: The maximum value. If type is 'float', max is the maximum
                exponent, not max value. If type is not 'float', the max value
                is the maximum value.
        """
        type = type.lower()
        if min is None:
            min = -2 if type == 'float' else 0
        if max is None:
            max = 2 if type == 'float' else maxsize
        if type == 'float':
            parameter = FloatLogSlider(
                value=value, min=min, max=max, description=symbol + ':', 
                readout_format='.2f')
        elif type == 'int':
            parameter = IntSlider(
                value=value, min=min, max=max, description=symbol + ':', 
                readout_format='d')
        elif type == 'constrained':
            parameter = HTML(value=str(value), description=symbol + ':')
        else:
            raise ValueError(
                "type must be either 'float', 'int' or 'constrained'")
        hbox = HBox([parameter, Label(value=desc)])
        self.gui.children += (hbox,)
        self.parameters.update({var_name: parameter})


