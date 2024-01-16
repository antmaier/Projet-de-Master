from ipywidgets import GridspecLayout, BoundedFloatText, BoundedIntText, Label, HTML
from sys import maxsize, float_info


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

    def __init__(self, n_rows):
        self.n_rows = n_rows
        self.n_cols = 3
        self.grid = GridspecLayout(self.n_rows, self.n_cols)
        self.parameters = {}

    def add_general_parameters(self, row: int = 0) -> None:
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
        self.grid[row, :] = HTML(value="<b>General Physical Parameters</b>")

        atp_adp_ratio = self._add_parameter(
            row + 1, "[ATP]/[ADP]", "ATP/ADP concentration ratio", value=10.0)
        
        equilibrium_atp_adp_ratio = self._add_parameter(
            row + 2, "([ATP]/[ADP])|eq.", 
            "Equilibrium ATP/ADP concentration ratio", value=0.01)
        
        K_d_atp = self._add_parameter(
            row + 3, "K_d^ATP", "Protomer-ATP dissociation constant", value=0.1)
        
        K_d_adp = self._add_parameter(
            row + 4, "K_d^ADP", "Protomer-ADP dissociation constant")
        
        k_DT = self._add_parameter(
            row + 5, "k_DT", "Effective ADP->ATP exchange rate")
        
        def compute_k_TD():
            return (k_DT.value * K_d_atp.value / K_d_adp.value 
                    / atp_adp_ratio.value)
        k_TD = self._add_parameter(
            row + 6, "k_TD", "Effective ATP->ADP exchange rate "\
                "(constrained by ATP<->ADP exchange model)", 
            value=compute_k_TD(), type='constrained')
        def change_k_TD(_): 
            k_TD.value = str(compute_k_TD())
        k_DT.observe(change_k_TD, names='value')
        K_d_atp.observe(change_k_TD, names='value')
        K_d_adp.observe(change_k_TD, names='value')
        atp_adp_ratio.observe(change_k_TD, names='value')

        k_h = self._add_parameter(row + 7, "k_h", "ATP hydrolysis rate")

        k_s = self._add_parameter(
            row + 8, "k_s", "ATP synthesis rate", value=0.1)

        self.parameters.update({
            "atp_adp_ratio": atp_adp_ratio,
            "equilibrium_atp_adp_ratio": equilibrium_atp_adp_ratio,
            "K_d_atp": K_d_atp,
            "K_d_adp": K_d_adp,
            "k_DT": k_DT,
            "k_TD": k_TD,
            "k_h": k_h,
            "k_s": k_s,
        })

    def add_sc2r_parameters(self, row: int) -> None:
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
        self.grid[row, :] = HTML(value="<b>Sequential Clockwise/2-Residue "\
                                 "Step Model Physical Parameters</b>")
        
        k_up = self._add_parameter(row + 1, "k_↑", "Translocation up rate")
        
        def compute_k_down():
            return (
                (self.parameters["k_h"].value * k_up.value 
                 * self.parameters["k_DT"].value)
                / (self.parameters["k_s"].value 
                   * float(self.parameters["k_TD"].value)) # Because k_TD is a Label
                * (self.parameters["equilibrium_atp_adp_ratio"].value 
                   / self.parameters["atp_adp_ratio"].value)
            )
        k_down = self._add_parameter(
            row + 2, "k_↓", 
            "Translocation down rate (constrained by the detailed balance)",
            value=compute_k_down(), type='constrained')
        def change_k_down(_):
            k_down.value = str(compute_k_down())
        k_up.observe(change_k_down, names='value')
        self.parameters['atp_adp_ratio'].observe(change_k_down, names='value')
        self.parameters['equilibrium_atp_adp_ratio'].observe(change_k_down, 
                                                             names='value')
        self.parameters['k_h'].observe(change_k_down, names='value')
        self.parameters['k_s'].observe(change_k_down, names='value')
        self.parameters['k_DT'].observe(change_k_down, names='value')

        self.parameters.update({
            "k_up": k_up,
            "k_down": k_down,
        })

    def add_disc_spiral_parameters(self, row: int) -> None:
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
        self.grid[row, 0] = HTML(
            value="<b>Disc-Spiral Model Physical Parameters</b>")
        
        n_protomers = self._add_parameter(
            row + 1, "n_protomers", "Number of protomers", value=6, type='int', 
            min=1)
        
        def compute_k_h_bar(): 
            return n_protomers.value * self.parameters["k_h"].value
        k_h_bar = self._add_parameter(
            row + 2, "ꝁ_h", "Effective ATP hydrolysis rate", 
            value=compute_k_h_bar(), type='constrained')
        def change_k_h_bar(_):
            k_h_bar.value = str(compute_k_h_bar())
        n_protomers.observe(change_k_h_bar, names='value')
        self.parameters['k_h'].observe(change_k_h_bar, names='value')

        k_extended_to_flat_up = self._add_parameter(
            row + 3, "k_⮫", "Spiral->disc up translocation rate")
        
        k_flat_to_extended_down = self._add_parameter(
            row + 4, r"k_⮯", "Disc->spiral down translocation rate")
        
        def compute_k_flat_to_extended_down_bar():
            return n_protomers.value * k_flat_to_extended_down.value
        k_flat_to_extended_down_bar = self._add_parameter(
            row + 5, "ꝁ_⮯", "Effective disc->spiral down translocation rate", 
            value=compute_k_flat_to_extended_down_bar(), type='constrained')
        def change_k_flat_to_extended_down_bar(_):
            k_flat_to_extended_down_bar.value = str(
                compute_k_flat_to_extended_down_bar())
        n_protomers.observe(change_k_flat_to_extended_down_bar, names='value')
        k_flat_to_extended_down.observe(change_k_flat_to_extended_down_bar,
                                        names='value')

        k_flat_to_extended_up = self._add_parameter(
            row + 6, r"k_⮭", "Disc->spiral up translocation rate")
        
        def compute_k_extended_to_flat_down():
            return ((self.parameters["k_h"].value * k_flat_to_extended_up.value 
                     * self.parameters["k_DT"].value 
                     * k_extended_to_flat_up.value)
                    / (self.parameters["k_s"].value 
                       * float(self.parameters["k_TD"].value) 
                       * k_flat_to_extended_down.value)
                    * (self.parameters["equilibrium_atp_adp_ratio"].value 
                       / self.parameters["atp_adp_ratio"].value))
        k_extended_to_flat_down = self._add_parameter(
            row + 7, r"k_⮩", "Spiral->disc down translocation rate "\
                "(constrained by the detailed balance)",
            value=compute_k_extended_to_flat_down(), type='constrained')
        def change_k_extended_to_flat_down(_):
            k_extended_to_flat_down.value = str(compute_k_extended_to_flat_down())
        self.parameters['k_h'].observe(change_k_extended_to_flat_down, 
                                       names='value')
        self.parameters['k_s'].observe(change_k_extended_to_flat_down, 
                                       names='value')
        self.parameters['k_DT'].observe(change_k_extended_to_flat_down, 
                                        names='value')
        k_extended_to_flat_up.observe(change_k_extended_to_flat_down, 
                                      names='value')
        k_flat_to_extended_down.observe(change_k_extended_to_flat_down,
                                        names='value')
        k_flat_to_extended_up.observe(change_k_extended_to_flat_down, 
                                      names='value')
        self.parameters['atp_adp_ratio'].observe(change_k_extended_to_flat_down, 
                                                 names='value')
        self.parameters['equilibrium_atp_adp_ratio'].observe(
            change_k_extended_to_flat_down, names='value')
        
        self.parameters.update({
            "n_protomers": n_protomers,
            "k_h_bar": k_h_bar,
            "k_extended_to_flat_up": k_extended_to_flat_up,
            "k_flat_to_extended_down": k_flat_to_extended_down,
            "k_flat_to_extended_down_bar": k_flat_to_extended_down_bar,
            "k_flat_to_extended_up": k_flat_to_extended_up,
            "k_extended_to_flat_down": k_extended_to_flat_down,
        })

    def add_defect_sc2r_parameters(self, row: int) -> None:
        pass # TODO

    def _add_parameter(
        self,
        row: int,
        symbol: str,
        desc: str,
        value: float = 1.0,
        type: str = 'float',
        min: float = 0.0,
    ) -> None:
        """Add a parameter to the grid.

        The parameter has a symbol, a value, and a description. If the parameter 
        is not constrained, the value is a FloatText widget. If the parameter is
        constrained, the value is a Label widget.

        Args:
            row: The row to add the parameter to.
            symbol: The LaTeX symbol of the parameter.
            desc: A description of the parameter.
            value: The initial value of the parameter.
            type: The type of the parameter. Either 'float', 'int' or 
                'constrained'.
            min: The minimum value of the parameter. Not relevant if type is
                'constrained'.
        """
        type = type.lower()
        if type == 'float':
            parameter = BoundedFloatText(
                value=value, min=min, max=float_info.max, 
                description=symbol + ':')
        elif type == 'int':
            parameter = BoundedIntText(
                value=value, min=min, max=maxsize, 
                description=symbol + ':')
        elif type == 'constrained':
            parameter = HTML(value=str(value), description=symbol + ':')
        else:
            raise ValueError(
                "type must be either 'float', 'int' or 'constrained'")
        self.grid[row, 0] = parameter
        self.grid[row, 1:] = Label(value=desc)
        return parameter


