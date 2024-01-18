from translocation_models import TranslocationModel, \
    SC2R, SC2R2Loops, DefectiveSC2R, \
    DiscSpiral, DefectiveDiscSpiral

from ipywidgets import Widget, FloatLogSlider, IntSlider, \
    HBox, VBox, HTML, Output, Layout
from IPython.display import display

import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class Experiment(ABC):
    """Base class for experiments.
    
    An experiment contains typically a GUI and one or more plots. The GUI 
    contains widgets that can be used to change the parameters of the model.
    The plots are dynamically updated when the user changes the parameters.

    To create a new experiment, inherit from this class and implement the
    following methods:
    - _construct_free_parameters: construct free parameters widgets
    - _construct_constrained_parameters: construct constrained parameters widgets
    - _construct_gui: construct GUI
    - _run: run experiment

    The _run method should update the models free parameters from the free
    parameters widgets, and update the constrained parameters widgets from the
    model. It should also update the plots.

    The subclass constructor should call the superclass constructor AFTER 
    defining the attributes needed for the experiment (typically the 
    translocation models). The parameters are not defined in the constructor
    but they are automatically constructed by the super class constructor via
    the _construct_free_parameters and _construct_constrained_parameters.

    Free parameters are parameters that can be changed by the user, using for
    example sliders. Constrained parameters are parameters that are computed
    from free parameters, and are displayed using HTML widgets for example.
    """

    def __init__(self):
        self._free_parameters = self._construct_free_parameters()
        self._constrained_parameters = self._construct_constrained_parameters()
        self._gui = self._construct_gui()

        for _, widget in self._free_parameters.items():
            widget.observe(lambda _: self._run(), names='value')
        self._run()
        display(self._gui)

    @abstractmethod
    def _construct_free_parameters(self) -> dict[str, Widget]:
        """Construct free parameters widgets.
        
        Free parameters are parameters that can be changed by the user.
        """
        pass

    @abstractmethod
    def _construct_constrained_parameters(self) -> dict[str, Widget]:
        """Construct constrained parameters widgets.
        
        Constrained parameters are parameters that are computed from free
        parameters.
        """
        pass

    @abstractmethod
    def _construct_gui(self) -> Widget:
        """Construct GUI."""
        pass

    @abstractmethod
    def _run(self) -> None:
        """Run experiment."""
        pass

    def _update_models_free_parameters(
        self,
        models: TranslocationModel | list[TranslocationModel],
        free_parameters: dict[str, Widget]
    ) -> None:
        """Update model free parameters from widgets."""
        if not isinstance(models, list):
            models = [models]
        for model in models:
            for name, widget in free_parameters.items():
                if name in vars(model):
                    setattr(model, name, widget.value)

    def _update_gui_constrained_parameters(
        self,
        models: TranslocationModel | list[TranslocationModel],
        constrained_parameters: dict[str, Widget]
    ) -> None:
        """Update GUI constrained parameter widgets from model."""
        if not isinstance(models, list):
            models = [models]
        for name, widget in constrained_parameters.items():
            for model in models:
                if name in dir(model):
                    widget.value = str(round(getattr(model, name), 2))
                    break

# TODO add slider to chose range order of magnitude of ATP/ADP ratio
#   Maybe put it without proportional to equilibrium_atp_adp_ratio
# TODO add info about what is the range of k_TD depending on ATP/ADP ratio
class VelocityVSATPADPRatio(Experiment):
    """Velocity vs [ATP]/[ADP] experiment.

    Plot the average velocity of the two SC2R and Disc-Spiral models for various
    values of [ATP]/[ADP] ratio.
    """

    def __init__(self):
        self._sc2r = SC2R()
        self._disc_spiral = DiscSpiral()
        super().__init__()

    def _construct_free_parameters(self) -> dict[str, Widget]:
        return {
            # Source for ATP/ADP ratio:
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6395684/#:~:text=The%20physiological%20nucleotide%20concentration%20ratio,is%20~10%E2%88%925).
            'equilibrium_atp_adp_ratio': _DefaultFloatLogSlider(
                value=1e-5, min=-7, max=-3, readout_format='.1e',
                description="([ATP]/[ADP])|eq.:"),
            'K_d_atp': _DefaultFloatLogSlider(
                value=0.1, description="K_d^ATP:"),
            'K_d_adp': _DefaultFloatLogSlider(description="K_d^ADP:"),
            'k_DT': _DefaultFloatLogSlider(description="k_DT:"),
            'k_h': _DefaultFloatLogSlider(description="k_h:"),
            'k_s': _DefaultFloatLogSlider(value=0.1, description="k_s:"),
            'k_up': _DefaultFloatLogSlider(description="k_↑:"),
            'n_protomers': _DefaultIntSlider(description="n_protomers:"),
            'k_extended_to_flat_up': _DefaultFloatLogSlider(description="k_⮫:"),
            'k_flat_to_extended_down': _DefaultFloatLogSlider(description="k_⮯:"),
            'k_flat_to_extended_up': _DefaultFloatLogSlider(description="k_⮭:"),
        }
    
    def _construct_constrained_parameters(self) -> dict[str, Widget]:
        return {
            #'k_TD': HTML(description="k_TD:"),
            'k_down': HTML(description="k_↓:"),
            'k_h_bar': HTML(description="ꝁ_h:"),
            'k_flat_to_extended_down_bar': HTML(description="ꝁ_⮯:"),
            'k_extended_to_flat_down': HTML(description="k_⮩:"),
        }
    
    def _construct_gui(self) -> Widget:
        gui_plot = Output()
        gui_parameters = VBox([
            HTML(value="<h1>Velocity vs [ATP]/[ADP]</h1>"),

            HTML(value="<b>General Physical Parameters</b>"),
            HBox([self._free_parameters['equilibrium_atp_adp_ratio'], 
                HTML(value="Equilibrium ATP/ADP concentration ratio")]),
            HBox([self._free_parameters['K_d_atp'],
                HTML(value="Protomer-ATP dissociation constant")]),
            HBox([self._free_parameters['K_d_adp'],
                HTML(value="Protomer-ADP dissociation constant")]),
            HBox([self._free_parameters['k_DT'],
                HTML(value="Effective ADP->ATP exchange rate")]),
            #HBox([self._constrained_parameters['k_TD'],
            #    HTML(value="Effective ATP->ADP exchange rate "\
            #        "(constrained by Protomer-ATP/ADP exchange model)")]),
            HBox([self._free_parameters['k_h'],
                HTML(value="ATP Hydrolysis rate")]),
            HBox([self._free_parameters['k_s'],
                HTML(value="ATP Synthesis rate")]),

            HTML(value="<b>SC2R Model Physical Parameters</b>"),
            HBox([self._free_parameters['k_up'],
                HTML(value="Translocation up rate")]),
            HBox([self._constrained_parameters['k_down'],
                HTML(value="Translocation down rate "\
                    "(constrained by detailed balance)")]),

            HTML(value="<b>Disc-Spiral Model Physical Parameters</b>"),
            HBox([self._free_parameters['n_protomers'],
                HTML(value="Number of protomers")]),
            HBox([self._constrained_parameters['k_h_bar'],
                HTML(value="Effective ATP hydrolysis rate")]),
            HBox([self._free_parameters['k_extended_to_flat_up'],
                HTML(value="Spiral->disc up translocation rate")]),
            HBox([self._free_parameters['k_flat_to_extended_down'],
                HTML(value="Disc->spiral down translocation rate")]),
            HBox([self._constrained_parameters['k_flat_to_extended_down_bar'],
                HTML(value="Effective disc->spiral down translocation rate")]),
            HBox([self._free_parameters['k_flat_to_extended_up'],
                HTML(value="Disc->spiral up translocation rate")]),
            HBox([self._constrained_parameters['k_extended_to_flat_down'],
                HTML(value="Spiral->disc down rate "\
                        "(constrained by detailed balance)")]),
        ])

        gui = HBox([gui_plot, gui_parameters], 
                   layout=Layout(align_items='center'))
        
        return gui
    
    def _run(self) -> None:
        models = [self._sc2r, self._disc_spiral]
        self._update_models_free_parameters(models, self._free_parameters)
        self._update_gui_constrained_parameters(models, 
                                                self._constrained_parameters)
        
        atp_adp_ratios = (np.logspace(-1, 3, 100) 
                          * models[0].equilibrium_atp_adp_ratio) # Equal for both models
        velocities = {model: [] for model in models}
        for atp_adp_ratio in atp_adp_ratios:
            for model in models:
                model.atp_adp_ratio = atp_adp_ratio
                velocities[model].append(model.average_velocity())

        gui_plot = self._gui.children[0]
        with gui_plot:
            gui_plot.clear_output(wait=True)
            plt.close('VelocityVSATPADPRatio')
            fig = plt.figure('VelocityVSATPADPRatio')
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_visible = False
            ax = fig.add_subplot(111)

            # Plot velocities vs [ATP]/[ADP]
            ax.plot(atp_adp_ratios / self._sc2r.equilibrium_atp_adp_ratio,
                    velocities[self._sc2r],
                    label='SC/2R',
                    color='#DDAA33')
            ax.plot(atp_adp_ratios / self._disc_spiral.equilibrium_atp_adp_ratio,
                    velocities[self._disc_spiral],
                    label='Disc-Spiral',
                    color='#004488')
            ax.set_xscale('log')

            # Plot grey bars to highlight that <v> = 0 when 
            # [ATP]/[ADP] = ([ATP]/[ADP])|eq.
            # (log_x1, y0) is the point where the grey bars intersect if the 
            # axes goes from 0 (far left/bottom) to 1 (far right/top).
            # Calculations need to be done after the plotting is done since it
            # depends on the limits of the axes, which are automatically 
            # determined by matplotlib.
            log_xmin, log_xmax = np.log10(ax.get_xlim())
            ymin, ymax = ax.get_ylim()
            # Calculations need to be done after the log scale is applied
            log_x1 = (np.log10(1) - log_xmin) / (log_xmax - log_xmin)
            y0 = -ymin / (ymax - ymin)
            ax.axhline(0, xmin=log_xmin, xmax=log_x1, color='#BBBBBB', linestyle='--', zorder=0)
            ax.axvline(1, ymin=ymin, ymax=y0, color='#BBBBBB', linestyle='--', zorder=0)
            
            ax.set_xlabel("([ATP]/[ADP])/([ATP]/[ADP])|eq.")
            ax.set_ylabel("<v>[Residue ∙ k]")
            ax.legend()
            plt.show() # TODO Indicate that THIS IS IMPORTANT otherwise plot is 
            # not in the gui, but below and is not updated but instead a new 
            # plot is displayed everytime a value changes


class _DefaultFloatLogSlider(FloatLogSlider):
    """FloatLogSlider with default values."""
    def __init__(self, 
                value: float = 1.0,
                min: float = -2,
                max: float = 2,
                readout_format: str = '.2f',
                continuous_update: bool = False,
                *args, 
                **kwargs):
        super().__init__(
            value=value, min=min, max=max, readout_format=readout_format, 
            continuous_update=continuous_update, *args, **kwargs)
        

class _DefaultIntSlider(IntSlider):
    """IntSlider with default values."""
    def __init__(self, 
                value: int = 6,
                min: int = 1,
                max: int = 10,
                readout_format: str = 'd',
                continuous_update: bool = False,
                *args, 
                **kwargs):
        super().__init__(
            value=value, min=min, max=max, readout_format=readout_format, 
            continuous_update=continuous_update, *args, **kwargs)