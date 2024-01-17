from translocation_model import TranslocationModel, \
    SC2R, SC2R2Loops, DefectiveSC2R, \
    DiscSpiral, DefectiveDiscSpiral

from ipywidgets import Widget, FloatLogSlider, IntSlider, \
    HBox, VBox, HTML, Output, Layout
from IPython.display import display

import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

# TODO Add docstrings, and give example of how to extend library
#   In particular explain what should appear in _run
class Experiment(ABC):
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


class VelocityVSATPADPRatio(Experiment):
    def __init__(self):
        self._models = [SC2R(), DiscSpiral()]
        super().__init__()

    def _construct_free_parameters(self) -> dict[str, Widget]:
        return {
            'equilibrium_atp_adp_ratio': DefaultFloatLogSlider(
                value=0.01, description="([ATP]/[ADP])|eq.:"),
            'K_d_atp': DefaultFloatLogSlider(value=0.1, description="K_d^ATP:"),
            'K_d_adp': DefaultFloatLogSlider(description="K_d^ADP:"),
            'k_DT': DefaultFloatLogSlider(description="k_DT:"),
            'k_h': DefaultFloatLogSlider(description="k_h:"),
            'k_s': DefaultFloatLogSlider(value=0.1, description="k_s:"),
            'k_up': DefaultFloatLogSlider(description="k_↑:"),
            'n_protomers': DefaultIntSlider(description="n_protomers:"),
            'k_extended_to_flat_up': DefaultFloatLogSlider(description="k_⮫:"),
            'k_flat_to_extended_down': DefaultFloatLogSlider(description="k_⮯:"),
            'k_flat_to_extended_up': DefaultFloatLogSlider(description="k_⮭:"),
        }
    
    def _construct_constrained_parameters(self) -> dict[str, Widget]:
        return {
            'k_TD': HTML(description="k_TD:"),
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
            HBox([self._constrained_parameters['k_TD'],
                HTML(value="Effective ATP->ADP exchange rate "\
                    "(constrained by Protomer-ATP/ADP exchange model)")]),
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
        self._update_models_free_parameters(self._models, self._free_parameters)
        self._update_gui_constrained_parameters(self._models, 
                                                self._constrained_parameters)
        
        atp_adp_ratios = (np.logspace(-2, 2, 100) 
                          * self._models[0].equilibrium_atp_adp_ratio)
        velocities = {model: [] for model in self._models}
        for atp_adp_ratio in atp_adp_ratios:
            for model in self._models:
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

            for model in self._models:
                ax.plot(atp_adp_ratios, velocities[model], 
                        label=model.__class__.__name__)
            ax.set_xscale('log')
            ax.set_xlabel("[ATP]/[ADP]")
            ax.set_ylabel("Average velocity [Residue * k]")
            ax.legend()
            plt.show()


class DefaultFloatLogSlider(FloatLogSlider):
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
        

class DefaultIntSlider(IntSlider):
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