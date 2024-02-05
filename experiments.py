from translocation_models import TranslocationModel, \
    SC2R, DefectiveSC2R, NonIdealSC2R, \
    DiscSpiral, DefectiveDiscSpiral, NonIdealDiscSpiral

from ipywidgets import \
    FloatLogSlider, FloatRangeSlider, IntSlider, IntRangeSlider, Dropdown, \
    Widget, HBox, VBox, HTML, Output, Layout
from IPython.display import display

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from abc import ABC, abstractmethod
import copy


# TODO Render labels and legends with LaTeX
# TODO faire un map_to comme matplotlib legend pour maps widget to translcation model parameter (for ex. dict {widget: Class.attribute} ou pas, réfléchis moins de 1min)
# TODO change self._ to self. everywhere, but think about it a bit before doing so
class Experiment(ABC):
    """Base class for experiments.

    An experiment contains typically a GUI and one or more plots. The GUI 
    contains widgets that can be used to change the parameters of the model.
    Some of these parameters can be freely determined by the users, some others
    are constraints due to physical laws (e.g. detailed balance).
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

# TODO understand why at the end of a trajectory, the last step is not visible
# and do the same for potential experiment
class SC2RVsDiscSpiral(Experiment):
    """SC/2R vs Disc-Spiral comparison.
    
    Plot trajectories, average position and std (analytical and/or emprirical)
    for both models.
    """

    def __init__(self):
        self._sc2r = SC2R()
        self._disc_spiral = DiscSpiral()
        super().__init__()

    def _construct_free_parameters(self) -> dict[str, Widget]:
        return {
            'max_time': _DefaultFloatLogSlider(
                value=100, min=1, max=3, readout_format='.2f',
                description="t_max:"),
            'n_trajectories': _DefaultIntSlider(
                value=1, min=1, max=10, description="n_trajectories:"),
            'atp_adp_ratio': _DefaultFloatLogSlider(
                value=1e2, min=0, max=4, readout_format='.1e',
                description="[ATP]/[ADP]:"),
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
            'k_TD': HTML(description="k_TD:"),
            'k_down': HTML(description="k_↓:"),
            'k_h_bar': HTML(description="ꝁ_h:"),
            'k_flat_to_extended_down_bar': HTML(description="ꝁ_⮯:"),
            'k_extended_to_flat_down': HTML(description="k_⮩:"),
        }

    def _construct_gui(self) -> Widget:
        self.gui_plot = Output()
        self.gui_parameters = VBox([
            HBox([
                VBox([
                    HTML(value="<b>Simulation Parameter</b>"),
                    HBox([self._free_parameters['max_time'],
                        HTML(value="Maximum simulation time")]),
                    HBox([self._free_parameters['n_trajectories'],
                            HTML(value="Number of trajectory samples")]),

                    HTML(value="<b>General Physical Parameters</b>"),
                    HBox([self._free_parameters['atp_adp_ratio'],
                        HTML(value="ATP/ADP concentration ratio")]),
                    HBox([self._free_parameters['equilibrium_atp_adp_ratio'],
                        HTML(value="Equilibrium ATP/ADP concentration ratio")]),
                    HBox([self._free_parameters['K_d_atp'],
                        HTML(value="Protomer-ATP dissociation constant")]),
                    HBox([self._free_parameters['K_d_adp'],
                        HTML(value="Protomer-ADP dissociation constant")]),
                    HBox([self._free_parameters['k_DT'],
                        HTML(value="Effective ADP->ATP exchange rate")]),
                    HBox([self._constrained_parameters['k_TD'],
                        HTML(value="Effective ATP->ADP exchange rate "
                            "(constrained by Protomer-ATP/ADP exchange model)")]),
                    HBox([self._free_parameters['k_h'],
                        HTML(value="ATP Hydrolysis rate")]),
                    HBox([self._free_parameters['k_s'],
                        HTML(value="ATP Synthesis rate")]),
                ]),
                VBox([
                    HTML(value="<b>SC2R Model Physical Parameters</b>"),
                    HBox([self._free_parameters['k_up'],
                        HTML(value="Translocation up rate")]),
                    HBox([self._constrained_parameters['k_down'],
                        HTML(value="Translocation down rate "
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
                        HTML(value="Spiral->disc down rate "
                            "(constrained by detailed balance)")]),
                ]),
            ]),
        ])

        gui = VBox([HTML(value="<h1>SC/2R and Disc-Spiral comparison</h1>"), 
                    self.gui_plot, self.gui_parameters],
            layout=Layout(align_items='center'))

        return gui

    def _run(self) -> None:
        # Update GUI<->Models
        models = [self._sc2r, self._disc_spiral]
        self._update_models_free_parameters(models, self._free_parameters)
        self._update_gui_constrained_parameters(models,
                                                self._constrained_parameters)

        # Normalize average velocity
        for model in models:
            model.normalize_average_velocity(inplace=True)
        assert np.isclose(self._sc2r.average_velocity(),
                          self._disc_spiral.average_velocity())

        # For both model, we do a few trajectories, compute analytical stats and
        # empirical stats, and then plot everything.
        plot_trajectories = True
        plot_analytical_stats = True
        plot_empirical_stats = True
        times = np.linspace(0, self._free_parameters['max_time'].value, 100)
        n_trajectories = self._free_parameters['n_trajectories'].value

        if plot_trajectories:
            trajectories = {model: None for model in models}
            for model in models:
                out = model.gillespie(
                    max_time=times[-1],
                    n_simulations=n_trajectories,
                    cumulative_sums='position')
                if isinstance(out, pd.DataFrame):
                    out = [out]
                trajectories[model] = out

        if plot_analytical_stats:
            analytical_position_stats = {model: None for model in models}
            for model in models:
                analytical_position_stats[model] = \
                    model.analytical_attribute_stats(
                        'position', times=times)

        if plot_empirical_stats:
            empirical_position_stats = {model: None for model in models}
            position_sojourn_times = {model: None for model in models}
            n_simulations = 100
            for model in models:
                empirical_position_stats[model], position_sojourn_times[model] = \
                    model.empirical_attribute_stats(
                        'position', times=times, n_simulations=n_simulations)

        # Plot everything
        with self.gui_plot:
            self.gui_plot.clear_output(wait=True)
            plt.close('SC2RVsDiscSpiral')
            fig = plt.figure('SC2RVsDiscSpiral', [13, 4.8])
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_visible = False
            ax_traj = fig.add_subplot(121)
            ax_hist = fig.add_subplot(122)

            yellow = '#DDAA33'
            blue = '#004488'
            alpha_0_2 = '33'
            alpha_0_5 = '80'
            hidden = '#00000000'
            colors = [yellow, blue]
            model_names = ['SC/2R', 'Disc-Spiral']
            for model, color, model_name in zip(models, colors, model_names):
                if plot_analytical_stats:
                    ax_traj.fill_between(
                    analytical_position_stats[model]['time'],
                    analytical_position_stats[model]['mean'] -
                    analytical_position_stats[model]['std'],
                    analytical_position_stats[model]['mean'] +
                    analytical_position_stats[model]['std'],
                    facecolor=color+alpha_0_2, edgecolor=hidden)

                if plot_empirical_stats:
                    ax_traj.plot(
                        empirical_position_stats[model]['time'],
                        (empirical_position_stats[model]['mean']
                            - empirical_position_stats[model]['std']),
                        color=color, linestyle='--', alpha=0.5)
                    ax_traj.plot(
                        empirical_position_stats[model]['time'],
                        (empirical_position_stats[model]['mean']
                            + empirical_position_stats[model]['std']),
                        color=color, linestyle='--', alpha=0.5)
                    
                    hist_label = (model_name 
                                  + r' $\langle τ \rangle='
                                  + str(round(np.mean(position_sojourn_times[model]), 2))
                                  + '$')
                    ax_hist.hist(position_sojourn_times[model], 
                                 bins=10, density=True, 
                                 edgecolor=color, facecolor=color + alpha_0_5, 
                                 label=hist_label)
                    
                if plot_trajectories:
                    linewidth = 2
                    for trajectory in trajectories[model]:
                        ax_traj.step(trajectory['time'], trajectory['position'],
                                where='post', color=color, alpha=1,
                                linewidth=linewidth)
            # Average velocity
            ax_traj.plot(
                [times[0], times[-1]],
                [model.average_velocity() * times[0],
                 model.average_velocity() * times[-1]],
                color='#BBBBBB', zorder=0, alpha=0.5)

            ax_traj.set_xlabel('Time [a.u.]')
            ax_traj.set_ylabel('Position [#Residue]')

            # Very ugly code for custom legend. We use Handlerclasses defined
            # below. This is definitely ugly but it works, and I don't have time
            # to do it better right now. The width of the legend is set in 
            # ModelsHandler class definition below.
            ax_traj.legend(
                [self.Models(), self.Sigmas(), self.Trajectories(), self.ATP()], 
                ['', '', '', ''],
                handler_map={
                    self.Models: self.ModelsHandler(),
                    self.Sigmas: self.SigmasHandler(),
                    self.Trajectories: self.TrajectoriesHandler(),
                    self.ATP: self.ATPHandler(models)}
            )
            ax_hist.set_xlabel('Time [a.u.]')
            ax_hist.set_ylabel('Density')
            ax_hist.set_title('Position sojourn time histogram')
            ax_hist.legend()



            plt.show()

    class Models():
        """Legend 1st row."""
        pass

    class Sigmas():
        """Legend 2nd row."""
        pass

    class Trajectories():
        """Legend 3rd row."""
        pass

    class ATP():
        """Legend 4th row."""
        pass

    class ModelsHandler:
        """Legend 1st row handler."""

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height

            x0 = width
            sc2r_circle = mpl.patches.Circle(
                (x0 + width/2, height/2), 0.6*fontsize, facecolor='#DDAA3380',
                edgecolor='#DDAA33', transform=handlebox.get_transform())
            sc2r_text = mpl.text.Text(x=x0 + width, y=0, text='SC/2R')

            disc_spiral_circle = mpl.patches.Circle(
                (x0 + width + 5*fontsize, height/2), 0.6*fontsize, 
                facecolor='#00448880', edgecolor='#004488', 
                transform=handlebox.get_transform())
            disc_spiral_text = mpl.text.Text(x=x0 + 1.5*width + 5*fontsize, y=0,
                                             text='Disc-Spiral')

            handlebox.width *= 8.7  # Width of full legend handled by this parameter, ugly but it works
            handlebox.add_artist(sc2r_circle)
            handlebox.add_artist(sc2r_text)
            handlebox.add_artist(disc_spiral_circle)
            handlebox.add_artist(disc_spiral_text)

    class SigmasHandler:
        """Legend 2nd row handler."""

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height

            # Analytical
            upper_triangle_xy = np.array(
                [[0, 0], [0, height], [width, height]])
            upper_triangle = mpl.patches.Polygon(
                upper_triangle_xy, closed=True, facecolor='#DDAA3380',
                edgecolor='#00000000', transform=handlebox.get_transform())
            lower_triangle_xy = np.array([[0, 0], [width, height], [width, 0]])
            lower_triangle = mpl.patches.Polygon(
                lower_triangle_xy, closed=True, facecolor='#00448880',
                edgecolor='#00000000', transform=handlebox.get_transform())
            analytical_line = mpl.lines.Line2D(
                [0, width], [0, height], color='#BBBBBB')
            analytical_text = mpl.text.Text(x=width + 0.5*fontsize, y=0,
                                            text='(Ana.);')

            # Empirical
            x0 = width + 4.5*fontsize
            upper_empirical_line = mpl.lines.Line2D(  # SC2R std
                [x0, x0 + width], [height, height], linestyle='--', color='#DDAA33', alpha=0.5)
            middle_empirical_line = mpl.lines.Line2D(  # Average velocity
                [x0, x0 + width], [height/2, height/2], color='#BBBBBB')
            lower_empirical_line = mpl.lines.Line2D(  # Disc-Spiral std
                [x0, x0 + width], [0, 0], linestyle='--', color='#00448880', alpha=0.5)
            empirical_text = mpl.text.Text(x=2*width + 5*fontsize, y=0,
                                           text='(Emp.):')

            text = mpl.text.Text(x=2*width + 9*fontsize, y=0, text='❬Pos.❭±σ')

            handlebox.add_artist(upper_triangle)
            handlebox.add_artist(lower_triangle)
            handlebox.add_artist(analytical_line)
            handlebox.add_artist(analytical_text)
            handlebox.add_artist(upper_empirical_line)
            handlebox.add_artist(middle_empirical_line)
            handlebox.add_artist(lower_empirical_line)
            handlebox.add_artist(empirical_text)
            handlebox.add_artist(text)

    class TrajectoriesHandler:
        """Legend 3rd row handler."""

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height

            sc2r_1 = mpl.lines.Line2D(
                [0, width/3], [height/3, height/3], color='#DDAA33')
            sc2r_2 = mpl.lines.Line2D(
                [width/3, width/3], [height/3, height], color='#DDAA33')
            sc2r_3 = mpl.lines.Line2D(
                [width/3, width], [height, height], color='#DDAA33')

            disc_spiral_1 = mpl.lines.Line2D(
                [0, 2*width/3], [0, 0], color='#004488')
            disc_spiral_2 = mpl.lines.Line2D(
                [2*width/3, 2*width/3], [0, 2*height/3], color='#004488')
            disc_spiral_3 = mpl.lines.Line2D(
                [2*width/3, width], [2*height/3, 2*height/3], color='#004488')

            text = mpl.text.Text(x=1.5*width + 0*fontsize,
                                 y=0, text='Some trajectory samples')

            handlebox.add_artist(sc2r_1)
            handlebox.add_artist(sc2r_2)
            handlebox.add_artist(sc2r_3)
            handlebox.add_artist(disc_spiral_1)
            handlebox.add_artist(disc_spiral_2)
            handlebox.add_artist(disc_spiral_3)
            handlebox.add_artist(text)

    class ATPHandler:
        """Legend 4th row handler."""

        def __init__(self, models: list[TranslocationModel, TranslocationModel]):
            self._models = models # [SC2R, Disc-Spiral]

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height

            r_atp = mpl.text.Text(x=0, y=0, text=r'$\frac{\text{#ATP}}{\text{Residue}}=$')

            x0 += 4*fontsize
            defectless_circle = mpl.patches.Circle(
                (x0 + width/2, height/2), 0.6*fontsize, facecolor='#DDAA3380',
                edgecolor='#DDAA33', transform=handlebox.get_transform())
            defectless_text = mpl.text.Text(
                x=x0 + width, y=0, 
                text=str(round(self._models[0].atp_consumption_rate()
                               / self._models[0].average_velocity(), 2)) + ";")
            
            x0 += 5*fontsize
            defective_circle = mpl.patches.Circle(
                (x0 + width/2, height/2), 0.6*fontsize, facecolor='#00448880',
                edgecolor='#004488', transform=handlebox.get_transform())
            defective_text = mpl.text.Text(
                x=x0 + width, y=0,
                text=str(round(self._models[1].atp_consumption_rate()
                               / self._models[1].average_velocity(), 2)))
            
            handlebox.add_artist(r_atp)
            handlebox.add_artist(defectless_circle)
            handlebox.add_artist(defectless_text)
            handlebox.add_artist(defective_circle)
            handlebox.add_artist(defective_text)


class VelocityVsATPADPRatio(Experiment):
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
            'ratio_magnitude_range': IntRangeSlider(
                value=[1, 6], min=-2, max=6, continuous_update=False,
                description="O(([ATP]/[ADP])/([ATP]/[ADP])|eq.):"),
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
            HBox([self._free_parameters['ratio_magnitude_range'],
                  HTML(value="(ATP/ADP)/([ATP]/[ADP])|eq. orders of magnitude")]),
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
                    "(constrained by Protomer-ATP/ADP exchange model, " \
                    "for current [ATP]/[ADP] range)")]),
            HBox([self._free_parameters['k_h'],
                  HTML(value="ATP Hydrolysis rate")]),
            HBox([self._free_parameters['k_s'],
                  HTML(value="ATP Synthesis rate")]),

            HTML(value="<b>SC/2R Model Physical Parameters</b>"),
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
        # Update k_TD for current range of ATP/ADP ratios
        def k_TD_range(k_DT, K_d_atp, K_d_adp, atp_adp_ratio_min, 
                       atp_adp_ratio_max):
            """k_TD widget, value for min/max [ATP]/[ADP] range boundaries."""
            k_TD_min = k_DT * K_d_atp / K_d_adp / atp_adp_ratio_max
            k_TD_max = k_DT * K_d_atp / K_d_adp / atp_adp_ratio_min
            return str(round(k_TD_max, 2)) + "●――●" + str(round(k_TD_min, 2))
        self._constrained_parameters['k_TD'].value = k_TD_range(
            self._free_parameters['k_DT'].value,
            self._free_parameters['K_d_atp'].value,
            self._free_parameters['K_d_adp'].value,
            (10**self._free_parameters['ratio_magnitude_range'].value[0]
             * self._free_parameters['equilibrium_atp_adp_ratio'].value),
            (10**self._free_parameters['ratio_magnitude_range'].value[1]
             * self._free_parameters['equilibrium_atp_adp_ratio'].value),
        )

        # Velocity for all (ATP/ADP)/(ATP/ADP)|eq. values in range
        min, max = self._free_parameters['ratio_magnitude_range'].value
        atp_adp_ratios = (np.logspace(min, max, 100)
                          * self._free_parameters['equilibrium_atp_adp_ratio'].value)
        velocities = {model: [] for model in models}
        for atp_adp_ratio in atp_adp_ratios:
            for model in models:
                model.atp_adp_ratio = atp_adp_ratio
                velocities[model].append(model.average_velocity())

        gui_plot = self._gui.children[0]
        with gui_plot:
            gui_plot.clear_output(wait=True)
            plt.close('VelocityVsATPADPRatio')
            fig = plt.figure('VelocityVsATPADPRatio')
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
            # Coordinates where both plots cross each other, on x_log scale
            cross_x, cross_y = np.log10(1), 0 
            # We plot it only if it is visible on the plot
            if (log_xmin < cross_x < log_xmax and ymin < cross_y < ymax):
                # Calculations need to be done after the log scale is applied
                log_x1 = (cross_x - log_xmin) / (log_xmax - log_xmin)
                y0 = (cross_y - ymin) / (ymax - ymin)
                ax.axhline(0, xmin=log_xmin, xmax=log_x1,
                        color='#BBBBBB', linestyle='--', zorder=0)
                ax.axvline(1, ymin=ymin, ymax=y0, color='#BBBBBB',
                        linestyle='--', zorder=0, label="Cross (1, 0)")

            ax.set_xlabel("([ATP]/[ADP])/([ATP]/[ADP])|eq.")
            ax.set_ylabel("❬v❭ [Residue ∙ k]")
            ax.legend()
            plt.show()  # TODO Indicate that THIS IS IMPORTANT otherwise plot is
            # not in the gui, but below and is not updated but instead a new
            # plot is displayed everytime a value changes


class VelocityVsPotential(Experiment):
    # https://fr.wikipedia.org/wiki/%C3%89quation_d%27Eyring
    # https://fr.wikipedia.org/wiki/Loi_d%27Arrhenius
    def __init__(self):
        self._sc2r = SC2R()
        self._disc_spiral = DiscSpiral()
        # Copy used for accessing rates before applying the Boltzmann factor due to potential
        self._sc2r_copy = copy.deepcopy(self._sc2r)
        self._disc_spiral_copy = copy.deepcopy(self._disc_spiral)
        super().__init__()

    def _construct_free_parameters(self) -> dict[str, Widget]:
        return {
            # ΔU/T potential for a unit displacement (i.e. multiply with
            # displacement size to have true potential difference)
            'unit_potential': FloatRangeSlider(
                value=[-1, 1], min=-3, max=3, continuous_update=False,
                description="u/T:"),
            # Source for ATP/ADP ratio:
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6395684/#:~:text=The%20physiological%20nucleotide%20concentration%20ratio,is%20~10%E2%88%925).
            'atp_adp_ratio': _DefaultFloatLogSlider(
                value=100, min=-1, max=4, readout_format='.1e',
                description="[ATP]/[ADP]:"),
            'equilibrium_atp_adp_ratio': _DefaultFloatLogSlider(
                value=1e-5, min=-7, max=1, readout_format='.1e',
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
            'k_TD': HTML(description="k_TD:"),
            'k_down': HTML(description="k_↓:"),
            'k_h_bar': HTML(description="ꝁ_h:"),
            'k_flat_to_extended_down_bar': HTML(description="ꝁ_⮯:"),
            'k_extended_to_flat_down': HTML(description="k_⮩:"),
        }

    def _construct_gui(self) -> Widget:
        gui_plot = Output()
        gui_parameters = VBox([
            HTML(value="<h1>Velocity vs Potential</h1>"),

            HBox([self._free_parameters['unit_potential'],
                  HTML(value="Potential (for a unit displacement) over temperature")]),

            HTML(value="<b>General Physical Parameters</b>"),
            HBox([self._free_parameters['atp_adp_ratio'],
                  HTML(value="ATP/ADP concentration ratio")]),
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
        models = [self._sc2r, self._disc_spiral]
        # Update models<->GUI
        self._update_models_free_parameters(models, self._free_parameters)
        self._update_gui_constrained_parameters(models,
                                                self._constrained_parameters)
        
        # Add potentials in range
        models_copy = [self._sc2r_copy, self._disc_spiral_copy]
        min, max = self._free_parameters['unit_potential'].value
        unit_potentials = np.linspace(min, max, 100)
        velocities = {model: [] for model in models}
        for model, model_copy in zip(models, models_copy):
            for unit_potential in unit_potentials:
                for u, v, attributes in model.kinetic_scheme.edges(data=True):
                    # For each displacement edge, we multiply the rate with the 
                    # potential Boltzmann factor and the displacement size
                    if 'position' in attributes:
                        old_rate = model_copy.kinetic_scheme.edges[u, v]['rate']
                        displacement = attributes['position']
                        def new_rate(old_rate=old_rate, 
                                     displacement=displacement, 
                                     unit_potential=unit_potential):
                            return old_rate() * np.exp(-displacement * unit_potential)
                        attributes['rate'] = new_rate
                    # Note that this does not affect the value of the rates
                    # constrained by the thermodynamic loop law, since they are
                    # not stored in the kinetic_scheme graph, but are defined
                    # in the model class. The only thing we modify is what 
                    # function will be called when taking this edge thanks to
                    # the way the rates in the kinectic scheme are defined (as
                    # functions rather than values).
                velocities[model].append(model.average_velocity())

        gui_plot = self._gui.children[0]
        with gui_plot:
            gui_plot.clear_output(wait=True)
            plt.close('VelocityVsPotential')
            fig = plt.figure('VelocityVsPotential')
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_visible = False
            ax = fig.add_subplot(111)

            # Plot velocities vs [ATP]/[ADP]
            sc2r_plot = ax.plot(unit_potentials,
                    velocities[self._sc2r],
                    label="SC/2R (Δx = 2 a.a.)",
                    color='#DDAA33')
            step_size = (self._disc_spiral.n_protomers - 1) * 2
            disc_spiral_plot = ax.plot(unit_potentials,
                    velocities[self._disc_spiral],
                    label="Disc-Spiral (Δx = " + str(step_size) + " a.a.)",
                    color='#004488')

            ax.set_xlabel("u/T")
            ax.set_ylabel("❬v❭ [Residue ∙ k]")
            ax.legend()
            plt.show()


# TODO add checkboxes for plotting trajectories, analytical and empirical
class DefectlessVsDefective(Experiment):
    """Defectless vs defective comparison.

    Plot trajectories, average position and std (analytical and/or emprirical)
    for all models (SC/2R and Disc-Spiral, and defectless and defective).
    """

    def __init__(self):
        self._sc2r = SC2R()
        self._defective_sc2r = DefectiveSC2R()
        self._disc_spiral = DiscSpiral()
        self._defective_disc_spiral = DefectiveDiscSpiral()
        super().__init__()

    def _construct_free_parameters(self) -> dict[str, Widget]:
        return {
            'defect_factor': _DefaultFloatLogSlider(
                value=0.1, min=-3, max=0, readout_format='.3f',
                description="defect_factor:"),
            'n_protomers': _DefaultIntSlider(description="n_protomers:"),
            'max_time': _DefaultFloatLogSlider(
                value=100, min=1, max=3, readout_format='.2f',
                description="t_max:"),
            'n_trajectories': _DefaultIntSlider(
                value=1, min=1, max=10, description="n_trajectories:"),
            'atp_adp_ratio': _DefaultFloatLogSlider(
                value=1e2, min=0, max=4, readout_format='.1e',
                description="[ATP]/[ADP]:"),
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
            'k_extended_to_flat_up': _DefaultFloatLogSlider(description="k_⮫:"),
            'k_flat_to_extended_down': _DefaultFloatLogSlider(description="k_⮯:"),
            'k_flat_to_extended_up': _DefaultFloatLogSlider(description="k_⮭:"),
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
        gui_parameters = HBox([
            VBox([
                HTML(
                    value="<h1>SC/2R and Disc-Spiral ATP consumption rate comparison</h1>"),

                HBox([self._free_parameters['defect_factor'],
                      HTML(value="Hydrolysis defect factor")]),
                HBox([self._free_parameters['n_protomers'],
                      HTML(value="Number of protomers")]),

                HTML(value="<b>Simulation Parameter</b>"),
                HBox([self._free_parameters['max_time'],
                      HTML(value="Maximum simulation time")]),
                HBox([self._free_parameters['n_trajectories'],
                      HTML(value="Number of trajectory samples")]),

                HTML(value="<b>General Physical Parameters</b>"),
                HBox([self._free_parameters['atp_adp_ratio'],
                      HTML(value="ATP/ADP concentration ratio")]),
                HBox([self._free_parameters['equilibrium_atp_adp_ratio'],
                      HTML(value="Equilibrium ATP/ADP concentration ratio")]),
                HBox([self._free_parameters['K_d_atp'],
                      HTML(value="Protomer-ATP dissociation constant")]),
                HBox([self._free_parameters['K_d_adp'],
                      HTML(value="Protomer-ADP dissociation constant")]),
                HBox([self._free_parameters['k_DT'],
                      HTML(value="Effective ADP->ATP exchange rate")]),
                HBox([self._constrained_parameters['k_TD'],
                      HTML(value="Effective ATP->ADP exchange rate "
                           "(constrained by Protomer-ATP/ADP exchange model)")]),
                HBox([self._free_parameters['k_h'],
                      HTML(value="ATP Hydrolysis rate")]),
                HBox([self._free_parameters['k_s'],
                      HTML(value="ATP Synthesis rate")]),
            ]),
            VBox([
                HTML(value="<b>SC2R Model Physical Parameters</b>"),
                HBox([self._free_parameters['k_up'],
                      HTML(value="Translocation up rate")]),
                HBox([self._constrained_parameters['k_down'],
                      HTML(value="Translocation down rate "
                           "(constrained by detailed balance)")]),

                HTML(value="<b>Disc-Spiral Model Physical Parameters</b>"),
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
                      HTML(value="Spiral->disc down rate "
                           "(constrained by detailed balance)")]),
            ])
        ])

        gui = VBox([gui_plot, gui_parameters],
                   layout=Layout(align_items='center'))

        return gui

    def _run(self) -> None:
        # Update GUI<->Models
        models = [self._sc2r, self._defective_sc2r,
                  self._disc_spiral, self._defective_disc_spiral]
        self._update_models_free_parameters(models, self._free_parameters)
        self._update_gui_constrained_parameters(models,
                                                self._constrained_parameters)

        # For each model, we do a few trajectories, compute analytical stats and
        # empirical stats, and then plot everything.
        plot_trajectories = True
        plot_analytical_stats = True
        plot_empirical_stats = True
        times = np.linspace(0, self._free_parameters['max_time'].value, 100)
        n_trajectories = self._free_parameters['n_trajectories'].value

        if plot_trajectories:
            trajectories = {model: None for model in models}
            for model in models:
                out = model.gillespie(
                    max_time=times[-1],
                    n_simulations=n_trajectories,
                    cumulative_sums='position')
                if isinstance(out, pd.DataFrame):
                    out = [out]
                trajectories[model] = out

        if plot_analytical_stats:
            analytical_position_stats = {model: None for model in models}
            for model in models:
                analytical_position_stats[model] = \
                    model.analytical_attribute_stats(
                        'position', times=times)

        if plot_empirical_stats:
            empirical_position_stats = {model: None for model in models}
            position_sojourn_times = {model: None for model in models}
            n_simulations = 100
            for model in models:
                empirical_position_stats[model], position_sojourn_times[model] = \
                    model.empirical_attribute_stats(
                        'position', times=times, n_simulations=n_simulations)

        # Plot everything
        gui_plot = self._gui.children[0]
        with gui_plot:
            gui_plot.clear_output(wait=True)
            plt.close('DefectlessVsDefective')
            fig = plt.figure('DefectlessVsDefective', figsize=(13, 10))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_visible = False
            ax_sc2r_traj = fig.add_subplot(221)
            ax_disc_spiral_traj = fig.add_subplot(222)
            ax_sc2r_hist = fig.add_subplot(223)
            ax_disc_spiral_hist = fig.add_subplot(224)

            yellow = '#DDAA33'
            blue = '#004488'
            red = '#BB5566'
            alpha_0_2 = '33'
            alpha_0_5 = '80'
            hidden = '#00000000'
            nested_models = [[self._sc2r, self._defective_sc2r],
                             [self._disc_spiral, self._defective_disc_spiral]]
            nested_colors = [[yellow, red], [blue, red]]
            nested_axes = [[ax_sc2r_traj, ax_sc2r_hist], 
                           [ax_disc_spiral_traj, ax_disc_spiral_hist]]
            names = ['SC/2R', 'Disc-Spiral']
            # First SC2R, then Disc-Spiral
            for models, colors, axes, name in zip(
                nested_models, nested_colors, nested_axes, names):
                ax_traj = axes[0]
                ax_hist = axes[1]
                # First Non-defective, then Defective
                for model, color in zip(models, colors):
                    if plot_analytical_stats:
                        ax_traj.fill_between(
                            analytical_position_stats[model]['time'],
                            analytical_position_stats[model]['mean'] -
                            analytical_position_stats[model]['std'],
                            analytical_position_stats[model]['mean'] +
                            analytical_position_stats[model]['std'],
                            facecolor=color+alpha_0_2, edgecolor=hidden)

                    if plot_empirical_stats:
                        ax_traj.plot(
                            empirical_position_stats[model]['time'],
                            (empirical_position_stats[model]['mean']
                             - empirical_position_stats[model]['std']),
                            color=color, linestyle='--', alpha=0.5)
                        ax_traj.plot(
                            empirical_position_stats[model]['time'],
                            (empirical_position_stats[model]['mean']
                             + empirical_position_stats[model]['std']),
                            color=color, linestyle='--', alpha=0.5)
                        
                        ax_hist.hist(
                            position_sojourn_times[model], bins=20, alpha=0.5,
                            facecolor=color + alpha_0_5, edgecolor=color,
                            label=name, density=True)

                    if plot_trajectories:
                        linewidth = 2
                        for trajectory in trajectories[model]:
                            ax_traj.step(trajectory['time'], trajectory['position'],
                                    where='post', color=color, alpha=1,
                                    linewidth=linewidth)
                    # Average velocity
                    ax_traj.plot(
                        [times[0], times[-1]],
                        [model.average_velocity() * times[0],
                         model.average_velocity() * times[-1]],
                        color=color, zorder=0, alpha=0.5)

                ax_traj.set_xlabel('Time [1/k]')
                ax_traj.set_ylabel('Position [#Residue]')

                # Very ugly code for custom legend. We use Handlerclasses defined
                # below. This is definitely ugly but it works, and I don't have time
                # to do it better right now. The width of the legend is set in
                # ModelsHandler class definition below.
                velocities = [model.average_velocity() for model in models]
                ax_traj.legend(
                    [self.Models(), self.Sigmas(), self.Trajectories(), self.Velocities()],
                    ['', '', '', ''],
                    handler_map={
                        self.Models: self.ModelsHandler(colors, velocities),
                        self.Sigmas: self.SigmasHandler(colors),
                        self.Trajectories: self.TrajectoriesHandler(colors),
                        self.Velocities: self.VelocitiesHandler(colors, velocities)},
                    title=name
                )

            plt.show()

    class Models():
        """Legend 1st row."""
        pass

    class Sigmas():
        """Legend 2nd row."""
        pass

    class Trajectories():
        """Legend 3rd row."""
        pass

    class Velocities():
        """Legend 4th row."""
        pass

    class ModelsHandler:
        """Legend 1st row handler."""

        def __init__(self, colors: list[str, str], velocities: list[float, float]):
            self._colors = colors  # [Defectless, Defective] colors
            # [Defectless, Defective] average velocities
            self._velocities = velocities

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height

            x0 = width
            defectless_circle = mpl.patches.Circle(
                (x0 + width/2, height/2), 0.6*fontsize,
                facecolor=self._colors[0] + '80', edgecolor=self._colors[0],
                transform=handlebox.get_transform())
            defectless_text = mpl.text.Text(x=x0 + width, y=0, text="Defectless")

            x0 += width + 6*fontsize
            defective_circle = mpl.patches.Circle(
                (x0 + width/2, height/2), 0.6*fontsize,
                facecolor=self._colors[1] + '80', edgecolor=self._colors[1],
                transform=handlebox.get_transform())
            deffective_text = mpl.text.Text(x=x0 + width, y=0, text='Defective')

            # Width of full legend handled by this parameter, ugly but it works
            handlebox.width *= 8.7
            handlebox.add_artist(defectless_circle)
            handlebox.add_artist(defectless_text)
            handlebox.add_artist(defective_circle)
            handlebox.add_artist(deffective_text)

    class SigmasHandler:
        """Legend 2nd row handler."""

        def __init__(self, colors: list[str, str]):
            self._colors = colors  # List[Defectless, Defective] colors

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height

            # Analytical
            defectless_triangle_xy = np.array(
                [[0, 0], [0, height], [width, height]])
            defectless_triangle = mpl.patches.Polygon(
                defectless_triangle_xy, closed=True,
                facecolor=self._colors[0]+'80', edgecolor='#00000000',
                transform=handlebox.get_transform())
            defective_triangle_xy = np.array(
                [[0, 0], [width, height], [width, 0]])
            defective_triangle = mpl.patches.Polygon(
                defective_triangle_xy, closed=True,
                facecolor=self._colors[1]+'80', edgecolor='#00000000',
                transform=handlebox.get_transform())
            defectless_analytical_line = mpl.lines.Line2D(
                [0, width/2], [0, height/2], color=self._colors[0], alpha=0.5)
            defective_analytical_line = mpl.lines.Line2D(
                [width/2, width], [height/2, height], color=self._colors[1],
                alpha=0.5)
            analytical_text = mpl.text.Text(x=width + 0.5*fontsize, y=0,
                                            text='(Ana.);')

            # Empirical
            x0 = width + 4.5*fontsize
            defectless_std_line = mpl.lines.Line2D(
                [x0, x0 + width], [height, height], linestyle='--',
                color=self._colors[0], alpha=0.5)
            defectless_mean_line = mpl.lines.Line2D(
                [x0, x0 + width], [height/2, height/2], color=self._colors[0],
                alpha=0.5)
            defective_std_line = mpl.lines.Line2D(
                [x0, x0 + width], [0, 0], linestyle='--',
                color=self._colors[1], alpha=0.5)
            defective_mean_line = mpl.lines.Line2D(
                [x0 + width/2, x0 + width], [height/2, height/2],
                color=self._colors[1], alpha=0.5)
            empirical_text = mpl.text.Text(x=2*width + 5*fontsize, y=0,
                                           text='(Emp.):')

            text = mpl.text.Text(x=2*width + 9*fontsize, y=0, text='❬Pos.❭±σ')

            handlebox.add_artist(defectless_triangle)
            handlebox.add_artist(defective_triangle)
            handlebox.add_artist(defectless_analytical_line)
            handlebox.add_artist(defective_analytical_line)
            handlebox.add_artist(analytical_text)
            handlebox.add_artist(defectless_std_line)
            handlebox.add_artist(defectless_mean_line)
            handlebox.add_artist(defective_std_line)
            handlebox.add_artist(defective_mean_line)
            handlebox.add_artist(empirical_text)
            handlebox.add_artist(text)

    class TrajectoriesHandler:
        """Legend 3rd row handler."""

        def __init__(self, colors: list[str, str]):
            self._colors = colors  # List[Defectless, Defective] colors

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height

            defectless_1 = mpl.lines.Line2D(
                [0, width/3], [height/3, height/3], color=self._colors[0])
            defectless_2 = mpl.lines.Line2D(
                [width/3, width/3], [height/3, height], color=self._colors[0])
            defectless_3 = mpl.lines.Line2D(
                [width/3, width], [height, height], color=self._colors[0])

            defective_1 = mpl.lines.Line2D(
                [0, 2*width/3], [0, 0], color=self._colors[1])
            defective_2 = mpl.lines.Line2D(
                [2*width/3, 2*width/3], [0, 2*height/3], color=self._colors[1])
            defective_3 = mpl.lines.Line2D(
                [2*width/3, width], [2*height/3, 2*height/3],
                color=self._colors[1])

            text = mpl.text.Text(x=1.5*width + 0*fontsize,
                                 y=0, text='Some trajectory samples')

            handlebox.add_artist(defectless_1)
            handlebox.add_artist(defectless_2)
            handlebox.add_artist(defectless_3)
            handlebox.add_artist(defective_1)
            handlebox.add_artist(defective_2)
            handlebox.add_artist(defective_3)
            handlebox.add_artist(text)

    class VelocitiesHandler:
        """Legend 4th row handler."""

        def __init__(self, colors: list[str, str], velocities: list[float, float]):
            self._colors = colors # List[Defectless, Defective] colors
            self._velocities = velocities # List[Defectless, Defective] velocities

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height

            v = mpl.text.Text(x=0, y=0, text='❬v❭ =')

            x0 += 4*fontsize
            defectless_line = mpl.lines.Line2D(
                [x0, x0 + 0.6*width], [height/2, height/2], color=self._colors[0],
                alpha=0.5)
            defectless_text = mpl.text.Text(
                x=x0 + 0.8*width, y=0,
                text=str(round(self._velocities[0], 2)) + ";")
            
            x0 += width + 4*fontsize
            defective_line = mpl.lines.Line2D(
                [x0, x0 + 0.6*width], [height/2, height/2], color=self._colors[1],
                alpha=0.5)
            defective_text = mpl.text.Text(
                x=x0 + 0.8*width, y=0,
                text=str(round(self._velocities[1], 2)) + ";")

            handlebox.add_artist(v)
            handlebox.add_artist(defectless_line)
            handlebox.add_artist(defectless_text)
            handlebox.add_artist(defective_line)
            handlebox.add_artist(defective_text)


# TODO show k_out for each state 
class NonIdeal(Experiment):
    def __init__(self):
        self._non_ideal_sc2r = NonIdealSC2R()
        self._non_ideal_disc_spiral = NonIdealDiscSpiral()
        super().__init__()

    def _construct_free_parameters(self) -> dict[str, Widget]:
        return {
            'k_out_range': IntRangeSlider(
                value=[-3, 2], min=-3, max=2, continuous_update=False,
                description="O(k_out):"),
            'k_in': _DefaultFloatLogSlider(description="k_in:"),
            'sc2r_reference_state': Dropdown(
                options=[state for state 
                         in self._non_ideal_sc2r.kinetic_scheme.nodes
                         if state != 'out'],
                value='TTT', continuous_update=False,
                description="SC/2R reference state:"),
            'disc_spiral_reference_state': Dropdown(
                options=[state for state 
                         in self._non_ideal_disc_spiral.kinetic_scheme.nodes
                         if state != 'out'],
                value='flat-ATP', continuous_update=False,
                description="Disc-Spiral reference state:"),
            # Source for ATP/ADP ratio:
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6395684/#:~:text=The%20physiological%20nucleotide%20concentration%20ratio,is%20~10%E2%88%925).
            'atp_adp_ratio': _DefaultFloatLogSlider(
                value=100, min=-1, max=4, readout_format='.1e',
                description="[ATP]/[ADP]:"),
            'equilibrium_atp_adp_ratio': _DefaultFloatLogSlider(
                value=1e-5, min=-7, max=1, readout_format='.1e',
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
            'k_TD': HTML(description="k_TD:"),
            'k_down': HTML(description="k_↓:"),
            'k_h_bar': HTML(description="ꝁ_h:"),
            'k_flat_to_extended_down_bar': HTML(description="ꝁ_⮯:"),
            'k_extended_to_flat_down': HTML(description="k_⮩:"),
        }

    def _construct_gui(self) -> Widget:
        gui_plot = Output()
        gui_parameters = VBox([
            HTML(value="<h1>Non-Ideal Models</h1>"),

            HBox([self._free_parameters['k_out_range'],
                  HTML(value="Order of out-of-main-loop rate")]),
            HBox([self._free_parameters['k_in'],
                  HTML(value="Back-in-main-loop rate")]),

            HTML(value="<b>General Physical Parameters</b>"),
            HBox([self._free_parameters['atp_adp_ratio'],
                  HTML(value="ATP/ADP concentration ratio")]),
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

            HTML(value="<b>SC/2R Model Physical Parameters</b>"),
            HBox([self._free_parameters['sc2r_reference_state'],
                  HTML(value="SC/2R out-of-main-loop reference state")]),
            HBox([self._free_parameters['k_up'],
                  HTML(value="Translocation up rate")]),
            HBox([self._constrained_parameters['k_down'],
                  HTML(value="Translocation down rate "\
                       "(constrained by detailed balance)")]),

            HTML(value="<b>Disc-Spiral Model Physical Parameters</b>"),
            HBox([self._free_parameters['disc_spiral_reference_state'],
                  HTML(value="Disc-Spiral out-of-main-loop reference state")]),
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
        models = [self._non_ideal_sc2r, self._non_ideal_disc_spiral]
        # Update models<->GUI
        self._update_models_free_parameters(models, self._free_parameters)
        self._update_gui_constrained_parameters(models,
                                                self._constrained_parameters)
        self._non_ideal_sc2r.reference_state = \
            self._free_parameters['sc2r_reference_state'].value
        self._non_ideal_disc_spiral.reference_state = \
            self._free_parameters['disc_spiral_reference_state'].value
        
        # Probability of being in the out state vs k_out range
        min, max = self._free_parameters['k_out_range'].value
        k_outs = np.logspace(min, max, 100)
        probabilities = {model: [] for model in models}
        for model in models:
            for k_out in k_outs:
                model.k_out = k_out
                probabilities[model].append(1-model._compute_probabilities()['out'])

        gui_plot = self._gui.children[0]
        with gui_plot:
            gui_plot.clear_output(wait=True)
            plt.close('NonIdeal')
            fig = plt.figure('NonIdeal')
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_visible = False
            ax = fig.add_subplot(111)

            # Plot velocities vs [ATP]/[ADP]
            sc2r_plot = ax.plot(k_outs,
                    probabilities[self._non_ideal_sc2r],
                    label="Non-Ideal SC/2R",
                    color='#DDAA33')
            disc_spiral_plot = ax.plot(k_outs,
                    probabilities[self._non_ideal_disc_spiral],
                    label="Non-Ideal Disc-Spiral",
                    color='#004488')

            ax.set_xlabel("k_out [k]")
            ax.set_ylabel("ℙ(main loop)")
            ax.legend()
            plt.show()


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
