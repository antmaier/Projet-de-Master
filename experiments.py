from translocation_model import TranslocationModel, SC2R, SC2R2Loops, DiscSpiral, DefectiveSC2R, DefectiveDiscSpiral
from gui import GUI
from ipywidgets import interactive_output, fixed, HBox, VBox, HTML
import matplotlib.pyplot as plt

def update_model_parameters(
        models: TranslocationModel | list[TranslocationModel],
        **kwargs,
) -> None:
    """Update the parameters of the given model(s).

    Typically called by an interactive_output widget:
    interactive_output(update_model_parameters, 
                       {'models': fixed(models), 
                       **parameters})
    where parameters is a dictionary {parameter_name: parameter_widget}.
    The parameters key must match the name of the corresponding parameters in 
    the model.

    Args:
        models: A TranslocationModel or a list of TranslocationModels.
        kwargs: A dictionary {parameter_name: parameter_value} of parameters to
            update.
    """
    if isinstance(models, TranslocationModel):
        models = [models]
    for model in models:
        for key, value in kwargs.items():
            if key in vars(model):
                setattr(model, key, value)

def trajectories():
    gui = GUI()
    gui.add_general_parameters()
    gui.add_sc2r_parameters()

    sc2r = SC2R()

    def experiment(
        model: TranslocationModel,
        **physical_parameters
    ) -> None:
        update_model_parameters(model, **physical_parameters)
        trajectory = model.gillespie(cumulative_sums='position')

        plt.close("trajectories")
        fig = plt.figure("trajectories")
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_visible = False
        ax = fig.add_subplot(111)

        sc2r.plot_position_evolution(trajectory, ax=ax)
        plt.show()

    out = interactive_output(experiment, {'model': fixed(sc2r), **gui.parameters})
    gui = HBox([out, gui.gui])
    gui.layout.align_items = 'center'
    gui.layout.height = '500px'
    gui = VBox([HTML(value='<h1>Simple trajectories</h1>'), gui])
    return gui

