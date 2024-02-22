from typing import Dict

from prefect import flow, get_run_logger, task

from lume_services.results import Result
from lume_services.tasks import (
    configure_lume_services,
    prepare_lume_model_variables,
    check_local_execution,
    SaveDBResult,
    LoadDBResult,
    LoadFile,
    SaveFile,
)
from lume_services.files import TextFile
from lume_model.variables import InputVariable, OutputVariable

import torch
import matplotlib.pyplot as plt
import pandas as pd
import sys

from xopt import Xopt, VOCS
from xopt.evaluator import Evaluator
from xopt.numerical_optimizer import LBFGSOptimizer
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor

sys.path.append("calibration/calibration_modules/")

from calibration.calibration_modules.decoupled_linear import OutputOffset, DecoupledLinearOutput
from utils import load_model, get_model_predictions, get_running_optimum



# Xopt evaluator function
@task()
def evaluate(input_dict, lume_model=None, objective_model=None, vocs=None, generator=None, noise_level=0.1):
    model_result = lume_model.evaluate(input_dict)
    objective_value = objective_model.function(
        sigma_x=model_result["OTRS:IN20:571:XRMS"],
        sigma_y=model_result["OTRS:IN20:571:YRMS"],
    )
    noise = torch.normal(
        mean=torch.zeros(objective_value.shape),
        std=noise_level * torch.ones(objective_value.shape),
    )
    v = objective_value + noise
    output_dict = {vocs.objective_names[0]: v.detach().item()}

    # dummy constraint
    output_dict["c1"] = output_dict[vocs.objective_names[0]] - 1.0

    # GP model predictions
    model_predictions = get_model_predictions(input_dict, generator)
    output_dict.update(model_predictions)

    return output_dict


@flow(name="torch-nn")
def torch_nn_flow(model_parameters: Dict[str, any]):
    print('Starting Flow Run')
    # CONFIGURE LUME-SERVICES
    # see https://slaclab.github.io/lume-services/workflows/#configuring-flows-for-use-with-lume-services

    # configure = configure_lume_services()

    # CHECK WHETHER THE FLOW IS RUNNING LOCALLY
    # If the flow runs using a local backend, the results service will not be available
    # running_local = check_local_execution()
    # running_local.set_upstream(configure)

    variables = ["SOLN:IN20:121:BCTRL", "QUAD:IN20:121:BCTRL"]
    # variables = [
    #     "SOLN:IN20:121:BCTRL", "QUAD:IN20:121:BCTRL", "QUAD:IN20:122:BCTRL",
    #     "QUAD:IN20:361:BCTRL", "QUAD:IN20:371:BCTRL", "QUAD:IN20:425:BCTRL",
    #     "QUAD:IN20:441:BCTRL", "QUAD:IN20:511:BCTRL", "QUAD:IN20:525:BCTRL",
    # ]
    filename = "files/variables.csv"
    variable_ranges = pd.read_csv(filename, index_col=0, header=None).T.to_dict(orient='list')
    vocs = VOCS(
        variables={ele: variable_ranges[ele] for ele in variables},
        objectives={"total_size": "MINIMIZE"},
        constraints={"c1": ["LESS_THAN", 0.0]},
    )
    print(vocs.as_yaml())

    objective_model = load_model(
        input_variables=vocs.variable_names,
        model_path="lcls_cu_injector_nn_model/",
    )
    lume_model = objective_model.model.model

    # define miscalibrated objective model
    y_size = len(vocs.objective_names)
    miscal_model = DecoupledLinearOutput(
        model=objective_model,
        y_offset_initial=torch.full((y_size,), -0.5),
        y_scale_initial=torch.ones(y_size),
    )
    miscal_model.requires_grad_(False);

    # define prior mean
    prior_mean = OutputOffset(
        model=miscal_model,
    )

    vocs.variables = {
        k: lume_model.input_variables[lume_model.input_names.index(k)].value_range
        for k in vocs.variable_names
    }
    vocs.variables["SOLN:IN20:121:BCTRL"] = [0.467, 0.479]
    print(vocs.as_yaml())

    # remember to set use low noise prior to false!!!
    gp_constructor = StandardModelConstructor(
        use_low_noise_prior=False,
        mean_modules={vocs.objective_names[0]: prior_mean},
        trainable_mean_keys=[vocs.objective_names[0]],
    )
    generator = ExpectedImprovementGenerator(
        vocs=vocs,
        gp_constructor=gp_constructor,
    )
    generator.numerical_optimizer.max_iter = 200
    evaluator = Evaluator(function=evaluate, function_kwargs={"generator": None})
    X = Xopt(generator=generator, evaluator=evaluator, vocs=vocs)

    # pass generator to evaluator to compute model predictions
    X.evaluator = Evaluator(function=evaluate, function_kwargs={"lume_model": lume_model, "objective_model": objective_model, "vocs": vocs, "generator": X.generator})

    n_init = 3
    X.random_evaluate(n_init, seed=0)

    n_step = 50
    for i in range(n_step):
        X.step()

    opt = get_running_optimum(
        data=X.data,
        objective_name=X.vocs.objective_names[0],
        maximize=X.vocs.objectives[X.vocs.objective_names[0]].upper() == "MAXIMIZE",
    )

    print("Done")
    print(X)
    print(opt)


    #return output_variables

    # SAVE RESULTS TO RESULTS DATABASE, requires LUME-services results backend
    # if not running_local:
    #    # CREATE LUME-services Result object
    #    formatted_result = format_result(
    #        input_variables=input_variable_parameter_dict, output_variables=output_variables, output_variables_names=output_variables_names
    #    )

    # RUN DATABASE_SAVE_TASK
    #    saved_model_rep = save_db_result_task(formatted_result)
    #    saved_model_rep.set_upstream(configure)


def get_flow():
    return flow


if __name__ == '__main__':
    torch_nn_flow.serve(name="torch-nn-test")
