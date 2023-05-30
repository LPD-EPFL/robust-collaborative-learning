import tools
tools.success("Module loading...")
import argparse
import math
import pathlib
import signal
import shlex
import sys
import torch
import experiments

# ---------------------------------------------------------------------------- #
# Miscellaneous initializations
tools.success("Miscellaneous initializations...")

# "Exit requested" global variable accessors
exit_is_requested, exit_set_requested = tools.onetime("exit")

# Signal handlers
signal.signal(signal.SIGINT, exit_set_requested)
signal.signal(signal.SIGTERM, exit_set_requested)

# ---------------------------------------------------------------------------- #
# Command-line processing
tools.success("Command-line processing...")

def process_commandline():
  """ Parse the command-line and perform checks.
  Returns:
    Parsed configuration
  """
  # Description
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument("--result-directory",
    type=str,
    default="results-data",
    help="Path of the data directory, containing the data gathered from the experiments")
  parser.add_argument("--plot-directory",
    type=str,
    default="results-plot",
    help="Path of the plot directory, containing the graphs traced from the experiments")
  parser.add_argument("--devices",
    type=str,
    default="auto",
    help="Comma-separated list of devices on which to run the experiments, used in a round-robin fashion")
  parser.add_argument("--supercharge",
    type=int,
    default=1,
    help="How many experiments are run in parallel per device, must be positive")
  # Parse command line
  return parser.parse_args(sys.argv[1:])

with tools.Context("cmdline", "info"):
  args = process_commandline()
  # Check the "supercharge" parameter
  if args.supercharge < 1:
    tools.fatal(f"Expected a positive supercharge value, got {args.supercharge}")
  # Make the result directories
  def check_make_dir(path):
    path = pathlib.Path(path)
    if path.exists():
      if not path.is_dir():
        tools.fatal(f"Given path {str(path)!r} must point to a directory")
    else:
      path.mkdir(mode=0o755, parents=True)
    return path
  args.result_directory = check_make_dir(args.result_directory)
  args.plot_directory = check_make_dir(args.plot_directory)
  # Preprocess/resolve the devices to use
  if args.devices == "auto":
    if torch.cuda.is_available():
      args.devices = list(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    else:
      args.devices = ["cpu"]
  else:
    args.devices = list(name.strip() for name in args.devices.split(","))

# ---------------------------------------------------------------------------- #
# Serial preloading of the dataset
tools.success("Pre-downloading datasets...")

# Pre-load the datasets to prevent the first parallel runs from downloading them several times
with tools.Context("dataset", "info"):
  for name in ("mnist", "cifar10"):
    with tools.Context(name, "info"):
      experiments.make_datasets(name)

# ---------------------------------------------------------------------------- #
# Run (missing) experiments
tools.success("Running experiments...")

# Command maker helper
def make_command(params):
  cmd = ["python3", "-OO", "peerToPeer.py"]
  cmd += tools.dict_to_cmdlist(params)
  return tools.Command(cmd)

# Jobs
jobs  = tools.Jobs(args.result_directory, devices=args.devices, devmult=args.supercharge)
seeds = jobs.get_seeds()

# Base parameters for the MNIST experiments
params_mnist = {
  "batch-size": 25,
  "model": "simples-conv",
  "loss": "nll",
  "learning-rate-decay-delta": 50,
  "learning-rate-decay": 50,
  "l2-regularize": 1e-4,
  "evaluation-delta": 5,
  "nb-steps": 600,
  "nb-workers": 26,
  "momentum-at": "worker",
  "learning-rate": 0.75,
  "dataset": "mnist",
  "gradient-clip-centered": 2,
  "numb-labels": 10
  }


# Hyperparameters to test
momentums = [0, 0.99]
gars = ["cva", "trmean", "rfa"]
attacks = ["little", "empire", "signflipping", "labelflipping"]
dataset = "mnist"
params_common = params_mnist
byzcounts = [5]
alphas = [1, 5]

result_directory = "results-data"
plot_directory = "results-plot"


# Submit all experiments
for alpha in alphas:
    params = params_common.copy()
    params["dirichlet-alpha"] = alpha
    #JS: DSGD
    params["momentum"] = params["dampening"] = 0.99
    jobs.submit(f"{dataset}-average-n_{params['nb-workers']}-m_{0.99}-model_{params['model']}-alpha_{alpha}_dsgd", make_command(params))

    #JS: Attacks
    for f in byzcounts:
        for gar in gars:
            for attack in attacks:
                for momentum in momentums:
                    params = params_common.copy()
                    params["dirichlet-alpha"] = alpha
                    params["momentum"] = params["dampening"] = momentum
                    params["nb-decl-byz"] = params["nb-real-byz"] = f
                    params["gar"] = gar
                    params["attack"] = attack
                    if attack == "labelflipping":
                        params["flip"] = True
                    if attack == "mimic_heuristic":
                        params["mimic-heuristic"] = True
                    jobs.submit(f"{dataset}-{attack}-{gar}-f_{f}-m_{momentum}-model_{params['model']}-alpha_{alpha}", make_command(params))

    #JS: Self-centered clipping with attacks
    for f in byzcounts:
        for attack in attacks:
            params = params_common.copy()
            params["momentum"] = params["dampening"] = 0.9
            params["nb-decl-byz"] = params["nb-real-byz"] = f
            params["gar"] = "centeredclip"
            params["attack"] = attack
            if attack == "labelflipping":
                params["flip"] = True
            if attack == "mimic_heuristic":
                params["mimic-heuristic"] = True
            jobs.submit(f"{dataset}-{attack}-scc-f_{f}-m_{0.9}-model_{params['model']}-alpha_{alpha}", make_command(params))

    #JS: Jungle with attacks
    for f in byzcounts:
        for attack in attacks:
            params = params_common.copy()
            params["dirichlet-alpha"] = alpha
            params["jungle"] = True
            params["nb-decl-byz"] = params["nb-real-byz"] = f
            params["gar"] = "trmean"
            params["attack"] = attack
            if attack == "labelflipping":
                params["flip"] = True
            if attack == "mimic_heuristic":
                params["mimic-heuristic"] = True
            jobs.submit(f"{dataset}-{attack}-trmean-f_{f}-m_{0}-model_{params['model']}-alpha_{alpha}_jungle", make_command(params))

# Wait for the jobs to finish and close the pool
jobs.wait(exit_is_requested)
jobs.close()

# Check if exit requested before going to plotting the results
if exit_is_requested():
  exit(0)

# Import additional modules
try:
 import numpy
 import pandas
 import study
except ImportError as err:
 tools.fatal(f"Unable to plot results: {err}")


def compute_avg_err_op(name, location, *colops, avgs="", errs="-err"):
  """ Compute the average and standard deviation of the selected columns over the given experiment.
  Args:
    name Given experiment name
    location Script to read from
    ...  Tuples of (selected column name (through 'study.select'), optional reduction operation name)
    avgs Suffix for average column names
    errs Suffix for standard deviation (or "error") column names
  Returns:
    Data frames for each of the computed columns,
    Tuple of reduced values per seed (or None if None was provided for 'op')
  Raises:
    'RuntimeError' if a reduction operation was specified for a column selector that did not select exactly 1 column
  """
# Load all the runs for the given experiment name, and keep only a subset
  datas = tuple(study.select(study.Session(result_directory + "/" + name + "-" +str(seed), location), *(col for col, _ in colops)) for seed in seeds)

  # Make the aggregated data frames
  def make_df_ro(col, op):
    nonlocal datas
    # For every selected columns
    subds = tuple(study.select(data, col).dropna() for data in datas)
    df    = pandas.DataFrame(index=subds[0].index)
    ro    = None
    for cn in subds[0]:
      # Generate compound column names
      avgn = cn + avgs
      errn = cn + errs
      # Compute compound columns
      numds = numpy.stack(tuple(subd[cn].to_numpy() for subd in subds))
      df[avgn] = numds.mean(axis=0)
      df[errn] = numds.std(axis=0)
      # Compute reduction, if requested
      if op is not None:
        if ro is not None:
          raise RuntimeError(f"column selector {col!r} selected more than one column ({(', ').join(subds[0].columns)}) while a reduction operation was requested")
        ro = tuple(getattr(subd[cn], op)().item() for subd in subds)
    # Return the built data frame and optional computed reduction
    return df, ro
  dfs = list()
  ros = list()
  for col, op in colops:
    df, ro = make_df_ro(col, op)
    dfs.append(df)
    ros.append(ro)
  # Return the built data frames and optional computed reductions
  return dfs, ros

# Plot results
with tools.Context("mnist", "info"):
    for alpha in alphas:

        #JS: DSGD
        name = f"{dataset}-average-n_{params_common['nb-workers']}-m_{0.99}-model_{params_common['model']}-alpha_{alpha}_dsgd"
        try:
          dsgd, _ = compute_avg_err_op(name, "eval", ("Accuracy", "max"))
        except Exception as err:
          tools.warning(f"Unable to process {name}: {err}")
          continue

        #JS: Attacks
        for f in byzcounts:
            for attack in attacks:
                attacked = dict()

                #MoNNA
                name = f"{dataset}-{attack}-cva-f_{f}-m_{0.99}-model_{params_common['model']}-alpha_{alpha}"
                try:
                  cols, _ = compute_avg_err_op(name, "eval", ("Accuracy", "max"))
                  attacked[("cva", 0.99)] = cols
                except Exception as err:
                  tools.warning(f"Unable to process {name !r}: {err}")
                  continue

                #NNA - beta = 0
                name = f"{dataset}-{attack}-cva-f_{f}-m_{0}-model_{params_common['model']}-alpha_{alpha}"
                try:
                  cols, _ = compute_avg_err_op(name, "eval", ("Accuracy", "max"))
                  attacked[("cva", 0)] = cols
                except Exception as err:
                  tools.warning(f"Unable to process {name !r}: {err}")
                  continue

                #CWTM - beta = 0.99
                name = f"{dataset}-{attack}-trmean-f_{f}-m_{0.99}-model_{params_common['model']}-alpha_{alpha}"
                try:
                  cols, _ = compute_avg_err_op(name, "eval", ("Accuracy", "max"))
                  attacked[("trmean", 0.99)] = cols
                except Exception as err:
                  tools.warning(f"Unable to process {name !r}: {err}")
                  continue

                #CWTM - beta = 0
                name = f"{dataset}-{attack}-trmean-f_{f}-m_{0}-model_{params_common['model']}-alpha_{alpha}"
                try:
                  cols, _ = compute_avg_err_op(name, "eval", ("Accuracy", "max"))
                  attacked[("trmean", 0)] = cols
                except Exception as err:
                  tools.warning(f"Unable to process {name !r}: {err}")
                  continue

                #GM - beta = 0.99
                name = f"{dataset}-{attack}-rfa-f_{f}-m_{0.99}-model_{params_common['model']}-alpha_{alpha}"
                try:
                  cols, _ = compute_avg_err_op(name, "eval", ("Accuracy", "max"))
                  attacked[("rfa", 0.99)] = cols
                except Exception as err:
                  tools.warning(f"Unable to process {name !r}: {err}")
                  continue

                #GM - beta = 0
                name = f"{dataset}-{attack}-rfa-f_{f}-m_{0}-model_{params_common['model']}-alpha_{alpha}"
                try:
                  cols, _ = compute_avg_err_op(name, "eval", ("Accuracy", "max"))
                  attacked[("rfa", 0)] = cols
                except Exception as err:
                  tools.warning(f"Unable to process {name !r}: {err}")
                  continue

                #SCC
                name = f"{dataset}-{attack}-scc-f_{f}-m_{0.9}-model_{params_common['model']}-alpha_{alpha}"
                try:
                  cols, _ = compute_avg_err_op(name, "eval", ("Accuracy", "max"))
                  attacked["scc"] = cols
                except Exception as err:
                  tools.warning(f"Unable to process {name !r}: {err}")
                  continue

                #Jungle
                name = f"{dataset}-{attack}-trmean-f_{f}-m_{0}-model_{params_common['model']}-alpha_{alpha}_jungle"
                try:
                  cols, _ = compute_avg_err_op(name, "eval", ("Accuracy", "max"))
                  attacked[("jungle", 0)] = cols
                except Exception as err:
                  tools.warning(f"Unable to process {name !r}: {err}")
                  continue

                # Plot top-1 cross-accuracies
                plot = study.LinePlot()
                plot.include(dsgd[0], "Accuracy", errs="-err", lalp=0.8)
                legend = ["D-SGD"]

                plot.include(attacked[("cva", 0.99)][0], "Accuracy", errs="-err", lalp=0.8)
                legend.append("MoNNA")

                plot.include(attacked[("cva", 0)][0], "Accuracy", errs="-err", lalp=0.8)
                legend.append("NNA")

                plot.include(attacked[("trmean", 0.99)][0], "Accuracy", errs="-err", lalp=0.8)
                legend.append("MoCWTM")

                plot.include(attacked[("trmean", 0)][0], "Accuracy", errs="-err", lalp=0.8)
                legend.append("BRIDGE")
                #legend.append("CWTM")

                plot.include(attacked[("rfa", 0.99)][0], "Accuracy", errs="-err", lalp=0.8)
                legend.append("MoGM")

                plot.include(attacked[("rfa", 0)][0], "Accuracy", errs="-err", lalp=0.8)
                legend.append("GM")

                plot.include(attacked["scc"][0], "Accuracy", errs="-err", lalp=0.8)
                legend.append("SCC")

                plot.include(attacked[("jungle", 0)][0], "Accuracy", errs="-err", lalp=0.8)
                legend.append("LEARN")

                #JS: plot every time graph in terms of the maximum number of steps
                plot.finalize(None, "Iteration", "Test accuracy", xmin=0, xmax=params_common['nb-steps'], ymin=0, ymax=1, legend=legend)
                plot.save(plot_directory + "/" + dataset + "_" + params_common['model'] + "_" + attack + "_f=" + str(f) + "_alpha=" + str(alpha) + "_momentum.pdf", xsize=3, ysize=1.5)
