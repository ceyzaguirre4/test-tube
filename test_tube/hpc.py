import datetime
import os
import signal
import sys
import time
import traceback
from subprocess import call

from .argparse_hopt import HyperOptArgumentParser


def exit():
    time.sleep(1)
    os._exit(1)


class AbstractCluster(object):

    RUN_CMD = 'sbatch'
    def __init__(
            self,
            hyperparam_optimizer=None,
            log_path=None,
            python_cmd='python3',
            enable_log_err=True,
            enable_log_out=True,
    ):
        self.hyperparam_optimizer = hyperparam_optimizer
        self.log_path = log_path

        self.enable_log_err = enable_log_err
        self.enable_log_out = enable_log_out
        self.slurm_files_log_path = None
        self.err_log_path = None
        self.out_log_path = None
        self.modules = []
        self.script_name = os.path.realpath(sys.argv[0])
        self.job_time = '15:00'
        self.minutes_to_checkpoint_before_walltime = 5
        self.per_experiment_nb_gpus = 1
        self.per_experiment_nb_cpus = 1
        self.per_experiment_nb_nodes = 1
        self.memory_mb_per_node = 2000
        self.email = None
        self.notify_on_end = False
        self.notify_on_fail = False
        self.job_name = None
        self.python_cmd = python_cmd
        self.gpu_type = None
        self.on_gpu = False
        self.call_load_checkpoint = False
        self.commands = []
        self.slurm_commands = []
        self.hpc_exp_number = 0

        # these are set via getters and setters so we can use a BaseManager which can be shared across processes
        self.checkpoint_save_function = None
        self.checkpoint_load_function = None

        # detect when this was called because a slurm object started a hopt.
        # if true, remove the flag so tt logs don't show it
        if hyperparam_optimizer is not None:
            self.is_from_slurm_object = HyperOptArgumentParser.TRIGGER_CMD in vars(self.hyperparam_optimizer) and vars(self.hyperparam_optimizer)[HyperOptArgumentParser.TRIGGER_CMD] == True
            if self.is_from_slurm_object:
                self.hyperparam_optimizer.__delattr__(HyperOptArgumentParser.TRIGGER_CMD)

            self.call_load_checkpoint = HyperOptArgumentParser.SLURM_LOAD_CMD in vars(self.hyperparam_optimizer)
            if self.call_load_checkpoint:
                self.hyperparam_optimizer.__delattr__(HyperOptArgumentParser.SLURM_LOAD_CMD)

            self.hpc_exp_number = self.hyperparam_optimizer.hpc_exp_number

    def set_checkpoint_save_function(self, fx, kwargs):
        self.checkpoint_save_function = [fx, kwargs]

    def get_checkpoint_save_function(self):
        return self.checkpoint_save_function

    def set_checkpoint_load_function(self, fx, kwargs):
        # if we were passed in the load flag, then we call the load function as soon as it's added
        if self.call_load_checkpoint:
            fx(**kwargs)

        self.checkpoint_load_function = [fx, kwargs]

    def get_checkpoint_load_function(self):
        return self.checkpoint_load_function

    def add_slurm_cmd(self, cmd, value, comment):
        self.slurm_commands.append((cmd, value, comment))

    def add_command(self, cmd):
        self.commands.append(cmd)

    def load_modules(self, modules):
        self.modules = modules

    def notify_job_status(self, email, on_done, on_fail):
        self.email = email
        self.notify_on_end = on_done
        self.notify_on_fail = on_fail

    def optimize_parallel_cluster(self, train_function, nb_trials, job_name):
        raise NotImplementedError

    def optimize_parallel_slurm(self, job_name, output_file, error_file, job_time, nb_gpus, nb_nodes, memory, notifications_email, gpu_types):
        pass


class SlurmCluster(AbstractCluster):
    def __init__(self, *args, **kwargs):
        super(SlurmCluster, self).__init__(*args, **kwargs)

    def optimize_parallel_cluster_gpu(
            self,
            train_function,
            nb_trials,
            job_name,
            enable_auto_resubmit=False,
            job_display_name=None,
            max_parallel_trials=None,         # Run at most `max_parallel_trials` at the same time
            debug=False,                        # if debug no experiments will run
            slurm_account=None,
    ):
        if job_display_name is None:
            job_display_name = job_name
        
        self.max_parallel_trials = max_parallel_trials
        self.slurm_account = slurm_account

        self.__optimize_parallel_cluster_internal(train_function, nb_trials, job_name, job_display_name,
                                                  enable_auto_resubmit, on_gpu=True, debug=debug)

    def optimize_parallel_cluster_cpu(
            self,
            train_function,
            nb_trials,
            job_name,
            enable_auto_resubmit=False,
            job_display_name=None,
            max_parallel_trials=None,         # Run at most `max_parallel_trials` at the same time
            debug=False,                        # if debug no experiments will run
            slurm_account=None,
    ):
        if job_display_name is None:
            job_display_name = job_name
        
        self.max_parallel_trials = max_parallel_trials
        self.slurm_account = slurm_account

        self.__optimize_parallel_cluster_internal(train_function, nb_trials, job_name, job_display_name,
                                                  enable_auto_resubmit, on_gpu=False, debug=debug)

    def __optimize_parallel_cluster_internal(
            self,
            train_function,
            nb_trials,
            job_name,
            job_display_name,
            enable_auto_resubmit,
            on_gpu,
            debug=False,

    ):
        """
        Runs optimization on the attached cluster
        :param train_function:
        :param nb_trials:
        :param job_name:
        :return:
        """
        self.job_name = job_name
        self.job_display_name = job_display_name
        self.on_gpu = on_gpu
        self.enable_auto_resubmit = enable_auto_resubmit

        if self.is_from_slurm_object:
            # Script is called by slurm: it's an actual experiment.
            self.__run_experiment(train_function)
        else:
            # Launcher script. Generate trials and launch jobs.

            # layout logging structure
            self.__layout_logging_dir()

            # generate hopt trials
            trials = self.hyperparam_optimizer.generate_trials(nb_trials)

            # get the max test tube exp version so far if it's there
            scripts_parent_dir = os.path.join(self.log_path, 'slurm_out_logs')
            next_trial_version = self.__get_max_trial_version(scripts_parent_dir)

            # for each trial, generate a slurm command
            script_paths = []
            for i, trial_params in enumerate(trials):
                exp_i = i + next_trial_version
                experiment_script_path = self.generate_experiment_script(trial_params, exp_i)
                script_paths.append(experiment_script_path)

            # schedule them all together in a single slurm array and launch array
            slurm_cmd_path = self.schedule_array(script_paths)
            if not debug:
                print('\nlaunching exp...')
                result = call('{} {}'.format(AbstractCluster.RUN_CMD, slurm_cmd_path), shell=True)
                if result == 0:
                    print('launched exp ', slurm_cmd_path)
                else:
                    print('launch failed...')
            else:
                print('\nDEBUG: skipping launch...', slurm_cmd_path)
    
    def schedule_array(self, script_paths):
        slurm_cmd_path = os.path.join(self.slurm_files_log_path, 'run_trials.sh')
        slurm_cmd = self.__build_slurm_command(self.on_gpu, script_paths)
        self.__save_script(slurm_cmd, slurm_cmd_path)
        return slurm_cmd_path

    def generate_experiment_script(self, trial_params, exp_i):
        """
        Generates the experiment script and saves it to `script_path`.

        Returns `script_path`.
        """

        # generate command
        script_path = os.path.join(self.slurm_files_log_path, 'trial_{}.sh'.format(exp_i))
        script = self.__build_experiment_script(trial_params, script_path, exp_i)
        self.__save_script(script, script_path)

        return script_path

    def call_save(self):
        print('calling save')

        # if save function was passed, call it
        if self.get_checkpoint_save_function() is not None:
            save_fx, kwargs = self.get_checkpoint_save_function()
            save_fx(**kwargs)

            # if we're here, the job didn't finish and we were given a save function
            # if we were given a load function, then schedule the program again and pass in the load function
            if self.get_checkpoint_load_function() is not None:
                job_id = os.environ['SLURM_JOB_ID']
                cmd = 'scontrol requeue {}'.format(job_id)

                print('\nrequeing job {}...'.format(job_id))
                result = call(cmd, shell=True)
                if result == 0:
                    print('requeued exp ', job_id)
                else:
                    print('requeue failed...')

        # stop program
        os._exit(0)

    def sig_handler(self, signum, frame):
        print("caught signal", signum)
        self.call_save()
        # sys.exit(-1)

    # ------------------------
    # HANDLE SLURM SIGNALS
    # ------------------------
    def term_handler(self, signum, frame):
        print("bypassing sigterm")

    def __run_experiment(self, train_function):
        if self.enable_auto_resubmit:
            print('setting signal')
            signal.signal(signal.SIGUSR1, self.sig_handler)
            signal.signal(signal.SIGTERM, self.term_handler)

        # if its part of a Weights and Biases sweep then get hyperparams from the controller
        if "wandb_project_name" in self.hyperparam_optimizer and "wandb_sweep_id" in self.hyperparam_optimizer:
            # TODO: move code to its own file
            import wandb

            def wrapped_train_function():
                with wandb.init() as run:
                    # update the hyperparams with the ones from the wandb controller
                    config = wandb.config
                    for param_name, param_value in config.items():
                        setattr(self.hyperparam_optimizer, param_name, param_value)

                    try:
                        # run training
                        train_function(self.hyperparam_optimizer, self)

                    except Exception as e:
                        print('Caught exception in worker thread', e)

                        # This prints the type, value, and stack trace of the
                        # current exception being handled.
                        traceback.print_exc()
                        raise SystemExit

            wandb.agent(
                sweep_id=self.hyperparam_optimizer.wandb_sweep_id,
                project=self.hyperparam_optimizer.wandb_project_name,
                function=wrapped_train_function,
                count=1,                                                # run one experiment per agent
            )

        else:
            try:
                # run training
                train_function(self.hyperparam_optimizer, self)

            except Exception as e:
                print('Caught exception in worker thread', e)

                # This prints the type, value, and stack trace of the
                # current exception being handled.
                traceback.print_exc()
                raise SystemExit

    def __save_script(self, script, script_path):
        with open(script_path, mode='w') as file:
            file.write(script)

    def __get_max_trial_version(self, path):
        files = os.listdir(path)
        version_files = [f for f in files if 'trial_' in f]
        if len(version_files) > 0:
            # regex out everything except file version for ve
            versions = [int(f_name.split('_')[1]) for f_name in version_files]
            max_version = max(versions)
            return max_version + 1
        else:
            return 0

    def __layout_logging_dir(self):
        """
        Generates dir structure for logging errors and outputs
        :return:
        """

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

        # creates a timestamped subdirectory with all the experiments

        slurm_out_path = os.path.join(self.log_path, self.job_name, timestamp)

        self.log_path = slurm_out_path

        # if we have a test tube name, make the folder and set as the logging destination
        if not os.path.exists(slurm_out_path):
            os.makedirs(slurm_out_path)

        # when err logging is enabled, build add the err logging folder
        if self.enable_log_err:
            err_path = os.path.join(slurm_out_path, 'slurm_err_logs')
            if not os.path.exists(err_path):
                os.makedirs(err_path)
            self.err_log_path = err_path

        # when out logging is enabled, build add the out logging folder
        if self.enable_log_out:
            out_path = os.path.join(slurm_out_path, 'slurm_out_logs')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            self.out_log_path = out_path

        # place where slurm files log to
        self.slurm_files_log_path = os.path.join(slurm_out_path, 'slurm_scripts')
        if not os.path.exists(self.slurm_files_log_path):
            os.makedirs(self.slurm_files_log_path)

    def __get_hopt_params(self, trial):
        """
        Turns hopt trial into script params
        :param trial:
        :return:
        """

        params = []
        for k in trial.__dict__:
            v = trial.__dict__[k]

            # don't add None params
            if v is None: # or v is False:
                continue

            # put everything in quotes except bools
            if self.__should_escape(v):
                cmd = '--{} \"{}\"'.format(k, v)
            else:
                cmd = '--{} {}'.format(k, v)
            params.append(cmd)

        # this arg lets the hyperparameter optimizer do its thing
        params.append('--{}'.format(HyperOptArgumentParser.TRIGGER_CMD))

        full_cmd = ' '.join(params)
        return full_cmd

    def __should_escape(self, v):
        v = str(v)
        return '[' in v or ';' in v or ' ' in v

    def __build_slurm_command(self, on_gpu, script_paths):
        sub_commands = []

        command =[
            '#!/bin/bash',
            '#',
            '# Auto-generated by test-tube (https://github.com/williamFalcon/test-tube)',
            '#################\n'
        ]
        sub_commands.extend(command)

        # add account info
        if self.slurm_account:
            command = [
                '# set a account info',
                '#SBATCH --account={}'.format(self.slurm_account),
                '#################\n',
            ]
            sub_commands.extend(command)

        # add job name
        command = [
            '# set a job name',
            '#SBATCH --job-name={}'.format(self.job_display_name),
            '#################\n',
        ]
        sub_commands.extend(command)

        # add out output
        if self.enable_log_out:
            out_path = os.path.join(self.out_log_path, "%A-%a.log")
            command = [
                '# a file for job output, you can check job progress',
                '#SBATCH --output={}'.format(out_path),
                '#################\n',
            ]
            sub_commands.extend(command)

        # add err output
        if self.enable_log_err:
            err_path = os.path.join(self.err_log_path, "%A-%a.err")
            command = [
                '# a file for errors',
                '#SBATCH --error={}'.format(err_path),
                '#################\n',
            ]
            sub_commands.extend(command)

        # add job time
        command = [
            '# time needed for job',
            '#SBATCH --time={}'.format(self.job_time),
            '#################\n'
        ]
        sub_commands.extend(command)

        # add nb of gpus
        if self.per_experiment_nb_gpus > 0 and on_gpu:
            command = [
                '# gpus per node',
                '#SBATCH --gres=gpu:{}'.format(self.per_experiment_nb_gpus),
                '#################\n'
            ]
            if self.gpu_type is not None:
                command = [
                    '# gpus per node',
                    '#SBATCH --gres=gpu:{}:{}'.format(self.gpu_type, self.per_experiment_nb_gpus),
                    '#################\n'
                ]
            sub_commands.extend(command)

        # add nb of cpus if not looking at a gpu job
        if self.per_experiment_nb_cpus > 0:
            command = [
                '# cpus per job',
                '#SBATCH --cpus-per-task={}'.format(self.per_experiment_nb_cpus),
                '#################\n'
            ]
            sub_commands.extend(command)

        # pick nb nodes
        command = [
            '# number of requested nodes',
            '#SBATCH --nodes={}'.format(self.per_experiment_nb_nodes),
            '#################\n'
        ]
        sub_commands.extend(command)

        # pick memory per node
        command = [
            '# memory per node',
            '#SBATCH --mem={}'.format(self.memory_mb_per_node),
            '#################\n'
        ]
        sub_commands.extend(command)

        # add signal command to catch job termination
        command = [
            '# slurm will send a signal this far out before it kills the job',
            f'#SBATCH --signal=USR1@{self.minutes_to_checkpoint_before_walltime * 60}',
            '#################\n'
        ]
        sub_commands.extend(command)

        # add slurm array command to launch multiple times
        if self.max_parallel_trials != None:      # Run at most `max_parallel_trials` at the same time
            command = [
                '# run as slurm array ',
                f'#SBATCH --array=0-{len(script_paths)-1}%{self.max_parallel_trials}',
                '#################\n'
            ]
        else:
            command = [
                '# run as slurm array ',
                f'#SBATCH --array=0-{len(script_paths)-1}',     # TODO: support for multiple scripts per GPU
                '#################\n'
            ]
        sub_commands.extend(command)

        # Subscribe to email if requested
        mail_type = []
        if self.notify_on_end:
            mail_type.append('END')
        if self.notify_on_fail:
            mail_type.append('FAIL')
        if len(mail_type) > 0:
            mail_type_query = [
                '# Have SLURM send you an email when the job ends or fails',
                '#SBATCH --mail-type={}'.format(','.join(mail_type))
            ]
            sub_commands.extend(mail_type_query)

            email_query = [
                '#SBATCH --mail-user={}'.format(self.email),
            ]
            sub_commands.extend(email_query)

        # add custom sbatch commands
        sub_commands.append('\n')
        for (cmd, value, comment) in self.slurm_commands:
            comment = '# {}'.format(comment)
            cmd = '#SBATCH --{}={}'.format(cmd, value)
            spaces = '#################\n'
            sub_commands.extend([comment, cmd, spaces])

        script_paths = [f'"{path}"' for path in script_paths]
        cmd = f"SCRIPT_ARRAY=( {' '.join(script_paths)} )"
        cmd += """
echo ${SCRIPT_ARRAY[SLURM_ARRAY_TASK_ID]}
sh ${SCRIPT_ARRAY[SLURM_ARRAY_TASK_ID]}
        """
        sub_commands.append(cmd)

        # remove spaces before the hash
        sub_commands = [x.lstrip() for x in sub_commands]

        # build full command with empty lines in between
        full_command = '\n'.join(sub_commands)
        return full_command

    def __build_experiment_script(self, trial, slurm_cmd_script_path, exp_i):
        sub_commands = []

        command =[
            '#!/bin/bash',
            '#',
            '# Auto-generated by test-tube (https://github.com/williamFalcon/test-tube)',
            '#################\n'
        ]
        sub_commands.extend(command)

        # load modules
        sub_commands.append('\n')
        for module in self.modules:
            cmd = 'module load {}'.format(module)
            sub_commands.append(cmd)

        # remove spaces before the hash
        sub_commands = [x.lstrip() for x in sub_commands]

        # add additional commands
        for cmd in self.commands:
            sub_commands.append(cmd)
            sub_commands.append('\n')

        # add run command
        trial_args = self.__get_hopt_params(trial)
        trial_args = '{} --{} {} --{} {}'.format(trial_args,
                                                 HyperOptArgumentParser.SLURM_CMD_PATH,
                                                 slurm_cmd_script_path,
                                                 HyperOptArgumentParser.SLURM_EXP_CMD,
                                                 exp_i)

        cmd = '{} {} {}'.format(self.python_cmd, self.script_name, trial_args)
        sub_commands.append(cmd)

        # build full command with empty lines in between
        full_command = '\n'.join(sub_commands)
        return full_command
