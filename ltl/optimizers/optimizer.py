from collections import namedtuple

from pypet import cartesian_product

from ltl import get_grouped_param_dict

OptimizerParameters = namedtuple('OptimizerParamters', [])


class Optimizer:
    """
    This is the base class for the Optimizers i.e. the outer loop algorithms. These algorithms generate parameters, \
    give them to the inner loop to be evaluated, and with the resulting fitness modify the parameters in some way.

    :param  ~pypet.trajectory.Trajectory traj: Use this pypet trajectory to store the parameters of the specific runs.
        The parameters should be initialized based on the values in :param parameters:

    :param optimizee_create_individual: A function which when called returns one instance of parameter (or "individual")

    :param optimizee_fitness_weights: The weights which should be multiplied with the fitness returned from the
        :class:`~ltl.optimizees.optimizee.Optimizee` -- one for each element of the fitness (fitness can be
        multi-dimensional). If some element is negative, the Optimizer minimizes that element of fitness instead of
        maximizing. By default, the `Optimizer` maximizes all fitness dimensions.

    :param parameters: A named tuple containing the parameters for the Optimizer class
    
    """

    def __init__(self, traj, optee_create_indiv, optee_indiv_param_spec, optee_fitness_weights, parameters):

        # Creating Placeholders for individuals and results that are about to be explored
        traj.f_add_parameter('generation', 0, comment='Current generation')
        traj.f_add_parameter('ind_idx', 0, comment='Index of individual')
        traj.f_add_parameter_group('individual', 'An individual of the population')
        self.init_individual = optee_create_indiv()
        for key, val in self.init_individual.items():
            traj.par.individual.f_add_parameter(key, val)

        # Initializing basic variables
        self.optee_create_indiv = optee_create_indiv
        self.optee_indiv_param_spec = optee_indiv_param_spec
        self.optee_fitness_weights = optee_fitness_weights

        #: The current generation number
        self.g = None
        #: The population (i.e. list of individuals) to be evaluated at the next iteration
        self.eval_pop = None

    def post_process(self, traj, fitnesses_results):
        """
        This is the key function of this class. Given a set of :obj:`fitnesses_results`,  and the :obj:`traj`, it uses
        the fitness to decide on the next set of parameters to be evaluated. Then it fills the :attr:`.eval_pop` with the
        list of parameters it wants evaluated at the next simulation cycle, increments :attr:`.g` and calls
        :meth:`._expand_trajectory`

        :param  ~pypet.trajectory.Trajectory traj: The :mod:`pypet` trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :param list fitnesses_results: This is a list of fitness results that contain tuples run index and the fitness.
            It is of the form `[(run_idx, run), ...]` as constructed by :mod:`pypet`

        """
        # NOTE: Always remember to keep the following two lines.
        # TODO: Set eval_pop to the values of parameters you want to evaluate in the next cycle
        # self.eval_pop = ...
        self.g += 1
        self._expand_trajectory(traj)

    def end(self):
        """
        Run any code required to clean-up, print final individuals etc.
        """
        pass

    def _expand_trajectory(self, traj):
        """
        Add as many explored runs as individuals that need to be evaluated. Furthermore, add the individuals as explored
        parameters.

        :param  ~pypet.trajectory.Trajectory traj: The :mod:`pypet` trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :return:
        """

        grouped_params_dict = get_grouped_param_dict(self.eval_pop, self.optee_indiv_param_spec)

        final_params_dict = {'generation': [self.g],
                             'ind_idx': range(len(self.eval_pop))}
        final_params_dict.update(grouped_params_dict)

        # We need to convert them to lists or write our own custom IndividualParameter ;-)
        # Note the second argument to `cartesian_product`: This is for only having the cartesian product
        # between ``generation x (ind_idx AND individual)``, so that every individual has just one
        # unique index within a generation.
        traj.f_expand(cartesian_product(final_params_dict,
                                        [('ind_idx',) + tuple(grouped_params_dict.keys()), 'generation']))
