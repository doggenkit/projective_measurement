import numpy as np
import tenpy
from tenpy.algorithms import tdvp
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
import copy
import sys

'''
Quench dynamics for hard-core bosons on a 1D lattice with length L.
Time evolution is performed using a hybrid 2TDVP-1TDVP approach.
The bosons are coupled to one or more ancilla pairs,
which are connected through interactions but not hopping.

Note that everything is expressed in units where the unit of
hopping t = 1 and hbar = 1.

Implementation by Elmer V. H. Doggen, based on the TeNPy code and examples.
Tested for TeNPy version 0.10.0
elmer.doggen@kit.edu

TeNPy documentation:
https://tenpy.readthedocs.io

Sample usage:
hcb_dynamics(V=0.5, meas_str=5.0, L=20, ms=4, chi=256,
             ancs=20, init_cond='init_gs')

    Arguments

    ---------

    V : the interactions between the hard-core bosons on the main chain

    meas_str : the strength of nearest-neighbour interactions between
        the ancilla(s) and the main chain

    L : the length of the main chain; the total system size will be
        Ltot = L + 2*ancs

    ms : the number of time steps before a new measurement is performed
        each time step is 0.25, consisting of 5 smaller steps of 0.05
        (delta_t * num_tdvp_steps)

    chi : the bond dimension of the MPS. Note that this example choice
        will lead to truncation! If exact simulations are desired,
        then choose chi = 2^(Ltot/2).

    ancs : the number of ancillas. Valid choices are 1, 2, L.
        1 : the ancilla is attached to the L/2'th site (1..L)
        2 : the ancillas are attached to the L/2'th and L/2+1'th sites
        L : there is an ancilla attached to every site

    init_cond : determine the initial condition. Valid choices:
        init_gs : initialize in the ground state of the uncoupled system,
            determined through DMRG
        init_wall : initialize in a domain-wall state

'''


def hcb_dynamics(V, meas_str, L, ms, chi, ancs, init_cond):

    assert (L % 2 == 0), f'L must be even, got {L}'

    assert (ancs == 1 or ancs == 2 or ancs == L), \
        f'expected a number of ancillas of 1, 2, L/2 or L, got {ancs}'

    assert (init_cond == 'init_gs' or init_cond == 'init_wall'), \
        f'expected initial condition init_gs or init_wall, got {init_cond}'

    # define time window for dynamics
    delta_t = 0.05
    num_dts = 4000
    num_tdvp_steps = 5

    # number of steps to keep measurement protocol fixed
    num_meassteps = ms
    time = np.linspace(delta_t, num_dts*delta_t, num_dts)

    # save data in this directory
    save_dir = define_output_dir(ancs, init_cond)

    Ltot = L + 2*ancs  # total system size

    # Hamiltonian parameters, arrays independent of no. of ancillas
    t = -1.e-9*np.ones(Ltot - 1)  # hopping strength
    t_proj = -1.e-9*np.ones(Ltot - 1)
    Vn = V*np.zeros(Ltot - 1)  # interaction strength
    Vn_proj = V*np.zeros(Ltot - 1)
    Vn_nomeas = V*np.zeros(Ltot - 1)
    tnn = np.zeros(Ltot - 3)  # next-next-nearest hopping strength
    tnn_proj = np.zeros(Ltot - 3)  # for use during quench
    Vnn = np.zeros(Ltot - 3)  # next-next-nearest interactions

    # determine the correct parameters for the Hamiltonian,
    # depending on the number of ancillas
    [t, Vn, tnn, Vnn] = determine_params(t, Vn, tnn, Vnn, L, ancs)

    model_params = {
        'L': Ltot,  # plus ancilla pairs
        'conserve': 'N',
        'n_max': 1,
        'lattice': 'Chain',
        'bc_MPS': 'finite',
        't': t,
        'tnn': tnn,
        'V': Vn,
        'Vnn': Vnn,
        'mu': 0.,
    }

    proj_hub = ProjModel(model_params)

    if init_cond == 'init_gs':
        # initialize the system in a half-filling product state
        product_state = ["1", "0"] * (Ltot//2)

        tnn_dmrg = copy.deepcopy(tnn)
        Vn_dmrg = copy.deepcopy(Vn)
        for i in range(L):
            Vn_dmrg = init_Vn_dmrg(Vn_dmrg, ancs, L)

        model_params_dmrg = {
            'L': Ltot,  # plus ancillas
            'conserve': 'N',
            'n_max': 1,
            'lattice': 'Chain',
            'bc_MPS': 'finite',
            't': t,
            'tnn': tnn_dmrg,
            'V': Vn_dmrg,
            'Vnn': Vnn,
            'mu': 0.,
        }

        proj_hub_dmrg = ProjModel(model_params_dmrg)
        psi = MPS.from_product_state(proj_hub_dmrg.lat.mps_sites(),
                                     product_state,
                                     bc=proj_hub_dmrg.lat.bc_MPS,
                                     form='B')

        # find ground state using DMRG
        dmrg_params = {
            'mixer': True,  # setting this to True helps to escape local minima
            'trunc_params': {
                'chi_max': 500,
                'svd_min': 1.e-13, },
            'max_E_err': 1.e-13,
            }

        info = dmrg.run(psi, proj_hub_dmrg, dmrg_params)

    if init_cond == 'init_wall':
        product_state = init_wall_state(ancs, L)
        print("initial state: ", product_state, len(product_state))
        psi = MPS.from_product_state(
            proj_hub.lat.mps_sites(), product_state,
            bc=proj_hub.lat.bc_MPS, dtype=complex)

    print("density of initial state: ")
    np.set_printoptions(suppress=True, precision=4)
    print(psi.expectation_value("N"), sum(psi.expectation_value("N")))
    print("entropy (all bipar.) \t", psi.entanglement_entropy())

    # parameters of the quench Hamiltonian
    model_params_quench = {
        'L': Ltot,  # plus ancilla pairs
        'conserve': 'N',
        'n_max': 1,
        'lattice': 'Chain',
        'bc_MPS': 'finite',
        't': t_proj,
        'tnn': tnn_proj,
        'V': Vn_proj,
        'Vnn': Vnn,
        'mu': 0.,
    }
    proj_hub_quench = ProjModel(model_params_quench)

    # parameters for dynamics (TDVP)

    # Note that we set svd_min = 1.e-30, ensuring a rapid increase of the
    # bond dimension. This causes the code to switch to the one-site TDVP
    # algorithm relatively quickly and is needed for robustness.

    tdvp_params = {
        'start_time': 0,
        'trunc_params': {
                    'chi_max': chi,
                    'svd_min': 1.e-30,
                    'trunc_cut': None
                    }}

    tdvp_params_proj = {
        'start_time': 0,
        'trunc_params': {
                    'chi_max': chi,
                    'svd_min': 1.e-30,
                    'trunc_cut': None
                    }}

    # now perform dynamics using TDVP
    times = []
    S_mid = []
    S_all = []
    density = []
    max_chi = []
    one_site_switch = False
    measured = False

    print("method: \t time: \t entropy: \t chi: \t N:")
    if init_cond == 'init_wall':
        # We take an initial tiny step, this is to create some entanglement
        # in the initial state before projection. This is needed for robustness
        # in the case of the domain-wall initial condition; the reason is the
        # presence of parts of the Hamiltonian only connected through nnn terms
        tdvp_engine = tdvp.TwoSiteTDVPEngine(
            psi=psi, model=proj_hub, options=tdvp_params)
        tdvp_engine.run_evolution(N_steps=1, dt=1.e-3)

    for i in range(num_dts//num_tdvp_steps):
        tdvp_params['start_time'] = time[i*num_tdvp_steps] - delta_t
        psi.test_sanity()

        if (i % num_meassteps == 0):
            print("dens. before proj: \t ", psi.expectation_value("N"))
            # measurements; from left to right ancilla(s)

            if ancs == 1:
                [model_params_quench['mu'], measured] = quench_func(
                    time[i*num_tdvp_steps], L//2, psi, born_rule, Ltot)
                proj_hub_quench = ProjModel(model_params_quench)
                tdvp_engine_proj = tdvp.TwoSiteTDVPEngine(
                    psi=psi, model=proj_hub_quench,
                    options=tdvp_params_proj)
                tdvp_engine_proj.run_evolution(N_steps=10, dt=1e-7*1.j*delta_t)

            if ancs == 2:
                for j in range(2):
                    [model_params_quench['mu'], measured] = quench_func(
                        time[i*num_tdvp_steps], L//2 + 3*j,
                        psi, born_rule, Ltot)
                    proj_hub_quench = ProjModel(model_params_quench)
                    tdvp_engine_proj = tdvp.TwoSiteTDVPEngine(
                        psi=psi, model=proj_hub_quench,
                        options=tdvp_params_proj)
                    tdvp_engine_proj.run_evolution(N_steps=10,
                                                   dt=1e-7*1.j*delta_t)

            if ancs == L:
                for j in range(L):
                    # measure the ancilla belonging to the j'th site
                    [model_params_quench['mu'], measured] = quench_func(
                        time[i*num_tdvp_steps], 3*j + 1, psi, born_rule, Ltot)
                    proj_hub_quench = ProjModel(model_params_quench)
                    tdvp_engine_proj = tdvp.TwoSiteTDVPEngine(
                        psi=psi, model=proj_hub_quench,
                        options=tdvp_params_proj)
                    tdvp_engine_proj.run_evolution(N_steps=10,
                                                   dt=1e-7*1.j*delta_t)

            print("dens. after proj: \t ", psi.expectation_value("N"))
            print("particle # after proj: \t", sum(psi.expectation_value("N")))
            print("bond dims. after proj: \t ", psi.chi)

        if not measured:
            model_params['V'] = Vn_nomeas
        else:
            model_params['V'] = Vn
        proj_hub = ProjModel(model_params)

        # the regular time evolution without measurements
        if not one_site_switch:
            tdvp_engine = tdvp.TwoSiteTDVPEngine(
                psi=psi, model=proj_hub, options=tdvp_params)
            tdvp_engine.run_evolution(N_steps=num_tdvp_steps, dt=delta_t)
        else:
            tdvp_engine = tdvp.SingleSiteTDVPEngine(
                psi=psi, model=proj_hub, options=tdvp_params)
            tdvp_engine.run_evolution(N_steps=num_tdvp_steps, dt=delta_t)

        # if the maximum bond dimension is reached, we switch to the 1-site
        # TDVP algorithm
        if (max(psi.chi) == chi and tdvp_engine.evolved_time > 2.0):
            one_site_switch = True

        # output quantities
        times.append(tdvp_engine.evolved_time)
        S_mid.append(psi.entanglement_entropy(bonds=[L // 2])[0])
        S_all.append(psi.entanglement_entropy())
        density.append(psi.expectation_value("N"))
        max_chi.append(max(psi.chi))
        mainchain = find_mainchain_density(psi, ancs, L)

        if one_site_switch:
            print("1TDVP \t", "%g" % times[i], "\t", S_mid[i], "\t",
                  max(psi.chi), "\t", np.sum(psi.expectation_value("N")),
                  "\t", np.sum(mainchain))
        else:
            print("2TDVP \t", "%g" % times[i], "\t", S_mid[i], "\t",
                  max(psi.chi), "\t", np.sum(psi.expectation_value("N")),
                  "\t", np.sum(mainchain))
        psi.test_sanity()

    print("final density: ")
    print(density[-1])
    # save data
    random_str = np.random.randint(1e9)

    np.save(save_dir + 'densV_%sA_%sL_%ims_%ichi_%iID_%i.npy'
            % (str(V), str(meas_str), L, ms, chi, random_str), density)
    np.save(save_dir + 'entV_%sA_%sL_%ims_%ichi_%iID_%i.npy'
            % (str(V), str(meas_str), L, ms, chi, random_str), S_mid)
    np.save(save_dir + 'entallV_%sA_%sL_%ims_%ichi_%iID_%i.npy'
            % (str(V), str(meas_str), L, ms, chi, random_str), S_all)

    return 0


def determine_params(t, Vn, tnn, Vnn, L, ancs):
    # Hamiltonian parameters, depending on the number of ancillas

    if ancs == 1:  # one ancilla at site i = L/2
        t[:] = -0.5  # hopping in main chain
        Vn[:] = V  # interaction in main chain

        tnn[L//2 - 1] = -0.5  # hopping in the main chain, across ancilla
        Vnn[L//2 - 1] = V  # interaction in the main chain, across ancilla

        Vn[L//2 - 1:L//2 + 2] = [0. for i in range(3)]
        t[L//2 - 1:L//2 + 2] = [0. for i in range(3)]

        Vn[L//2 - 1] = meas_str
        t[L//2] = -0.5  # hopping in ancilla pair

    if ancs == 2:  # two ancillas at sites i = L/2, L/2 + 1
        t[:] = -0.5  # hopping in main chain
        Vn[:] = V  # interaction in main chain

        tnn[L//2 - 1] = -0.5  # hopping in the main chain, across ancilla
        Vnn[L//2 - 1] = V  # interaction in the main chain, across ancilla
        tnn[L//2 + 2] = -0.5  # hopping in the main chain, across ancilla
        Vnn[L//2 + 2] = V  # interaction in the main chain, across ancilla

        Vn[L//2 - 1:L//2 + 5] = [0. for i in range(6)]
        t[L//2 - 1:L//2 + 5] = [0. for i in range(6)]

        Vn[L//2 - 1] = meas_str
        t[L//2] = -0.5  # hopping in ancilla pair
        Vn[L//2 + 2] = meas_str
        t[L//2 + 3] = -0.5  # hopping in ancilla pair

    if ancs == L:  # ancillas at every site
        for i in range(L-1):  # sum over bonds
            tnn[3*i] = -0.5  # hopping in main chain
            Vnn[3*i] = V  # interaction in main chain

        for i in range(L):
            Vn[3*i] = meas_str
            t[3*i+1] = -0.5  # hopping in ancilla pair

    return t, Vn, tnn, Vnn


def init_wall_state(ancs, L):
    # define a domain wall initial condition

    if ancs == 1:
        product_state = [1] * (L//2+1) + [0] * (L//2+1)
    if ancs == 2:
        product_state = [1] * (L//2+1) + [0] + [0] + [1] + [0] * (L//2)
    if ancs == L:
        product_state = ([1] + [1] + [0]) * (L//2) + ([0] + [1] + [0]) * (L//2)

    return product_state


def init_Vn_dmrg(Vn_dmrg, ancs, L):
    # disable interactions between ancillas and the main chain
    # to find the g.s. of the uncoupled system

    if ancs == 1:
        Vn_dmrg[L//2 - 1] = 0.
    if ancs == 2:
        Vn_dmrg[L//2 - 1] = 0.
        Vn_dmrg[L//2 + 2] = 0.
    if ancs == L:
        for i in range(L):
            Vn_dmrg[3*i] = 0.

    return Vn_dmrg


def define_output_dir(ancs, init_cond):
    # determine the output directory
    save_dir = './MEAS_'

    if ancs == 1:
        save_dir = save_dir + 'SINGLE_'
    if ancs == 2:
        save_dir = save_dir + 'DOUBLE_'
    if ancs == L:
        save_dir = save_dir + 'MANY_'

    if init_cond == 'init_gs':
        save_dir = save_dir + 'DMRG/'
    if init_cond == 'init_wall':
        save_dir = save_dir + 'WALL/'

    return save_dir


def find_mainchain_density(psi, ancs, L):
    # the density of the main chain depends on how the ancillas
    # are folded into the one-dimensional MPS
    dens = psi.expectation_value("N")

    if ancs == 1:
        mainchain = np.concatenate((dens[0:L//2], dens[L//2 + 2:]))
    if ancs == 2:
        mainchain = np.concatenate((np.append(dens[0:L//2],
                                    dens[L//2 + 2]), dens[L//2 + 5:]))
    if ancs == L:
        mainchain = np.zeros(L)
        for j in range(L):
            mainchain[j] = dens[3*j]

    return mainchain


def quench_func(current_time, j, psi, born_rule, Ltot):
    # auxiliary function for quenching
    # j is the measured site
    prob = 1.   # probability of "measurement"
    mu_array = np.zeros(Ltot, dtype='complex')
    rnd_array = psi.expectation_value("N")
    if current_time < 500.0:
        meas = np.random.rand()
        meas_sign = np.random.rand()
        if meas < prob:
            if born_rule and current_time > 0.01:
                # Born rule: stochastic measurement
                mu_array[j] = 3.e8*np.sign(rnd_array[j] - meas_sign)
            else:
                # always 'measure' particle
                mu_array[j] = 3.e8
            measured = True
        else:
            measured = False
        return mu_array, measured
    else:
        return mu_array, measured


###############################################################


class ProjModel(CouplingMPOModel):
    # this class defines a model of hard-core bosons on a 1D lattice
    # with on-site, nearest neighbour and next-next-nearest neighbour terms
    default_lattice = "Chain"
    force_default_lattice = True

    def init_sites(self, model_params):
        return tenpy.networks.site.BosonSite(Nmax=1, conserve='N')

    def init_terms(self, model_params):
        # read out parameters
        t = model_params.get('t', -0.5)
        tnn = model_params.get('tnn', -0.5)
        V = model_params.get('V', 0.)
        Vnn = model_params.get('Vnn', 0.)
        mu = model_params.get('mu', 0.)
        L = model_params.get('L', 1)
        n_max = model_params.get('n_max', 1)
        conserve = model_params.get('conserve', 'N')
        for u in range(len(self.lat.unit_cell)):  # on-site terms
            self.add_onsite(mu, u, 'N')
        # nearest-neighbour terms
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(t, u1, 'Bd', u2, 'B', dx, plus_hc=True)
            self.add_coupling(V, u1, 'N',  u2, 'N', dx)
        # next-nearest-neighbour terms
        for u1, u2, dx in self.lat.pairs['next_next_nearest_neighbors']:
            self.add_coupling(tnn, u1, 'Bd', u2, 'B', dx, plus_hc=True)
            self.add_coupling(Vnn, u1, 'N',  u2, 'N', dx)


##################################################################

if __name__ == "__main__":
    # the script can be run from the command line
    # if provided the following arguments
    arg_list = sys.argv

    # interaction strength
    V = float(arg_list[1])

    # measurement strength; interaction with ancilla
    meas_str = float(arg_list[2])

    # system size
    L = int(arg_list[3])

    # interval between measurements (4 steps = 1 inv. hopping unit)
    ms = int(arg_list[4])

    # bond dimension (numerical convergence parameter)
    chi = int(arg_list[5])

    # number of ancillas, 1, 2, L are supported
    ancs = int(arg_list[6])

    # initial condition: choose 'init_gs' or 'init_wall'
    init_cond = str(arg_list[7])

    ################
    born_rule = True   # stochastic measurements
    ################

    hcb_dynamics(V, meas_str, L, ms, chi, ancs, init_cond)
