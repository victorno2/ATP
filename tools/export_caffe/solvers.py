# Created by ay27 at 06/12/2017
import struct
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import os

# typedef enum lr_decay_type {
#     ITER = 0,
#     LOSS = 1
# } lr_decay_type;
#
# typedef enum solver_type {
#     MOMENTUM = 0,
#     SGD = 1,
#     NESTEROV = 2,
#     ADAGRAD = 3,
#     RMSPROP = 4
# } solver_type;

SolverCaffe = ['SGD', 'SGD', 'Nesterov', 'AdaGrad', 'RMSProp']


class Solver(object):
    def __init__(self, solver_type, lr, weight_decay, policy_type, policy):
        super(Solver, self).__init__()
        self.solver_type = solver_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.policy_type = policy_type
        self.policy = policy


class Momentum(Solver):
    def __init__(self, solver_type, lr, weight_decay, policy_type, policy, m):
        super(Momentum, self).__init__(solver_type, lr, weight_decay, policy_type, policy)
        self.m = m


class SGD(Solver):
    def __init__(self, solver_type, lr, weight_decay, policy_type, policy):
        super(SGD, self).__init__(solver_type, lr, weight_decay, policy_type, policy)


class NESTEROV(Solver):
    def __init__(self, solver_type, lr, weight_decay, policy_type, policy, m):
        super(NESTEROV, self).__init__(solver_type, lr, weight_decay, policy_type, policy)
        self.m = m


class ADAGRAD(Solver):
    def __init__(self, solver_type, lr, weight_decay, policy_type, policy, m):
        super(ADAGRAD, self).__init__(solver_type, lr, weight_decay, policy_type, policy)
        self.m = m


class RMSPROP(Solver):
    def __init__(self, solver_type, lr, weight_decay, policy_type, policy, eps, rms_decay):
        super(RMSPROP, self).__init__(solver_type, lr, weight_decay, policy_type, policy)
        self.eps = eps
        self.rms_decay = rms_decay


def read_solver(bin_file, fmt, size_of_type):
    # solver_type,lr,weight_decay,decay_type,policy_len
    # {{point, lr}, {point, lr}, ...}
    # param_cnt, {extra params}

    solver_type, lr, weight_decay, decay_type, policy_len, = struct.unpack(fmt * 5, bin_file.read(size_of_type * 5))
    solver_type = int(solver_type)
    decay_type = int(decay_type)
    policy_len = int(policy_len)

    print solver_type, lr, weight_decay, decay_type, policy_len

    # read policy
    policy = []
    for ii in range(policy_len):
        _point, _lr, = struct.unpack(fmt * 2, bin_file.read(size_of_type * 2))
        policy.append([_point, _lr])

    print policy

    extra_params_cnt, = struct.unpack(fmt, bin_file.read(size_of_type))

    if solver_type == 0:
        assert extra_params_cnt == 1
        m, = struct.unpack(fmt, bin_file.read(size_of_type))
        solver = Momentum(solver_type, lr, weight_decay, decay_type, policy, m)
    elif solver_type == 1:
        print 'SGD'
        assert extra_params_cnt == 0
        solver = SGD(solver_type, lr, weight_decay, decay_type, policy)
    elif solver_type == 2:
        assert extra_params_cnt == 1
        m, = struct.unpack(fmt, bin_file.read(size_of_type))
        solver = NESTEROV(solver_type, lr, weight_decay, decay_type, policy, m)
    elif solver_type == 3:
        assert extra_params_cnt == 1
        eps, = struct.unpack(fmt, bin_file.read(size_of_type))
        solver = ADAGRAD(solver_type, lr, weight_decay, decay_type, policy, eps)
    elif solver_type == 4:
        assert extra_params_cnt == 2
        eps, rms_decay, = struct.unpack(fmt * 2, bin_file.read(size_of_type * 2))
        solver = RMSPROP(solver_type, lr, weight_decay, decay_type, policy, eps, rms_decay)
    else:
        raise ValueError('unrecognized solver type %d' % solver_type)

    return solver


class LineTxt(object):
    def __init__(self):
        self.inner_str = ''

    def add_line(self, line):
        self.inner_str += line + '\n'

    def raw(self):
        return self.inner_str


def gen_solver(solver, out_dir, prefix):
    solver_txt = LineTxt()
    solver_txt.add_line('# Auto-generated by ExportCaffe tool. Powered by ATP.')
    solver_txt.add_line('# To use generated prototxt and caffemodel, you must specify data path in data layer.')
    solver_txt.add_line('train_net: "' + os.path.join(out_dir, prefix + '_train.prototxt') + '"')
    solver_txt.add_line('test_net: "' + os.path.join(out_dir, prefix + '_test.prototxt') + '"')
    solver_txt.add_line('type: "' + SolverCaffe[solver.solver_type] + '"')
    solver_txt.add_line('base_lr: %f' % solver.lr)
    solver_txt.add_line('weight_decay: %f' % solver.weight_decay)
    if solver.policy_type == 0 and len(solver.policy) > 0:
        # decay in Iter
        solver_txt.add_line('lr_policy: "multistep"')

        gamma_line = 'gamma: %f' % (solver.policy[1][1] / solver.policy[0][1])
        multi_lrs = ' # source lr:'
        for ii in range(len(solver.policy)):
            multi_lrs += ' %f' % solver.policy[ii][1]
        solver_txt.add_line(gamma_line + multi_lrs)

        for ii in range(len(solver.policy)):
            solver_txt.add_line('stepvalue: %d' % int(solver.policy[ii][0]))
    elif solver.policy_type == 1:
        # decay in Loss
        lr_line1 = '# TODO: source lr policy is loss value sensitive, we can not determine which lr policy to use.'
        lr_line2 = '# The source lr policy is:'
        lr_line3 = '# lr: \t'
        lr_line4 = '# loss point: \t'
        for ii in range(len(solver.policy)):
            lr_line3 += ' \t%f' % solver.policy[ii][1]
            lr_line4 += ' \t%f' % solver.policy[ii][0]

        solver_txt.add_line(lr_line1)
        solver_txt.add_line(lr_line2)
        solver_txt.add_line(lr_line3)
        solver_txt.add_line(lr_line4)
    else:
        solver_txt.add_line('lr_policy: "fixed"')
    solver_txt.add_line('solver_mode: GPU')
    solver_txt.add_line('test_iter:         # TODO: test iter w.r.t validation set')
    solver_txt.add_line('test_interval:     # TODO: set test interval')
    solver_txt.add_line('display:           # TODO: display interval')
    solver_txt.add_line('max_iter:          # TODO: max iteration to train')
    solver_txt.add_line('snapshot:          # Optional: take snapshot interval')
    solver_txt.add_line('snapshot_prefix:   # Optional: snapshot files prefix')

    return solver_txt.raw()
