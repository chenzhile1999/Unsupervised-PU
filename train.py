import torch
import numpy as np
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import platform, os, copy

from utils import *
from modules.network import *
from data import *
from metrics import *

################################################# Parameter Setup #################################################
parser = ArgumentParser(description='Phase Unwrapping')
parser.add_argument('--expe_name', default='PU', type=str, help='experiment name defined by user')

parser.add_argument('--stage_num', default=3, type=int, help='number of stages in the network')

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--traindata_id', type=str, default='', help='measurement dataset name')

parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')

parser.add_argument('--optimizer', default='Adam', type=str, help='SGD | Adam | Adamw')
parser.add_argument('--scheduler', default='multistep', type=str, help='step | multistep | cosine | cycle')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--lr_step', default=50, type=int, help='epochs to decay learning rate by')
parser.add_argument('--gamma', default=0.5, type=float, help='gamma for step scheduler')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD')
parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1 of Adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 of Adam')
parser.add_argument('--eps', default=1e-8, type=float, help='eps of Adam')
parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay of optimizer')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--distil_epoch', type=int, default=200, help='epoch number of start distillation')
parser.add_argument('--end_epoch', type=int, default=700, help='epoch number of end training')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--save_interval', type=int, default=10, help='interval epochs of saving models during training')
args_Parser = Parser(parser)
args = args_Parser.get_arguments()

model_dir = "./%s/PU_lr%.4f_%s" % (args.model_dir, args.lr, args.expe_name)
log_file_name = "./%s/Log_PU_lr%.4f_%s.txt" % (args.log_dir, args.lr, args.expe_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
args_Parser.write_args(model_dir.split('/')[-1])
args_Parser.print_args()
################################################# Parameter Setup #################################################


################################################# Environment Setup #################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
################################################# Environment Setup #################################################


################################################# Network / Datasetloader / Optimizer / Scheduler Setup #################################################
model = network(args.stage_num).to(device)
model.train()

train_dataset = PUData(os.path.join('./data', args.traindata_id + '.hdf5'), aug=True)
if (platform.system() == "Windows"):
    traindata_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, pin_memory=True)
else:
    traindata_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)

optimizer_model = get_Optimizer(args, model)
scheduler_model = get_Scheduler(args, optimizer_model)
################################################# Network / Datasetloader / Optimizer / Scheduler Setup #################################################


############################### Initial ###############################
x_init = torch.ones(args.batch_size,1,256,256).to(device)
a_init = torch.zeros(*x_init.shape, 2).to(device)
############################### Initial ###############################


######################################### Training #########################################
for epoch_i in range(args.start_epoch + 1, args.end_epoch-args.distil_epoch + 1):
    print('Current epoch: {}'.format(epoch_i))
    Lsr_list = []
    gradMAE_list = []

    with tqdm(total=len(traindata_loader)) as t:
        for i, data in enumerate(traindata_loader):
            gt = data['gt'].to(device)
            WGy = data['Wrapped_Grad_y'].to(device)
            WGy_plus = data['WGy_plus'].to(device)
            WGy_minus = data['WGy_minus'].to(device)
            cond = data['std'].to(device)

            final_pred, x_preds = model(WGy_plus, cond, x_init, a_init)
            L_sr = 0
            for j, each_x in enumerate(x_preds):
                L_sr += Loss_SR(WGy_minus, each_x) / (len(x_preds) - j)

            optimizer_model.zero_grad()
            L_sr.backward()
            optimizer_model.step()

            gradMAE = Loss_SD(gt, final_pred)
            Lsr_list.append(L_sr.item())
            gradMAE_list.append(gradMAE.item())

            t.set_description('Epoch %i' % epoch_i)
            t.set_postfix(L_sr=L_sr.item(), gradMAE=gradMAE.item())
            t.update(1)

    scheduler_model.step()

    avg_L_sr = np.array(Lsr_list).mean()
    avg_gradMAE = np.array(gradMAE_list).mean()

    output_data = "[%02d/%02d] L_sr: %.4f, gradMAE: %.4f \n" % (epoch_i, args.end_epoch, avg_L_sr, avg_gradMAE)
    print(output_data)
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % args.save_interval == 0:
        torch.save(model.state_dict(), "./%s/params_dict_epoch%d.pth" % (model_dir, epoch_i))

optimizer_model = get_Optimizer(args, model)
scheduler_model = get_Scheduler(args, optimizer_model)
model_distil = copy.deepcopy(model)
model_distil.eval()

for epoch_i in range(args.end_epoch-args.distil_epoch + 1, args.end_epoch + 1):
    print('Current epoch: {}'.format(epoch_i))
    Lsd_list = []
    gradMAE_list = []

    with tqdm(total=len(traindata_loader)) as t:
        for i, data in enumerate(traindata_loader):
            gt = data['gt'].to(device)
            WGy = data['Wrapped_Grad_y'].to(device)
            WGy_plus = data['WGy_plus'].to(device)
            cond = data['std'].to(device)

            L_sd = 0
            with torch.no_grad():
                target, _ = model_distil(WGy_plus, cond, x_init, a_init)
            final_pred, x_preds_distill = model(WGy, cond, x_init, a_init)
            for j, each_x in enumerate(x_preds_distill):
                L_sd += Loss_SD(target, each_x) / (len(x_preds_distill) - j)

            optimizer_model.zero_grad()
            L_sd.backward()
            optimizer_model.step()

            gradMAE = Loss_SD(gt, final_pred)
            Lsd_list.append(L_sd.item())
            gradMAE_list.append(gradMAE.item())

            t.set_description('Epoch %i' % epoch_i)
            t.set_postfix(L_sd=L_sd.item(), gradMAE=gradMAE.item())
            t.update(1)

    scheduler_model.step()

    avg_L_sd = np.array(Lsd_list).mean()
    avg_gradMAE = np.array(gradMAE_list).mean()

    output_data = "[%02d/%02d] L_sd: %.4f, gradMAE: %.4f \n" % (epoch_i, args.end_epoch, avg_L_sd, avg_gradMAE)
    print(output_data)
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % args.save_interval == 0:
        torch.save(model.state_dict(), "./%s/params_dict_epoch%d.pth" % (model_dir, epoch_i))
######################################### Training #########################################
