from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import platform, time
from matplotlib import pyplot as plt
from utils import *
from modules.network import *
from data import *
from metrics import *

################################################# Parameter Setup #################################################
parser = ArgumentParser(description='Phase Unwrapping')
parser.add_argument('--stage_num', default=3, type=int, help='number of stages in the network')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model_dir', type=str, default='model', help='trained model directory')
parser.add_argument('--model_id', type=str, default='model', help='trained model ID')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--testdata_id', type=str, default='', help='test dataset ID')
parser.add_argument('--save', action='store_true', help='save the predictions or not')
parser.set_defaults(save=False)
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
args_Parser = Parser(parser)
args = args_Parser.get_arguments()
################################################# Parameter Setup #################################################


################################################# Environment Setup #################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
################################################# Environment Setup #################################################


################################################# Network / Datasetloader Setup #################################################
model = network(args.stage_num).to(device)
model_weights = torch.load(os.path.join(args.model_dir, args.model_id))
model.load_state_dict(model_weights, strict=True)
model.eval()

test_dataset = PUData(os.path.join('./data', args.testdata_id + '.hdf5'), aug=False)
if (platform.system() == "Windows"):
    testdata_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,pin_memory=True)
else:
    testdata_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,pin_memory=True)
################################################# Network / Datasetloader Setup #################################################


######################################## Test ########################################
ys, gts, outs, nrmses = [], [], [], []
with tqdm(total=len(testdata_loader)) as t:
    for i, data in enumerate(testdata_loader):
        WGy = data['Wrapped_Grad_y'].to(device)
        gt = data['gt'].to(device)
        cond = data['std'].to(device).float()
        x_init = torch.ones(WGy.shape[0], 1, WGy.shape[-3], WGy.shape[-2]).to(device)
        a_init = torch.zeros(*x_init.shape, 2).to(device)

        with torch.no_grad():
            pred, _ = model(WGy, cond, x_init, a_init)
        nrmse_batch, _ = NRMSE_batch(pred, gt)
        nrmses.append(nrmse_batch)

        gts.append(gt)
        ys.append(data['Wrapped'])
        outs.append(pred)

        t.set_postfix(NRMSE=torch.mean(nrmse_batch).item())
        time.sleep(0.001)
        t.update(1)

nrmse_arr = torch.cat(nrmses, dim=0).detach().cpu().numpy()
gts_arr = torch.cat(gts, dim=0).squeeze(1).detach().cpu().numpy()
outs_arr = torch.cat(outs, dim=0).squeeze(1).detach().cpu().numpy()
ys_arr = torch.cat(ys, dim=0).squeeze(1).detach().cpu().numpy()

output_data = 'Test Result on %s: avg_NRMSE: %.4f%% \n' % (args.testdata_id, np.mean(nrmse_arr))
print(output_data)
######################################## Test ########################################


############################################ Save Result ############################################
if args.save:
    save_path = './results/{}/{}'.format(args.model_id.split('/')[0], args.testdata_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'nrmses.npy'), nrmse_arr)
    for i in range(outs_arr.shape[0]):
        plt.imsave(os.path.join(save_path, '{}_pred.png'.format(i)), outs_arr[i], cmap='jet')
        plt.imsave(os.path.join(save_path, '{}_gt.png'.format(i)), gts_arr[i], cmap='jet')
        plt.imsave(os.path.join(save_path, '{}_wrapped.png'.format(i)), ys_arr[i], cmap='jet')
############################################ Save Result ############################################
