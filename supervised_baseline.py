import torch
from utils import get_dataset, get_net, get_strategy, get_handler
from data import Data
from config import parse_args

from seed import setup_seed
# fix random seed
# setup_seed(42)
#supervised learning baseline
args = parse_args()
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# get dataset
eval_handler = get_handler(args.dataset_name) # 用于val 与 test 以及baseline 单张为单位

X_train, Y_train, X_val, Y_val, X_test, Y_test, train_num_slices_per_patient, val_num_slices_per_patient, test_num_slices_per_patient = get_dataset(args.dataset_name,supervised=False)

dataset = Data(X_train, Y_train, X_val, Y_val, X_test, Y_test)
dataset.supervised_training_labels()
_, norm_train_loader = dataset.get_labeled_data(eval_handler, pseudo_idxs=None)
norm_val_data = dataset.get_val_data(eval_handler)

print(f"number of testing pool: {dataset.n_test}")
print()
# get network
net = get_net(args.dataset_name, device)

# start supervised learning baseline

net.supervised_val_loss(norm_train_loader,norm_val_data,rd=0)
test_preds,targets = net.predict(dataset.get_test_data(eval_handler),test_num_slices_per_patient)
print(f"testing dice: {dataset.cal_test_acc(test_preds,targets)}")