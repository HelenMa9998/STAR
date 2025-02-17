from data import get_MSSEG
from handlers import MSSEG_Handler_2d, Prop_Handler, Prop_pseudo_Handler
from nets import Net, prop_model
from query_strategies import RandomSampling, EntropySampling, EntropySamplingDropout, BALDDropout, MarginSampling, HybridSampling, KCenterGreedy
from seed import setup_seed

# important settings
setup_seed()
params = {
    'MSSEG':
        {'n_epoch': 200,
         'train_args': {'batch_size': 32,'shuffle':True, 'num_workers': 4,'drop_last':False},
         'prop_args': {'batch_size': 1,'shuffle':False, 'num_workers': 1,'drop_last':False},
         'norm_args': {'batch_size': 32,'shuffle':True, 'num_workers': 4,'drop_last':False},
         'val_args': {'batch_size': 128,'shuffle':False, 'num_workers': 4,'drop_last':False},
         'test_args': {'batch_size': 128,'shuffle':False, 'num_workers': 4,'drop_last':False},
         'optimizer_args': {'lr': 0.001}},  
}


# Get data loader
def get_handler(name,train=False,prop=False):
    if train:
        return Prop_Handler
        # return MSSEG_Handler_2d
    if prop:
        return Prop_pseudo_Handler
    else:
        return MSSEG_Handler_2d


# Get dataset
def get_dataset(name,supervised):
    if name == 'Messidor':
        return get_Messidor(get_handler(name))
    elif name == 'MSSEG':
        if supervised == True:
            return get_MSSEG(get_handler(name),supervised = True)
        else:
            return get_MSSEG(get_handler(name))
    else:
        raise NotImplementedError


# define network for specific dataset
def get_net(name, device, prop=False):
    if name == 'Messidor':
        # return Net(Res_Net, params[name], device)
        if init==False:
            return Net(Inception_V3, params[name], device)

    elif name == 'MSSEG':
        if prop: 
            return Net(prop_model, params[name], device)
            # return Net(BraTS_model, params[name], device)

        else: 
            return Net(prop_model, params[name], device)
    else:
        raise NotImplementedError
    
# get strategies
def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "HybridSampling":
        return HybridSampling
    else:
        raise NotImplementedError
