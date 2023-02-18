from utils import read_json, data_split, read_csv_copd
from model_wrapper import CNN_Wrapper, FCN_Wrapper
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True

from dataloader import FCN_Data
from torch.utils.data import Dataset, DataLoader
from model import _FCN
from torchsummary import summary

def cnn_main(seed):
    cnn_setting = config['cnn']
    for exp_idx in range(repe_time):
        cnn = CNN_Wrapper(fil_num         = cnn_setting['fil_num'],
                          drop_rate       = cnn_setting['drop_rate'],
                          batch_size      = cnn_setting['batch_size'],
                          balanced        = cnn_setting['balanced'],
                          Data_dir        = cnn_setting['Data_dir'],
                          exp_idx         = exp_idx,
                          seed            = seed,
                          model_name      = 'cnn',
                          metric          = 'accuracy')

        cnn.train(lr     = cnn_setting['learning_rate'],
                  epochs = cnn_setting['train_epochs'])
        cnn.test()
        cnn.gen_features()


def fcn_main(seed):
    fcn_setting = config['fcn']
    for exp_idx in range(repe_time):
        fcn = FCN_Wrapper(fil_num        = fcn_setting['fil_num'],
                        drop_rate       = fcn_setting['drop_rate'],
                        batch_size      = fcn_setting['batch_size'],
                        balanced        = fcn_setting['balanced'],
                        Data_dir        = fcn_setting['Data_dir'],
                        patch_size      = fcn_setting['patch_size'],
                        exp_idx         = exp_idx,
                        seed            = seed,
                        model_name      = 'fcn',
                        metric          = 'accuracy')
        fcn.train(lr     = fcn_setting['learning_rate'],
                  epochs = fcn_setting['train_epochs'])
        fcn.test_and_generate_DPMs()


if __name__ == "__main__":

    config = read_json('./config.json')
    seed, repe_time = 1000, config['repeat_time']  # if you only want to use 1 data split, set repe_time = 1
    # data_split function splits ADNI dataset into training, validation and testing for several times (repe_time)
    #data_split(repe_time=repe_time)


    read_csv_copd("/Users/qishi/Desktop/ai/brain2020/lookupcsv/ADNI.csv")

    # train_data = FCN_Data("/Users/qishi/Desktop/ai/brain2020/outputs/step4/", 0, 
    #                       whole_volume=False, stage='train',seed=1000, patch_size=47)
    # train_dataloader = DataLoader(train_data, batch_size=1,  shuffle=False, drop_last=True)

    # with torch.no_grad():
    #     for idx, (inputs, labels) in enumerate(train_dataloader):
    #         inputs, labels = inputs.cpu(), labels.cpu()

    #         print("batch_data2 shape:", inputs.shape)
    #         print("data:", inputs[0,0,0,0])
    #         # fcn = _FCN(3, 0.5)
    #         fcn = _FCN(3, 0.5)
    #         summary(fcn,(1, 47,47,47),batch_size=-1,device="cpu")
    #         break
            

    # train_data = FCN_Data("/Users/qishi/Desktop/ai/brain2020/outputs/step4/", 0, stage='train', whole_volume=True, seed=1000, patch_size=47)
    # train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)


    # with torch.no_grad():
    #     for idx, (inputs, labels) in enumerate(train_dataloader):
    #         inputs, labels = inputs.cpu(), labels.cpu()

    #         print("batch_data shape:", inputs.shape)
    #         print("data:", inputs[0,0,0,0])
    #         # fcn = _FCN(3, 0.5)
    #         # DPM = _FCN(3, 0.5).dense_to_conv()(inputs, stage='inference')
    #         # print("DPM.shape:",DPM.shape)
    #         break
    #         # net =  nn.Sequential(
    #         #     *(list(fcn.features)+ list(fcn.classifier)))
    #         # summary(fcn,(1,47,47,47),batch_size=-1,device="cpu")
            



    # cnn_main(seed)
    #fcn_main(seed)
    # with torch.device('cpu'):
    #     fcn_main(seed)    

    
    # with torch.cpu():
    #     cnn_main(seed)


    # # to perform FCN training #####################################
    # with torch.cuda.device(2):  # specify which gpu to use
    #     fcn_main(seed)  # each FCN model will be independently trained on the corresponding data split

    # # to perform CNN training #####################################
    # with torch.cuda.device(2): # specify which gpu to use
    #     cnn_main(seed)  # each CNN model will be independently trained on the corresponding data split
        



