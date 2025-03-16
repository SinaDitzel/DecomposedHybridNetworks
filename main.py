import time
import os
import argparse
import numpy as np
from logger import Logger
import traceback
import torch.nn as nn
import torch.optim as optim

from datasets.GTSRB_dataset import GTSRB
# from datasets.RESIST_dataset import RESISTSimulatedRust
# from datasets.CODEBRIM_dataset import CODEBRIM

from models.lbp import MultipleLBP
from models.rg import NormalizedRG

from models.decomposed_network import DecomposedNetwork
from models.loss_function_ocdvae import ocdvae_loss, vae_loss

from benchmark.run_model import train_model, accuracy_multiTarget, accuracy_classification, accuracy_vae, vae_val,train_classifier
from benchmark.learning_rate_scheduling import MetaQNNLrSheduler150epochs, \
    LRShedulerfromList, LrShedulerOneNet_paper, MetaQNNLrSheduler, FixedLR

from device import device

from priors import P_H0_rg_GTSRB, P_H0_LBP_5_40_GTSRB,P_H0_LBP_1_8_GTSRB,P_H0_LBP_3_24_GTSRB

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Hybrid Network')

    parser.add_argument('--data_path',type = str, default= "/data/GTSRB/", help = "path to dataset")
    parser.add_argument('--reduced_classes', action = "store_true", default = False, help ="uses a reduced set of classes for gtsrb")
    parser.add_argument('-p', '--samples_per_class', type=int, default=-1, help="max samples per class used for training")
    parser.add_argument('--seed', type=int, default =0, help='seed with which samples are choosen')
    parser.add_argument('--rg',action = 'store_true', default = False, help = "applies rg transform")
    parser.add_argument('--rgConf', action = 'store_true', default = False, help = "applies rg transform with noise Prop")
    parser.add_argument('--lbp',action = 'store_true', default = False, help = "applies lbp transform")
    parser.add_argument('--lbpConf',action = 'store_true', default = False, help = "applies lbp transform with noise prop")
    #parser.add_argument('--lambdas', type = int, default = 0, help = "lambda configs")
    #parser.add_argument('--sigma', type =float, default = None, help ="applies gaussian noise to input")
    parser.add_argument('-r', '--reconstruction',action = 'store_true', default = False, help = "reconstruct image")
    parser.add_argument('-lbphalf', '--nodecimalLBP',action = 'store_true', default = False, help = "uses binary maps instead of decimal number")
    parser.add_argument('--no_rotate', action='store_true', default=False)
    parser.add_argument('-unsup', '--unsup_train', action='store_true', default=False, help="unsupervised Training")
    parser.add_argument('--input_size', type=int, default=48, help='size input images are scaled to')
    parser.add_argument('--min_iterations_eval', type=int, default=100, help='size input images are scaled to')

    #general learning parameters
    parser.add_argument('-ep','--epochs', type = int, default = 50, help = "number of epochs networks are trained")
    parser.add_argument('-lr','--learningrate', type = float, default = 1e-3, help = "learningrate")
    parser.add_argument('-bs','--batch_size', type = int, default = 64, help = "batch size for training/default gtsrb=64, CODEBRIm =64")
    parser.add_argument('-nw','--num_workers', type = int, default = 8, help = "")
    parser.add_argument('--debug',action = 'store_true', default = False, help = "debug, less traindata")
    parser.add_argument('-t', "--tag", "--text", type = str, default ="", help = "adds a text to the runs name")
    parser.add_argument('--log_images', type=bool, default=False)

    # specific model choice:
    args = parser.parse_args()
    args.rg = args.rg or args.rgConf
    args.lbp = args.lbp or args.lbpConf


    # Network Name
    name = "%s%s%s%s%s%s%s%s%s%s" % ("gtsrb-small" if args.reduced_classes else 'gtsrb',
                                     "_rg" if args.rg else "",
                                     "Conf" if args.rgConf else "",
                                     "_lbp" if args.lbp else "",
                                     "Conf" if args.lbpConf else "",
                                     "_DCNN" + ("(2nconv)" if args.rgConf or args.lbpConf else ""),
                                     "_VAE" if args.unsup_train else "",
                                     "_OCDVAE" if args.reconstruction else "",
                                     "_p"+ str(args.samples_per_class) if args.samples_per_class>0 else "",
                                     args.tag)
    
    print("logname:", name)
    log = Logger(name)
    log.log_args(args)
    try: # to log exceptions if anything goes wrong
        decompositions = []
        if args.rg:
            decompositions.append(NormalizedRG(args.rgConf, P_H0_rg_GTSRB))
        if args.lbp:
            decompositions.append(
                MultipleLBP([(1, 8), (3, 24), (5, 40)], conf=args.lbpConf, no_decimal=args.nodecimalLBP,
                            priors=[P_H0_LBP_1_8_GTSRB, P_H0_LBP_3_24_GTSRB, P_H0_LBP_5_40_GTSRB]
                            ))
        dataset = GTSRB(root = args.data_path,
                        num_workers=args.num_workers, less_classes=args.reduced_classes,
                        decompositions=decompositions, 
                        batch_size=args.batch_size, reconstruction=args.reconstruction, unsup_train=args.unsup_train,
                        p=args.samples_per_class, p_seed= args.seed, random_rotate=(not args.no_rotate), input_size=args.input_size)

        print("created datasetinstance")

        input_data = next(iter(dataset.trainloader))[0].to(device)
        print("loaded one batch")

        net = DecomposedNetwork(dataset.num_classes, decompositions, args.batch_size, args.input_size,  vae=args.unsup_train,
                                ocdvae=args.reconstruction).to(device)
        log.add(net)
        print("created networkintance")

        if args.reconstruction:
            criterion = ocdvae_loss
        elif args.unsup_train:
            criterion = vae_loss
        else:
            criterion = nn.CrossEntropyLoss()
        
        if args.unsup_train:
            eval_model = lambda n, trainset, testset: accuracy_vae(n, trainset, testset, dataset.num_classes,
                                                                    entropies_csv=os.path.join(log.path,f'{name}_entropies.csv')) #TODO change this
        else:
            eval_model = lambda n, d, save_rec: accuracy_classification(n, d, dataset.num_classes, args.reconstruction,
                                                                entropies_csv=os.path.join(log.path,f'{name}__entropies.csv'), save_reconstruction=save_rec)
        print("created evaluate-instance")

        # VIZUALIZE NN INPUT
        log.vizualize_rg_lbp_nn_pipeline(dataset, net, -1, rg=args.rg, rgConf=args.rgConf, lbp=args.lbp,
                                        lbpConf=args.lbpConf)
        # optimizer
        if args.epochs > 50:
            remaining_ep = int(args.epochs - 50)
            lr = args.learningrate
            lrscheduler = LRShedulerfromList(
                [(50, lr), (int(remaining_ep * 0.75), lr * 0.1), (remaining_ep - int(remaining_ep * 0.75), lr * 0.01)])
        else:
            lrscheduler = FixedLR(args.learningrate)
        log.add_txt(" sheduler: "+ lrscheduler.info) 

        optimizer = optim.Adam(net.parameters(), lr=lrscheduler.lr)
        log.add_txt(" optimizer: Adam")

        # TRAINING+EVAL
        if args.unsup_train:
            best_val_loss = 100 #TODO change this
        else:
            #max_accuracy_val, _ = eval_model(net, dataset.valloader,False)
            max_accuracy_val=0
        #What to show them: Accuracy for all forms
        #Confidence Maps for Correct / False
        #Plots for Confidence Scores
        #Plots for the NN Calibration
        #Impact of Perturbation: Show the Entropy evolution and Accuracy Evolution plots for different models
    
        training_start = time.time()
        iterations = 0
        iterations_last_eval = 0
        for epoch in range(args.epochs):
            t0 = time.time()
            # TRAIN
            train_model(net, epoch, dataset.trainloader, optimizer, criterion, lrscheduler, log,
                        print_every=len(dataset.trainset) / 3, reconstruction=args.reconstruction, vae=args.unsup_train)
            
            # VISUALIZE
            iterations += len(dataset.trainloader)
            if iterations- iterations_last_eval > args.min_iterations_eval: # only do evaluation after x amounts of updates 
                iterations_last_eval = iterations
                if args.log_images:
                    log.vizualize_rg_lbp_nn_pipeline(dataset, net, epoch, rg=args.rg, rgConf=args.rgConf, lbp=args.lbp,
                                                    lbpConf=args.lbpConf)
                
                #EVALUATE
                if args.unsup_train:
                    val_loss = vae_val(net,dataset.valloader,save_reconstruction=True,criterion=criterion) #TODO fix args
                    log.add_data("Reconstruction(val)", val_loss, epoch)
                    print("epoch %i: Reconstruction(val): %.2f" % (epoch, val_loss))
                    if (val_loss < best_val_loss) or (epoch == 0):
                        best_val_loss = val_loss
                        log.save_model(net)
                else:
                    acc, _ = eval_model(net, dataset.valloader,False)
                    log.add_data("accurracy(val)", acc, epoch)
                    print("epoch %i: accuracy(val): %.2f" % (epoch, acc))
                    if (acc > max_accuracy_val) or (epoch == 0):
                        max_accuracy_val = acc
                        if hasattr(dataset, 'testloader'):
                            acc_test, _ = eval_model(net, dataset.testloader,False)
                            print("epoch %i: accuracy(test): %.2f" % (epoch, acc_test))
                            log.add_data("accurracy(test)", acc_test, epoch)
                        log.save_model(net)
                print(F'epoch {epoch} took {time.time() -t0:.2f}s')
        print(F'training took {time.time() - training_start:.2f}s')
        net.load_state_dict(log.load_model())
        # This dataloader returns images without the operators.
        dataset_vanilla = GTSRB(root = args.data_path,
                        num_workers=args.num_workers, less_classes=args.reduced_classes,
                        decompositions=[], 
                        batch_size=args.batch_size, reconstruction=args.reconstruction, unsup_train=args.unsup_train,
                        p=args.samples_per_class, p_seed= args.seed, random_rotate=(not args.no_rotate), input_size=args.input_size)
        
        if args.unsup_train:
            classifier_net = train_classifier(model=net,train_loader=dataset.trainloader,num_classes=dataset.num_classes)
            max_accuracy_val, _ = accuracy_vae(net,classifier_net,dataset.valloader,num_classes=dataset.num_classes)
            acc_test, _ = accuracy_vae(net,classifier_net,dataset.testloader,num_classes=dataset.num_classes)
            #eval_model_pert(net, dataset_vanilla.testloader)    #TODO another function here for evaluaion on perturbed data
        else:
            acc_test, _ = eval_model(net, dataset.testloader, True)
            # eval_model_pert(net, dataset_vanilla.testloader)

        log.add("best accuracy(val) :%.3f" % (max_accuracy_val))

        if hasattr(dataset, 'testloader'):
            log.add("according accurracy(test): %.3f " % (acc_test))
    except Exception as e:
        print('Exception raised', e)
        log.add(e)
        log.add(traceback.format_exc())
