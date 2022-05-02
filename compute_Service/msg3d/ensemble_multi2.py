import argparse
import pickle
import os
import sys

import numpy as np
import itertools

from tqdm import tqdm

import pandas as pd

def getCombinations(num_elements):
    print ('getCombinations: {}'.format(num_elements)) 
    list_values= np.round(np.arange(0, 1.02, 0.02), 2).tolist() 
    list_values_r= np.round(np.arange(0, 0.52, 0.02), 2).tolist()
    list_values_r2= np.round(np.arange(0, 0.40, 0.02), 2).tolist()
    list_values_r3= np.round(np.arange(0, 0.20, 0.02), 2).tolist()
    
    try:
        if (num_elements==1):
            combis = list(itertools.product(list_values))
        elif (num_elements==2):
            combis = list(itertools.product(list_values, list_values))
        elif (num_elements==3):
            combis = list(itertools.product(list_values, list_values, list_values))
        elif (num_elements==4): 
            combis = list(itertools.product(list_values_r, list_values_r, list_values_r, list_values_r))
        elif (num_elements==5): 
            combis = list(itertools.product(list_values_r, list_values_r, list_values_r, list_values_r, list_values_r))
        elif (num_elements==8): 
            combis = list(itertools.product(list_values_r3, list_values_r3, list_values_r3, list_values_r3, list_values_r3,list_values_r3,list_values_r3,list_values_r3))
    except:
        print ('more than 4 combinations not implemented yet.')
        sys.exit(1) 
    
    return combis

def sum_elements(arr_in):
    sum = 0
    for value in arr_in:
        sum +=value
    return np.round(sum,2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--label_folder',
                        default='/home/temporal2/mvazquez/skeleton_action_recognition/data_autsl/autsl/labels',
                        help='the work folder for storing results')
    
    parser.add_argument('--set',
                        required=True,
                        choices={'val', 'test', 'test_movil', 'test_collaborations'},
                        help='the work folder for storing results')
   
    parser.add_argument('--inputs',
                        required=True,
                        nargs=2,                  ## set how many results will be emsembled
                        help='Directory containing "epoch1_test_score.pkl" for bone eval/test results')
    
    parser.add_argument('--out',
                        default=None,
                        help='out folder combination')
    
    parser.add_argument('--csv',
                        action='store_true',
                        default=False,
                        help='Generate or not predictions.csv')
    
    parser.add_argument('--csv5',
                        action='store_true',
                        default=False,
                        help='Generate or not predictions.csv')
    
    parser.add_argument('--search',
                        action='store_true',
                        default=False,
                        help='Generate or not predictions.csv')

    parser.add_argument('--weights',
                        default=None,
                        nargs=2,
                        type=float,
                        help='Generate or not predictions.csv')


    arg = parser.parse_args()

    label_folder = arg.label_folder
    set = arg.set
    search = arg.search
    
    print (label_folder)
    print (set)
    
    inputs = arg.inputs
    print (arg.inputs)
    for input in inputs:
        print (input)

    with open(label_folder + '/'+set+'_label.pkl', 'rb') as label:
        label = np.array(pickle.load(label))
        
    if (arg.search == True):
        weigths = getCombinations(len(inputs))
    else:
        print ('search false')
        print ('default weights')
        # weigths  = [[1]]
        # weigths  = [[0.5,0.5]]
        # weigths  = [[0.33333,0.33333,0.33333]]
        weigths  = [[0.25,0.25,0.25,0.25]]
        # weigths  = [[0.2,0.2,0.2,0.2,0.2]]  
        # weigths = [[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]]

    if (arg.weights != None):
        print ('LOAD WEIGHTS')
        print ('WEIGTHS: {}'.format(arg.weights))
        weigths  = [arg.weights]

        print ('weight[0]: {}'.format(weigths[0]))
        print ('type weight[0]: {}'.format(type(weigths[0])))
    else:
        print ('NO LOAD WEIGHTS')
        
    best_result = np.zeros(3)   ## [acc, w_1, w_2]
    best_acc = 0
    best_acc5 = 0
    best_weigths = []
    idx_wrong_top1 = []
    
    process = tqdm(weigths, dynamic_ncols=True)
    for weigth in process:
        if (sum_elements(weigth)==1):
            arr_out = np.zeros((len(label[0]),82))
            arr_predictions = []
            arr_predictions_top5 = []
            arr_top1 = []
            right_num = total_num = right_num_5 = 0
            
            for idx, input in enumerate(inputs):
                with open(os.path.join(input), 'rb') as r_data:
                    r_data = list(pickle.load(r_data).items())     
                for i in range(len(label[0])):
                    arr_out[i]+=r_data[i][1] * weigth[idx]
                    
            for i in range(len(label[0])):
                _, l = label[:, i]
                rank_5 = arr_out[i].argsort()[-5:]
                right_num_5 += int(int(l) in rank_5)
                r = np.argmax(arr_out[i])
                right_num += int(r == int(l))
                arr_top1.append(int(r == int(l)))
                total_num += 1
                arr_predictions.append(r)
                arr_predictions_top5.append(rank_5)

                if ((r == int(l))==0):
                    idx_wrong_top1.append(int(i))
        
            acc = right_num / total_num
            acc5 = right_num_5 / total_num
                
            # print ('weigth: {}'.format(weigth))    
            # print('Top1 Acc: {:.4f}%'.format(acc * 100))
            # print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
            
            if (acc>=best_acc):
                best_acc = acc
                best_acc5 = acc5
                best_weigths = weigth
            
    print ('Final. Best results:')
    print ('weights: {} > Top1: {} & Top5: {}'.format(best_weigths,best_acc, best_acc5))
    
    if (arg.out != None):
        
        with open(os.path.join(input, 'epoch1_test_score.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        for idx, key in enumerate(list(data.keys())):
            data[key] = arr_out[idx]
        
        with open('{}ensemble_score.pkl'.format(arg.out), 'wb') as f:
            pickle.dump(data, f)
            
        print ("Save out pickle: {}ensemble_score.pkl".format(arg.out))  
        
        
        results = np.vstack((list(data.keys()), arr_top1))
        
        results = results.transpose()
        pd.DataFrame(results).to_csv('{}arr_top1.csv'.format(arg.out),index=False,header=False)
        print ('Save out csv: {}arr_top1.csv'.format(arg.out))  
        
    
    if (arg.csv == True):
        
        print ("Generating predictions.csv file")
        with open(input, 'rb') as f:
            data = pickle.load(f)

        arr_names = []
        for name in list(data.keys()):
            #arr_names.append(name[:-11])
            arr_names.append(name)
            
        results = np.vstack((arr_names, arr_predictions))
        results = results.transpose()
        pd.DataFrame(results).to_csv("predictions.csv",index=False,header=False)

    if (arg.csv5 == True):
        
        print ("Generating predictions.csv file")
        with open(input, 'rb') as f:
            data = pickle.load(f)

        arr_names = []
        for name in list(data.keys()):
            #arr_names.append(name[:-11])
            arr_names.append(name)
            
        #print(arr_predictions_top5)
        
        arr = np.array(arr_predictions_top5)
        top1 = arr[:,4]
        top2 = arr[:,3]
        top3 = arr[:,2]
        top4 = arr[:,1]
        top5 = arr[:,0]
        # results = np.vstack((arr_names, top1, top2, top3, top4, top5))

        arr_names_filter = [arr_names[i] for i in idx_wrong_top1]
        label_filter = [label[:, i][1] for i in idx_wrong_top1]
        top1_filter = [top1[i] for i in idx_wrong_top1]
        top2_filter = [top2[i] for i in idx_wrong_top1]
        top3_filter = [top3[i] for i in idx_wrong_top1]
        top4_filter = [top4[i] for i in idx_wrong_top1]
        top5_filter = [top5[i] for i in idx_wrong_top1]

        results = np.vstack((arr_names_filter, label_filter, top1_filter, top2_filter, top3_filter, top4_filter, top5_filter))
        results = results.transpose()
        pd.DataFrame(results).to_csv("predictions_top5.csv",index=False,header=False)

    
    # print (arr_out.shape)
    # print (len(arr_predictions))


    #print ('idx_wrong')
    #print (len(idx_wrong_top1))
    #print (idx_wrong_top1)
