# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 15:10:54 2023

@author: CLJ
"""
import os, copy, numpy, itertools, logging, json
from . import optimizer
import service.utils as utils

class Strategy(object):
    def __init__(self,metric=""):
        self.metric=metric
        self.budget=100
        return
    def setbudget(self,budget=10,):
        self.budget = budget
    def train(self,responses:dict,labels:dict):
        raise NotImplementedError
    def evaluate(self,responses:dict,labels:dict):
        return utils.evaluate_batch(responses,labels)            
    def loadstrategy(self,strategy_path=""):
        if(1<2):
            filepath = self.strategy_path
            if(strategy_path==""):
                filepath = self.strategy_path
            else:
                filepath = strategy_path
                self.strategy_path = strategy_path

            budget = self.budget
            logging.critical("loading strategy from:{}".format(filepath))
            strategy = json.load(open(filepath))    
            data = strategy['budget'][str(budget)]
            self.thres = data['thres_list']
            self.model_ids = data['model_list']
            self.quantile = data['quantile']
            return True

        try:
            filepath = self.strategy_path
            if(strategy_path==""):
                filepath = self.strategy_path
            else:
                filepath = strategy_path
                self.strategy_path = strategy_path

            budget = self.budget
            logging.critical("loading strategy from:{}".format(filepath))
            strategy = json.load(open(filepath))    
            data = strategy['budget'][str(budget)]
            self.thres = data['thres_list']
            self.model_ids = data['model_list']
            self.quantile = data['quantile']
            return True
        except:
            print("fail to load")
            return False
        
    def savestrategy(self,strategy_path=""):
        filepath = self.strategy_path
        if(strategy_path==""):
            filepath = self.strategy_path
        else:
            filepath = strategy_path
            self.strategy_path = strategy_path
        thres_list = self.thres
        model_list = self.model_ids
        budget = self.budget
        isExist = os.path.exists(filepath)
        if(isExist):
            strategy = json.load(open(filepath))     
        else:
            strategy = dict()
            strategy['budget'] = dict()
        data = dict()
        data['thres_list'] = list(thres_list)
        data['model_list'] = list(model_list)
        data['quantile'] = list(self.quantile)
        strategy['budget'][str(budget)] = data
        #print("the strategy is",strategy)
        json_object = json.dumps(strategy, indent=4)
        # Writing to sample.json
        with open(filepath, "w") as outfile:
            outfile.write(json_object)
        return
    
class SingleAPI(Strategy):
    
    def train(self,responses:dict,labels:dict):
        return 
    def predict(self,):
        return 
    
    
class LLMChain(Strategy):
    def __init__(self,
                 metric="em_mc",
                 L_max=2,
                 strategy_path="strategy/temp2.json",):
        self.metric=metric    
        self.model_ids=["CHATGPT","GPT-4"]
        self.thres = [-1]
        self.strategy_path = strategy_path
        self.L_max = L_max
        
    def train(self,responses:dict,labels:dict,score:dict):
        # load strategy
        '''
        if(self.loadstrategy()):
            print("loading strategy successfully!")
            return 
        '''
        # train contains two stages.
        #print("start training")
        # 1. Enumerate all LLM chain and compute the results
        L_max = self.L_max
        service_ids = responses.keys()
        results = []
        for ell in range(L_max,L_max+1):
            # fix the choice of all apis in the chain
            selected_ids = list(itertools.permutations(service_ids, ell))
            # get results
            results += [ self._find_param(responses,labels,selected_id,score) for selected_id in selected_ids]

        # 2. Pick the one with the best results
        acc = -1
        model_ids = []
        thres = []
        quantile = []		
        for result in results:
            if(result['acc']>acc):
                acc = result['acc']
                model_ids = result['model_ids']
                thres = result['thres']
                quantile = result['quantile']
        self.model_ids = model_ids
        self.thres = thres
        self.quantile = quantile
        #print("finish training!")
        # 3 save 
        self.savestrategy()
        return 
    
    def _find_param(self,
                    responses,
                    labels,
                    selected_id,
					scores):
        # construct data
        L_mat, C_mat, d_mat = optimizer.construct_data(responses,
                                                       labels,
                                                       selected_id,
                                                       metric=self.metric,
													   scores=scores,
                                                       )
        # optimze results
        obj, var, qual = optimizer.optimize(L_mat,C_mat,d_mat,budget=self.budget)
        result = {"acc":obj,"model_ids":selected_id,"thres":var,"quantile":qual}
        return result
    
    def predict(self, responses:dict,scores:dict):
        base_id = self.model_ids[0]
        result = copy.deepcopy(responses[base_id])
        result['query_apis'] = dict()
        
        dist_full = [optimizer.compute_distance_batch(responses,self.model_ids[0:i+1],scores=scores) for i in range(len(self.model_ids))]
        i = 0
        for key in responses[base_id]['answer']:
            data1 = [responses[m_id]['answer'][key] for m_id in self.model_ids]
            cost1 = [responses[m_id]['cost'][key] for m_id in self.model_ids]
            dist_1 = [dist_full[j][i] for j in range(len(self.model_ids))]
            i+=1
            answer, cost, apis = self.predict_one(data1, cost1,dist_1)
            result['answer'][key] = answer
            result['cost'][key] = cost
            result['query_apis'][key] = apis
        return result
    
    def predict_one(self, data1, cost1,dist_1):
        #full_cost = cost1[0]
        #apis=[self.model_ids[0]]
        apis = []
        full_cost= 0
        for i in range(0,len(self.model_ids)-1):
            #dist = optimizer.compute_dist(data1[0:i+1])
            dist = dist_1[i]
            full_cost += cost1[i]
            apis.append(self.model_ids[i])
            if(dist<self.thres[i]):
                return data1[i], full_cost, apis
        full_cost += cost1[-1]
        apis.append(self.model_ids[-1])
        return data1[-1], full_cost, apis
    
    
    def show(self):
        print("chain models", self.model_ids)
    def getAPInames(self):
        return self.model_ids

    def reset(self):
        self.APIptr = 0

    def nextAPIandScore(self,):
        if (self.APIptr>=len(self.model_ids)):
            return None, None
        name = self.model_ids[self.APIptr]
        scorethres = self.thres[self.APIptr]
        self.APIptr+=1
        return name, scorethres
