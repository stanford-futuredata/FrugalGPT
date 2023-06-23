import numpy, logging, scipy
import service.utils as utils
#from scipy import optimize

eps = 1e-3


def compute_dist(answers, scores, query=None):
    dist = 1-scores[-1]
    #print("scores is:",scores)
    return dist

def compute_distance_batch(responses,selected_id,scores):
    numpy.random.seed(2023)
    #print("scores are",scores['GPT-4']['8606'])
    answer_list = [ [responses[id1]['answer'][key] for id1 in selected_id ] for key in responses[selected_id[0]]['answer']]
    score_list = [ [scores[id1][key] for id1 in selected_id ] for key in responses[selected_id[0]]['answer']]
    dists = [compute_dist(answer_list[i],scores=score_list[i]) for i in range(len(answer_list))]
    return dists
    

def compute_loss_and_cost_batch(response,labels,metric):
    results_eval = utils.evaluate_batch(response,labels)
    costs = results_eval['cost_list']
    loss = results_eval[metric+"_list"]
    return loss, costs

def construct_data(responses,
                   labels,
                   selected_id,
                   metric,
				   scores,
                   ):
    # L: n by d matrix, loss
    # C: n by d matrix, cost up to chain pos
    # d: n by d matrix, distance up to chain pos
    n = len(labels)
    d = len(selected_id)
    L_mat = numpy.zeros((n,d))
    C_mat = numpy.zeros((n,d))
    d_mat = numpy.zeros((n,d))
    # compute L
    for idx, id1 in enumerate(selected_id):
        L_mat[:,idx], C_mat[:,idx] = compute_loss_and_cost_batch(responses[id1],labels,metric)
    # compute C
    for i in range(1,len(selected_id)):
        C_mat[:,i]+=C_mat[:,i-1]
    # compute d
    for idx, id1 in enumerate(selected_id):
        d_mat[:,idx] = compute_distance_batch(responses,selected_id[0:idx+1],scores=scores)
    
    logging.debug("Showing the average (acc) of the L_mat")
    logging.debug(numpy.average(L_mat,0))
    logging.critical("construct data for {}".format(selected_id))
    #logging.critical("SHOW Distance mat",d_mat[0:20,])
    return L_mat,C_mat,d_mat

def optimize(L_mat,C_mat,d_mat,budget):
    #return 0.1, [0,1,1,1]
    if(numpy.average(C_mat[:,0])>budget):
        logging.critical("Base API too expensive, skip")
        return -999, [], []
    mask_full = numpy.full(L_mat.shape[0],True)
    def f(delta):
        #delta.append(1.0)
        #delta_0 = numpy.append(delta,1)
        e_full = (d_mat < delta)
        #print("shape of e_full",e_full.shape)
        for i in range(e_full.shape[0]):
            has_accept = False
            for j in range(e_full.shape[1]):
                if(has_accept):
                    e_full[i,j] = False
                else:
                    if(e_full[i,j]==1):
                        has_accept=True
            
        #print("e_full sum:",numpy.sum(e_full,1))
        #delta.pop()       
        acc = numpy.sum(numpy.multiply(e_full,L_mat))       
        cost = numpy.sum(numpy.multiply(e_full,C_mat))
        if(cost>budget*len(L_mat)):
            return 10000
        return -acc

    def g(qual):
        if(numpy.all(numpy.diff(qual) >= 0)==False):
            return 10000
        #logging.info("qual value is {}".format(qual))
        thres = quatile2thres_batch(qual)
        #thres = [-0.1,-0.1,0.2,1]
        logging.info("qual and thres value are:{} and {}".format(qual, thres))        
        return f(thres)

    def quatile2thres(q,i,mask_last):
        data = d_mat[mask_last,i]
        logging.debug("data is",i, data)
        q = max(min(q,1),0)
        thres1 = numpy.quantile(data, 1-q)
        masknew = d_mat[:,i]>=thres1
        masknew = numpy.logical_and(mask_last,masknew)
        return thres1, masknew

    def quatile2thres_batch(qual):
        thres = numpy.zeros(L_mat.shape[1])
        thres[-1] = 1
        mask_last = mask_full
        for i in range(0,len(thres)-1):
            # map quatile to the thres
            qi = qual[i]
            thres[i],mask_last = quatile2thres(qi,i,mask_last)
        return thres
    #delta_ranges = [(-0.1,1.01)]*(L_mat.shape[1]-1)
    #delta_ranges[-1] = (1.0,1.0)
    #d1 = [-0.1, -0.1, 0.05,1]
    #print("Test of the function f:",f(d1)/len(L_mat),d1)
    #print("delta range",delta_ranges, L_mat.shape)
    qual_ranges = [(1e-5,1-1e-5)]*(L_mat.shape[1]-1)
#    qual_ranges = [[0.1,0.5,0.95]]*(L_mat.shape[1]-1)
    logging.debug("start searching")
    

#    optimize.brute()
    resbrute = scipy.optimize.brute(g, 
                              qual_ranges, 
                              #args=params, 
                              full_output=True,
                              finish=scipy.optimize.fmin,
                              #workers=2,
                              Ns=40,
                              )
    logging.critical("the obj is {} and the var is {}".format(resbrute[1],resbrute[0]))
    #time.sleep(10000)
    thres_final = quatile2thres_batch(resbrute[0])
    return -resbrute[1]/len(L_mat), thres_final, resbrute[0]
        