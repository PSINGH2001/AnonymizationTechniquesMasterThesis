'''
This code incorporates elements from the HybridGNet project available on GitHub(https://github.com/ngaggion/HybridGNet/blob/main). 
The original code was created by Nicolas Gaggion. 
Portions of the code have been adapted and modified to suit the specific needs and requirements of this project. 
The original implementation can be accessed at HybridGNet GitHub Repository.
'''

import numpy as np
import cv2 
from networks.HybridGNet2IGSC import Hybrid 
from utils import utils
import scipy.sparse as sp
import torch

class Segmentation_Hybrid_Gnet():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hybrid = None
        if self.hybrid is None:
            self.hybrid = self.loadModel(self.device)

    def getDenseMask(self, landmarks, h, w):
        
        RL = landmarks[0:44]
        LL = landmarks[44:94]
        H = landmarks[94:]
        
        img = np.ones([h, w], dtype = 'uint8')
        
        RL = RL.reshape(-1, 1, 2).astype('int')
        
        LL = LL.reshape(-1, 1, 2).astype('int')
        H = H.reshape(-1, 1, 2).astype('int')

        img = cv2.drawContours(img, [RL], -1, (255,0,0), -1)
        img = cv2.drawContours(img, [LL], -1, (255,0,0), -1)
        img = cv2.drawContours(img, [H], -1, (255,0,0), -1)
        
        return img

    def drawOnTop(self, img, landmarks, original_shape):
        h, w = original_shape

        output = self.getDenseMask(landmarks, h, w)
        '''image = np.zeros([h, w, 3])
        image[:,:,0] = img + 0.3 * (output == 1).astype('float') - 0.1 * (output == 2).astype('float')
        image[:,:,1] = img + 0.3 * (output == 2).astype('float') - 0.1 * (output == 1).astype('float') 
        image[:,:,2] = img - 0.1 * (output == 1).astype('float') - 0.2 * (output == 2).astype('float') 

        image = np.clip(image, 0, 1)
        cv2.imwrite('/home/woody/iwi5/iwi5155h/segmented_images/drawonTop.png',image)
        RL, LL, H = landmarks[0:44], landmarks[44:94], landmarks[94:]
        '''
        
        return output
        
    def loadModel(self, device):    
        A, AD, D, U = utils.genMatrixesLungsHeart()
        N1 = A.shape[0]
        N2 = AD.shape[0]

        A = sp.csc_matrix(A).tocoo()
        AD = sp.csc_matrix(AD).tocoo()
        D = sp.csc_matrix(D).tocoo()
        U = sp.csc_matrix(U).tocoo()

        D_ = [D.copy()]
        U_ = [U.copy()]

        config = {}

        config['n_nodes'] = [N1, N1, N1, N2, N2, N2]
        A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
        
        A_t, D_t, U_t = ([utils.scipy_to_torch_sparse(x).to(device) for x in X] for X in (A_, D_, U_))

        config['latents'] = 64
        config['inputsize'] = 1024

        f = 32
        config['filters'] = [2, f, f, f, f//2, f//2, f//2]
        config['skip_features'] = f

        hybrid = Hybrid(config.copy(), D_t, U_t, A_t).to(device)
        hybrid.load_state_dict(torch.load("/home/woody/iwi5/iwi5155h/ExperimentWithoutClassifierFreeze/Experiment_0.01/test/networks/segmentation_weights.pt", map_location=torch.device(device)))
        hybrid.eval()
        
        return hybrid


    def pad_to_square(self, img):
        h, w = img.size[::-1]
        
        if h > w:
            padw = (h - w) 
            auxw = padw % 2
            img = np.pad(img, ((0, 0), (padw//2, padw//2 + auxw)), 'constant')
            
            padh = 0
            auxh = 0
            
        else:
            padh = (w - h) 
            auxh = padh % 2
            img = np.pad(img, ((padh//2, padh//2 + auxh), (0, 0)), 'constant')

            padw = 0
            auxw = 0
            
        return img, (padh, padw, auxh, auxw)
        

    def preprocess(self, input_img):
        img, padding = self.pad_to_square(input_img)
        
        h, w = img.shape[:2]
        if h != 1024 or w != 1024:
            img = cv2.resize(img, (1024, 1024), interpolation = cv2.INTER_CUBIC)
            
        return img, (h, w, padding)


    def removePreprocess(self, output, info):
        h, w, padding = info
        
        if h != 1024 or w != 1024:
            output = output * h
        else:
            output = output * 1024
        
        padh, padw, auxh, auxw = padding
        
        output[:, 0] = output[:, 0] - padw//2
        output[:, 1] = output[:, 1] - padh//2
        
        return output   


    def segment(self, input_img):
        original_shape = input_img.size[::-1]
        img, (h, w, padding) = self.preprocess(input_img)
        data = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device).float()
        
        with torch.no_grad():
            output = self.hybrid(data)[0].cpu().numpy().reshape(-1, 2)
            
        output = self.removePreprocess(output, (h, w, padding))
        output = output.astype('int')
        outseg = self.drawOnTop(input_img, output, original_shape) 
        arr = np.where(outseg == 255, 0, outseg)

        return arr

